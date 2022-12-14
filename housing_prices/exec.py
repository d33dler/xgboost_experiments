import time
from argparse import ArgumentParser
from typing import Tuple, List

import geopy.distance
import matplotlib.pyplot as plt
import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
import xgboost as xgb
import yaml
from easydict import EasyDict
from hyperopt import hp, STATUS_OK
from hyperopt import fmin, tpe
from hyperopt.pyll import scope
from sklearn.feature_extraction import FeatureHasher
from sklearn.metrics import mean_absolute_error, roc_auc_score
from sklearn.model_selection import train_test_split  # Perforing grid search
from xgboost import XGBModel
import mlflow


def load_data(data: List[str]):
    return [pd.read_csv(d, header=0) for d in data]


def transform_data(data):
    pass


def clean_dataset(ax: plt.Axes, data: pd.DataFrame, cols: List[str], intervals: List[Tuple[float, float]]):
    ax.set_xlabel(cols[0])
    #ax.set_ylabel(cols[1])
    data = data[(data[cols[0]] > intervals[0][0]) & (data[cols[0]] < intervals[0][1])]
    #data = data[(data[cols[1]] > intervals[1][0]) & (data[cols[1]] < intervals[1][1])]
#    ax.scatter(data[cols[0]], data[cols[1]], c="green", marker="s")
    return data


def normalize(data: pd.DataFrame):
    """
    Normalization is not needed for xgboost, so use only for experimental purposes
    Parameters
    ----------
    data

    Returns
    -------

    """
    num_cols = [col for col in data.columns if data[col].dtype not in [object, bool, str, 'category']]
    data[num_cols] = np.log1p(data[num_cols])


def gen_xgd_input():
    pass


def make_categorical(n_samples: int, n_features: int, n_categories: int, onehot: bool) -> \
        Tuple[pd.DataFrame, pd.Series]:
    """Make some random data for demo."""
    rng = np.random.RandomState(1994)

    pd_dict = {}
    for i in range(n_features + 1):
        c = rng.randint(low=0, high=n_categories, size=n_samples)
        pd_dict[str(i)] = pd.Series(c, dtype=np.int64)

    df = pd.DataFrame(pd_dict)
    label = df.iloc[:, 0]
    df = df.iloc[:, 1:]
    for i in range(0, n_features):
        label += df.iloc[:, i]
    label += 1
    df = df.astype("category")
    categories = np.arange(0, n_categories)
    for col in df.columns:
        df[col] = df[col].cat.set_categories(categories)

    if onehot:
        return pd.get_dummies(df), label
    return df, label


def load_config(config):
    with open(config, mode="r") as f:
        cfg = yaml.load(f, Loader=yaml.SafeLoader)
    return EasyDict(cfg)


def change_type(data: List[pd.DataFrame], label: str, dtype: str):
    update = [d[label].astype(dtype) for d in data]
    for d, U in zip(data, update):
        d[label] = U


def update_cols(data: List[pd.DataFrame], newcols: List[Tuple[pd.DataFrame, str]], drop_labels: List[str]):
    output = []
    for d, (new_c, col_id) in zip(data, newcols):
        d.drop(drop_labels, axis=1, inplace=True)
        df = pd.DataFrame(new_c.toarray(), columns=[f'F_{col_id}_{i}' for i in range(0, new_c.shape[1])])
        output.append(pd.concat([d, df], axis=1))
    return output


def optimize_hyperparams(train_X, train_Y):
    search_space = {
        'learning_rate': hp.loguniform('learning_rate', 0.1, 0.7),
        'max_depth': scope.int(hp.uniform('max_depth', 1, 18)),
        'eta': hp.quniform('eta', 0.01, 1, 0.1),
        'min_child_weight': hp.loguniform('min_child_weight', 1, 5),
        'subsample': hp.uniform('subsample', 0.5, 1),
        'colsample_bytree': hp.uniform('colsample_bytree', 0.5, 1),
        'gamma': hp.loguniform('gamma', 1, 10),
        'alpha': hp.loguniform('alpha', 0.1, 5),
        'lambda': hp.loguniform('lambda', 0.1, 5),
        'objective': 'reg:squarederror',
        'eval_metric': 'mae',
        'seed': 123,
    }

    X_train, X_test, y_train, y_test = train_test_split(train_X, train_Y, test_size=0.25, random_state=666)

    train = xgb.DMatrix(data=X_train, label=y_train)
    test = xgb.DMatrix(data=X_test, label=y_test)

    def train_model(params):
        # With MLflow autologging, hyperparameters and the trained model are automatically logged to MLflow.
        mlflow.xgboost.autolog(silent=True)

        # However, we can log additional information by using an MLFlow tracking context manager
        with mlflow.start_run(nested=True):
            start_time = time.time()
            booster = xgb.train(params=params, dtrain=train, num_boost_round=2048, evals=[(test, "test")],
                                early_stopping_rounds=20, verbose_eval=False)
            run_time = time.time() - start_time
            mlflow.log_metric('runtime', run_time)

            predictions_test = booster.predict(test)
            mae_score = mean_absolute_error(y_test, predictions_test)

            return {'status': STATUS_OK, 'loss': mae_score, 'booster': booster.attributes()}

    with mlflow.start_run(run_name='xgb_loss_threshold'):
        best_params = fmin(
            fn=train_model,
            space=search_space,
            algo=tpe.suggest,
            loss_threshold=75,  # stop the grid search once we've reached an AUC of 0.92 or higher
            timeout=60 * 10,  # stop after 5 minutes regardless if we reach an AUC of 0.92
            # trials=spark_trials,
            rstate=np.random.default_rng(666)
        )
    return best_params


def update_distances(data, cities_ds):
    if len(cities_ds[cities_ds.city == data["city"]]) == 1:
        city = cities_ds[cities_ds.city == data["city"]].iloc[0]
        return geopy.distance.geodesic((city["lat"], city["lng"]), (data["latitude"], data["longitude"])).km
    else:
        return 1


def get_outliers(train_ds, col, ds_cfg):
    majority = train_ds.groupby(col).filter(lambda x: len(x) >= ds_cfg.OUTLIER_THRESHOLD)  # requires optimization
    outliers = train_ds.groupby(col).filter(lambda x: len(x) < ds_cfg.OUTLIER_THRESHOLD)  # requires optimization
    return majority, outliers


def execute(_argz):
    """
    Ugliest method I wrote so far.
    Parameters
    ----------
    _argz :

    Returns
    -------

    """
    cfg = load_config(_argz.config)  # not used
    ds_cfg = load_config(_argz.ds_config)
    cities_ds = load_data([_argz.ds_cities])[0]
    train_ds, testing_ds = load_data([_argz.train_ds, _argz.test_ds])

    ##################################
    print(ds_cfg)
    y_target = ds_cfg.TARGET

    print("-------------FILLING MISSING DATA-------------")
    train_ds.fillna("Missing", inplace=True)
    testing_ds.fillna("Missing", inplace=True)

    for col in ds_cfg.OUTLIER_OCCURANCES:
        majority, outliers = get_outliers(train_ds, col, ds_cfg)
        test_maj, test_outl = get_outliers(testing_ds, col, ds_cfg)
        if outliers[col].dtype == object:
            outliers[col] = test_outl[col] = 'Missing'
        else:
            outliers[col] = test_outl[col] = 0
        train_ds = pd.concat([majority, outliers], ignore_index=True)
        testing_ds = pd.concat([test_maj, test_outl], ignore_index=True)
    [print(train_ds[col].unique()) for col in ds_cfg.OUTLIER_OCCURANCES]

    ###########################################################################
    # ADDING EXTRA FEATURE - Geodesic distance to city center
    train_ds["dist_cc"] = train_ds.apply(lambda x: update_distances(x, cities_ds), axis=1)
    testing_ds["dist_cc"] = testing_ds.apply(lambda x: update_distances(x, cities_ds), axis=1)
    ###########################################################################

    testings_ids = testing_ds["id"].copy()

    ##########################################################################
    # DROPPING IRRELEVANT FEATURES
    train_ds.drop(ds_cfg.EXCLUDE, axis=ds_cfg.EXCLUDE_AXIS, inplace=True)
    testing_ds.drop(ds_cfg.EXCLUDE, axis=ds_cfg.EXCLUDE_AXIS, inplace=True)

    ##########################################################################
    # OUTLIER DATASET CLEANING
    scatters = plt.figure()
    scatter_axes = scatters.add_subplot(111)
    clean_dataset(scatter_axes, train_ds, [o.ID for o in ds_cfg.OUTLIERS], [o.INTERVAL for o in ds_cfg.OUTLIERS])

    cats_dict = {o.ID: o for o in ds_cfg.CATS}
    split = [train_ds, testing_ds]
    print("---------HASHING FEATURES & SETTING CATEGORICAL DATA----------")
    for label in train_ds.columns:
        if train_ds[label].dtype == object:
            if label in cats_dict:
                match cats_dict[label].TYPE:
                    case "HASHING":
                        change_type(split, label=label, dtype='str')
                        hashing = FeatureHasher(n_features=len(train_ds[label].unique()), input_type='string')
                        split = update_cols(split, [(hashing.transform(d[label].copy().values), label) for d in split],
                                            [label])
                        train_ds, testing_ds = split
                    case _:
                        change_type(split, label=label, dtype='category')
            else:
                change_type(split, label=label, dtype='category')

    ########################################################
    # SPLIT DATASET SOURCE(X) AND TARGET(Y) FEATURES
    train_X = train_ds.drop([y_target], axis=ds_cfg.EXCLUDE_AXIS, inplace=False)
    train_Y = train_ds[ds_cfg.TARGET]
    train_X = pd.get_dummies(train_X)
    testing_ds = pd.get_dummies(testing_ds)

    print(train_X.info(verbose=True))
    ######################################################
    # HYPER-PARAM FINE TUNING

    # best_params = optimize_hyperparams(train_X, train_Y)
    # print(best_params)

    #######################################################
    # OPTIONAL DATA SPLIT
    # X_train, X_test, y_train, y_test = train_test_split(train_X, train_Y, test_size=0.5, random_state=1337)

    reg: XGBModel = xgb.XGBRegressor(tree_method="hist",
                                     objective='reg:squarederror',
                                     enable_categorical=True,
                                     eval_metric='mae',
                                     max_cat_to_onehot=5,
                                     max_depth=17,
                                     min_child_weight=1.5,
                                     gamma=4,
                                     eta=0.01,
                                     learning_rate=0.04,
                                     subsample=1,
                                     reg_lambda=0.5,
                                     seed=666,
                                     n_estimators=1024,
                                     early_stoping_rounds=5,
                                     colsample_bytree=0.8)

    ################################################################
    # MODEL TRAINING
    reg.fit(train_X, train_Y, eval_set=[(train_X, train_Y)])

    ################################################################
    # DEBUGGING, EVALUATION & PLOTTING
    ################################################################
    # print(reg.get_booster().get_score(importance_type='gain'))
    # [print(o) for o in sorted(reg.get_booster().get_score(importance_type='gain'), key=lambda x:x[1], reverse=True)]
    # reg_results = np.array(reg.evals_result()["validation_0"]["rmse"])

    # Convert to DMatrix for SHAP value
    # m = xgb.DMatrix(train_X, enable_categorical=True)  # specify categorical data support.
    # SHAP = booster.predict(m, pred_contribs=True)
    # margin = booster.predict(m, output_margin=True)
    # np.testing.assert_allclose(
    #    np.sum(SHAP, axis=len(SHAP.shape) - 1), margin, rtol=1e-3
    # )
    # pred_test =
    # kfold = KFold(n_splits=10, random_state=7, shuffle=True)
    # results = cross_val_score(reg, train_X, train_Y, cv=kfold)
    # print("Accuracy: %.2f%% (%.2f%%)" % (results.mean() * 100, results.std() * 100))
    # print(
    #    f"mAE: {mean_absolute_error(y_test, reg.get_booster().predict(xgb.DMatrix(X_test, enable_categorical=True)))}")

    # tree_fig = plt.figure()
    # tree_ax = tree_fig.add_subplot(111)

    # plot_tree(reg, num_trees=1, rankdir='LR', ax=tree_ax)
    #################################################################
    # OUTPUT
    #################################################################
    pred_test = reg.get_booster().predict(xgb.DMatrix(testing_ds, enable_categorical=True))
    print("-----------PREDICTION PIPELINE COMPLETED-------------")
    pd.DataFrame({'id': testings_ids, 'rent': np.round(pred_test)}).to_csv("submission_test.csv", index=False)


def arg_parse():
    arg_parser = ArgumentParser()
    arg_parser.add_argument("--train_ds", required=True, type=str)
    arg_parser.add_argument("--test_ds", required=True, type=str)
    arg_parser.add_argument("--ds_config", required=True, type=str)
    arg_parser.add_argument("--config", required=True, type=str)
    arg_parser.add_argument("--ds_cities", required=True, type=str)
    return arg_parser


if __name__ == "__main__":
    argz = arg_parse().parse_args()
    execute(argz)
