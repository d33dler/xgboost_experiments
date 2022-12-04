from argparse import ArgumentParser
from typing import Tuple, List

import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
import sklearn.metrics
from easydict import EasyDict
from pandas.core.generic import NDFrame
from xgboost import plotting, XGBModel
import xgboost as xgb
from sklearn.feature_extraction import FeatureHasher
import csv as csv
from xgboost import plot_importance, plot_tree
from matplotlib import pyplot
from sklearn.model_selection import cross_val_score, KFold

from sklearn.metrics import mean_absolute_error, accuracy_score
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV, train_test_split  # Perforing grid search
from scipy.stats import skew
from collections import OrderedDict
import yaml
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score


def load_data(train: str, test: str):
    train_dataset = pd.read_csv(train, header=0)
    test_dataset = pd.read_csv(test, header=0)
    return train_dataset, test_dataset


def transform_data(data):
    pass


def clean_dataset(data, col: list):
    pass


class CategoricalFix:

    @staticmethod
    def leader_board(data: NDFrame, col: str):
        pass

    @staticmethod
    def binary_one(data: NDFrame, col: str):
        pass

    @staticmethod
    def binary_keys(data: NDFrame, col: str):
        pass


def normalize(data):
    pass


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


def execute(_argz):
    cfg = load_config(_argz.config)
    ds_cfg = load_config(_argz.ds_config)
    train_ds, testing_ds = load_data(train=_argz.train_ds, test=_argz.test_ds)

    ##################################
    print(ds_cfg)
    y_target = ds_cfg.TARGET

    train_ds.drop(ds_cfg.EXCLUDE, axis=ds_cfg.EXCLUDE_AXIS, inplace=True)
    testing_ds.drop(ds_cfg.EXCLUDE, axis=ds_cfg.EXCLUDE_AXIS, inplace=True)
    train_ds.fillna(0, inplace=True)
    testing_ds.fillna(0, inplace=True)
    cats_dict = {o.ID: o for o in ds_cfg.CATS}
    split = [train_ds, testing_ds]

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

    ##############################################

    train_X = train_ds.drop([y_target], axis=ds_cfg.EXCLUDE_AXIS, inplace=False)
    train_Y = train_ds[ds_cfg.TARGET]
    train_X = pd.get_dummies(train_X)
    print(train_X.info(verbose=True))

    ##############################################

    reg: XGBModel = xgb.XGBRegressor(tree_method="gpu_hist",
                                     objective='reg:squarederror',
                                     enable_categorical=True,
                                     max_cat_to_onehot=8,
                                     max_depth=100,
                                     gamma=0.05,
                                     eta=0.1,
                                     learning_rate=0.07,
                                     min_child_weight=1.5,
                                     subsample=0.3,
                                     seed=33,
                                     n_estimators=100,
                                     colsample_bylevel=0.7)

    ################################################################

    X_train, X_test, y_train, y_test = train_test_split(train_X, train_Y, test_size=0.3, random_state=123)
    reg.fit(X_train, y_train, eval_set=[(train_X, train_Y)], verbose=True)

    # reg_results = np.array(reg.evals_result()["validation_0"]["rmse"])

    # Convert to DMatrix for SHAP value
    booster: xgb.Booster = reg.get_booster()
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
    print(f"mAE: {mean_absolute_error(y_test, [np.round(v) for v in booster.predict(xgb.DMatrix(X_test, enable_categorical=True))])}")
    plot_tree(reg, num_trees=1, rankdir='LR')
    plt.show()

    # pd.DataFrame({'id': testing_ds.id, 'rent': pred_test}).to_csv("submission_test.csv", index=False)


def arg_parse():
    arg_parser = ArgumentParser()
    arg_parser.add_argument("--train_ds", required=True, type=str)
    arg_parser.add_argument("--test_ds", required=True, type=str)
    arg_parser.add_argument("--ds_config", required=True, type=str)
    arg_parser.add_argument("--config", required=True, type=str)
    return arg_parser


if __name__ == "__main__":
    argz = arg_parse().parse_args()
    execute(argz)
