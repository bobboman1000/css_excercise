import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from scipy.stats import norm, lognorm
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
import sklearn.gaussian_process as gp
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import StratifiedKFold, KFold, RepeatedStratifiedKFold, train_test_split


def evaluate_feature_importance(estimator, threshold, X_, y_,  n_splits=10, random_state=None):
    """
    Evaluate feature importance using a classifier providing information about it.
    Conducts StratifiedCV to evaluate the baseline = all features vs. the trimmed classifer, that only considers features with importance above the treshold.
    :param estimator: Any sklearn estimator
    :param threshold: Threshold of feature importance to exceed to be selected for trimmed model
    :param n_splits: Number of splits for StratifiedCV
    :param random_state: Numpy random state for reproducability
    :return: pd.DataFrame Baseline vs. Trimmed model
    """
    kfold = StratifiedKFold(n_splits=20, random_state=random_state, shuffle=True)

    baseline = []
    trimmed = []

    for train_idx, test_idx in kfold.split(X_, y_):

        X_train = X_.iloc[train_idx, :]
        y_train = y_[train_idx]

        dt_baseline = DecisionTreeClassifier(random_state=random_state).fit(X_train, y_train)
        feature_importances = {key: dt_baseline.feature_importances_[idx] for idx, key in zip(range(len(X_.columns)), X_.columns)}

        trimmed_features = {key: feature_importances[key] for key in feature_importances if feature_importances[key] > 0.01}
        dt_trimmed = DecisionTreeClassifier(random_state=random_state).fit(X_train[trimmed_features], y_train)

        baseline.append(dt_baseline.score(X_.iloc[test_idx,:], y_[test_idx]))
        trimmed.append(dt_trimmed.score(X_[trimmed_features.keys()].iloc[test_idx, :], y_[test_idx]))

    return pd.DataFrame({"Baseline": baseline, "Trimmed": trimmed}), trimmed_features


def build_resampled_df(db, tf, verbose=False):
    """
    Resampled Datensatz, aggregiert jedes Attribut auf Summe, Mittel, Varianz, Min und Max.
    Samples die nicht eine eindeutige Klasse beinhalten, werden gelöscht.
    :param db: Vollständiger Sensordatensatz
    :param tf: Time frames zum sampling ind er Form ["1ms", "10ms", ...]
    :param verbose: Ausführliches logging Ja/Nein
    :return: gibt zurück ein Tupel aus observations-df und target-df
    """
    # Resample Dataset, aggregiere und entferne NA Zeilen
    db_tf = db.resample(tf).agg(['sum', 'mean', 'var', 'min', 'max'])
    db_tf = db_tf[db_tf["activity"]["mean"].isin([0, 1])]
    db_tf = db_tf.dropna()

    if verbose:
        print(tf, "shape:", db_tf.shape)

    y_tf = db_tf["activity"]["mean"]    #  if one is NA, mean will be NA as well => gets dropped in the next line
    X_tf = db_tf.drop(columns=("activity"))
    X_tf.columns = [' '.join(col).strip() for col in X_tf.columns.values]

    return X_tf, y_tf


def time_frame_optimization(db, model, time_frames, metric=accuracy_score, verbose=False, random_state=None) -> pd.DataFrame:
    """
    Findet mittels RepeatedStratifiedKfoldCV den besten timeframe anhand der angegeben metric.
    :param db:
    :param model:
    :param time_frames:
    :param metric:
    :param verbose:
    :return:
    """
    all_scores = pd.DataFrame(columns=time_frames)

    for tf in time_frames:

        X_tf, y_tf = build_resampled_df(db, tf, verbose)

        # Time Series Datasplit, behält Reihenfolge bei
        tscv = RepeatedStratifiedKFold(n_splits=5, n_repeats=10, random_state=random_state)
        scores = []
        for train_index, test_index in tscv.split(X_tf, y_tf):

            X_train, X_test = X_tf.iloc[train_index, :], X_tf.iloc[test_index, :]
            y_train, y_test = y_tf.iloc[train_index], y_tf.iloc[test_index]
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            scores.append(metric(y_test, y_pred))

            if verbose:
                print("Train:", len(train_index), "Test:", len(test_index), "Score:", metric(y_test, y_pred))
                print()

        all_scores[tf] = scores

    return all_scores
#%%
