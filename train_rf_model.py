# -*- coding: utf-8 -*-

from pathlib import Path
import h5py
import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from typing import Tuple, List

INPUT_H5 = Path("input_txt/data_input.h5")
MODEL_PATH = Path("model/final_model.m")
METRICS_PATH = Path("model/final_model.txt")

FEATURE_NAMES: List[str] = [
    'branch', 'subcode', 'division', 'diff_year1', 'diff_year2',
    'name1', 'name2', 'name3', 'name4', 'funding',
    'institution1', 'institution2', 'institution3',
    'keywords1', 'keywords2', 'keywords3', 'keywords4',
    'participator1', 'participator2', 'type_value'
]


def load_h5(path: Path) -> Tuple[np.ndarray, np.ndarray]:
    with h5py.File(str(path), "r") as f:
        data_x = f["data_input_x"][()]
        data_y = f["data_input_y"][()]
    return np.asarray(data_x), np.asarray(data_y)


def split_80_20(X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    n = len(X)
    split = int(0.8 * n)
    train_x = X[:split]
    train_y = y[:split]
    test_x = X[split:]
    test_y = y[split:]
    return train_x, train_y, test_x, test_y


def train_rf(train_x: np.ndarray, train_y: np.ndarray) -> RandomForestClassifier:
    clf = RandomForestClassifier(
        n_estimators=100,
        max_depth=22,
        min_samples_split=2,
        min_samples_leaf=1,
        max_features=11,
        random_state=10,
    )
    clf.fit(train_x, train_y)
    return clf


def evaluate_and_save(clf: RandomForestClassifier, test_x: np.ndarray, test_y: np.ndarray) -> None:
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    METRICS_PATH.parent.mkdir(parents=True, exist_ok=True)

    joblib.dump(clf, str(MODEL_PATH))

    preds = clf.predict(test_x)
    ac = accuracy_score(test_y, preds)
    pr = precision_score(test_y, preds)
    re = recall_score(test_y, preds)
    f1 = f1_score(test_y, preds)

    with METRICS_PATH.open("a", encoding="utf-8") as fw:
        fw.write(f"{ac}\t{pr}\t{re}\t{f1}\t\n")
        fw.flush()

    importances = clf.feature_importances_
    order = np.argsort(importances)[::-1]
    with METRICS_PATH.open("a", encoding="utf-8") as fw:
        for i in order:
            fname = FEATURE_NAMES[i] if i < len(FEATURE_NAMES) else f"feature_{i}"
            fw.write(f"{fname}\t{importances[i]}\n")
            fw.flush()


def main() -> None:
    X, y = load_h5(INPUT_H5)

    train_x, train_y, test_x, test_y = split_80_20(X, y)

    clf = train_rf(train_x, train_y)

    evaluate_and_save(clf, test_x, test_y)


if __name__ == "__main__":
    main()
