import joblib
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, confusion_matrix,
    classification_report
)
from xgboost import XGBClassifier


DATA_PATH = "/home/amor/Projects/ETA/data/csv/Projects/online+shoppers+purchasing+intention+dataset/online_shoppers_intention.csv"
RANDOM_STATE = 42

TARGET = "Revenue"

# These integer columns are category IDs, not quantities.
CAT_COLS = [
    "Month",
    "OperatingSystems",
    "Browser",
    "Region",
    "TrafficType",
    "VisitorType",
    "Weekend",
]

NUM_COLS = [
    "Administrative",
    "Administrative_Duration",
    "Informational",
    "Informational_Duration",
    "ProductRelated",
    "ProductRelated_Duration",
    "BounceRates",
    "ExitRates",
    "PageValues",
    "SpecialDay",
]


def make_preprocessor():
    # No scaling: both final models are tree ensembles.
    # OneHotEncoder is required because the categorical integer codes
    # do not represent numerical magnitude.
    return ColumnTransformer(
        transformers=[
            (
                "num",
                SimpleImputer(strategy="median"),
                NUM_COLS,
            ),
            (
                "cat",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        (
                            "onehot",
                            OneHotEncoder(
                                handle_unknown="ignore",
                                sparse_output=True,
                            ),
                        ),
                    ]
                ),
                CAT_COLS,
            ),
        ],
        remainder="drop",
    )


def make_models(y_train):
    positive = int((y_train == 1).sum())
    negative = int((y_train == 0).sum())
    scale_pos_weight = negative / positive

    return {
        "random_forest": RandomForestClassifier(
            n_estimators=400,
            max_features="sqrt",
            min_samples_leaf=3,
            class_weight="balanced",
            n_jobs=-1,
            random_state=RANDOM_STATE,
        ),
        "xgboost": XGBClassifier(
            n_estimators=400,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.85,
            colsample_bytree=0.85,
            min_child_weight=3,
            reg_lambda=2,
            scale_pos_weight=scale_pos_weight,
            objective="binary:logistic",
            eval_metric="logloss",
            n_jobs=-1,
            random_state=RANDOM_STATE,
        ),
    }


def best_f1_threshold(y_true, probabilities):
    thresholds = np.linspace(0.10, 0.90, 161)
    scores = [
        f1_score(y_true, (probabilities >= t).astype(int))
        for t in thresholds
    ]
    i = int(np.argmax(scores))
    return float(thresholds[i]), float(scores[i])


def evaluate(name, model, X_test, y_test, threshold):
    p = model.predict_proba(X_test)[:, 1]
    pred = (p >= threshold).astype(int)

    print(f"\n===== {name} =====")
    print(f"Threshold : {threshold:.3f}")
    print(f"ROC-AUC   : {roc_auc_score(y_test, p):.4f}")
    print(f"PR-AUC    : {average_precision_score(y_test, p):.4f}")
    print(f"Precision : {precision_score(y_test, pred):.4f}")
    print(f"Recall    : {recall_score(y_test, pred):.4f}")
    print(f"F1        : {f1_score(y_test, pred):.4f}")
    print(f"Accuracy  : {accuracy_score(y_test, pred):.4f}")
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, pred))
    print(classification_report(y_test, pred))


def main():
    df = pd.read_csv(DATA_PATH)

    # Exact duplicate rows are removed as a data-cleaning decision.
    # The original file had 125 exact duplicates.
    df = df.drop_duplicates().reset_index(drop=True)

    X = df.drop(columns=TARGET)
    y = df[TARGET].astype(int)

    # Final untouched test set.
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        stratify=y,
        random_state=RANDOM_STATE,
    )

    # A validation split is used only to select the classification threshold.
    X_fit, X_val, y_fit, y_val = train_test_split(
        X_train,
        y_train,
        test_size=0.20,
        stratify=y_train,
        random_state=RANDOM_STATE,
    )

    for name, estimator in make_models(y_train).items():

        # 1) Fit on train-fit
        validation_model = Pipeline(
            [
                ("preprocess", make_preprocessor()),
                ("model", estimator),
            ]
        )
        validation_model.fit(X_fit, y_fit)

        # 2) Select threshold using validation data only
        val_prob = validation_model.predict_proba(X_val)[:, 1]
        threshold, _ = best_f1_threshold(y_val, val_prob)

        # 3) Refit on all training data
        final_model = Pipeline(
            [
                ("preprocess", make_preprocessor()),
                ("model", estimator),
            ]
        )
        final_model.fit(X_train, y_train)

        # 4) Evaluate exactly once on untouched test set
        evaluate(name, final_model, X_test, y_test, threshold)

        # 5) Save preprocessing + model + threshold together
        joblib.dump(
            {
                "pipeline": final_model,
                "threshold": threshold,
                "feature_columns": list(X.columns),
            },
            f"{name}.joblib",
        )


if __name__ == "__main__":
    main()
