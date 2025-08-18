# %% imports y setup
import os, json, tempfile
import numpy as np
import mlflow
import mlflow.sklearn
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
import matplotlib.pyplot as plt

# opcionales
try:
    from xgboost import XGBClassifier
except Exception:
    XGBClassifier = None
try:
    from lightgbm import LGBMClassifier
except Exception:
    LGBMClassifier = None

mlflow.set_tracking_uri("http://127.0.0.1:5000")
mlflow.set_experiment("iris_trees_demo")

X, y = load_iris(return_X_y=True)
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

models = {
    "DecisionTree": DecisionTreeClassifier(random_state=42),
    "RandomForest": RandomForestClassifier(random_state=42, n_estimators=200, n_jobs=-1),
    "GradientBoosting": GradientBoostingClassifier(random_state=42),
}
if XGBClassifier is not None:
    models["XGBoost"] = XGBClassifier(
        n_estimators=200, max_depth=4, learning_rate=0.1,
        subsample=0.9, colsample_bytree=0.9,
        tree_method="hist", random_state=42, n_jobs=-1
    )
if LGBMClassifier is not None:
    models["LightGBM"] = LGBMClassifier(
        n_estimators=200, num_leaves=31, learning_rate=0.1,
        subsample=0.9, colsample_bytree=0.9, random_state=42
    )

# helper: log de matriz de confusión como artifact
def log_confusion_matrix(y_true, y_pred, run_name):
    cm = confusion_matrix(y_true, y_pred)
    fig = plt.figure()
    plt.imshow(cm, interpolation="nearest")
    plt.title(f"Confusion matrix - {run_name}")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    for (i, j), v in np.ndenumerate(cm):
        plt.text(j, i, str(v), ha="center", va="center")
    tmpdir = tempfile.mkdtemp()
    path = os.path.join(tmpdir, "confusion_matrix.png")
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    mlflow.log_artifact(path, artifact_path="plots")

# entrenamiento + tracking
for name, clf in models.items():
    with mlflow.start_run(run_name=name):
        try:
            mlflow.log_params({k: v for k, v in clf.get_params().items() if isinstance(v, (int, float, str, bool))})
        except Exception:
            pass

        clf.fit(Xtr, ytr)
        yp = clf.predict(Xte)

        acc = accuracy_score(yte, yp)
        f1m = f1_score(yte, yp, average="macro")
        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("f1_macro", f1m)

        report = classification_report(yte, yp, output_dict=True)
        tmp = tempfile.mkdtemp()
        with open(os.path.join(tmp, "classification_report.json"), "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)
        mlflow.log_artifact(os.path.join(tmp, "classification_report.json"), artifact_path="reports")

        if hasattr(clf, "feature_importances_"):
            importances = clf.feature_importances_.tolist()
            with open(os.path.join(tmp, "feature_importances.json"), "w", encoding="utf-8") as f:
                json.dump(importances, f, indent=2)
            mlflow.log_artifact(os.path.join(tmp, "feature_importances.json"), artifact_path="feature_importance")

        log_confusion_matrix(yte, yp, name)
        mlflow.sklearn.log_model(clf, artifact_path="model")
        print(f"{name:16s} | acc={acc:.4f} | f1_macro={f1m:.4f}")
