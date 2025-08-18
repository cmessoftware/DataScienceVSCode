import os
import tempfile
from matplotlib import pyplot as plt
import mlflow
import numpy as np
import seaborn as sns
import pandas as pd
from sklearn.metrics import (accuracy_score,
                             log_loss,
                             precision_score,
                             recall_score,
                             f1_score,
                             roc_auc_score,
                             confusion_matrix,
                             classification_report,
                             ConfusionMatrixDisplay)
from sklearn.model_selection import cross_val_score 
from sklearn.metrics import (
    roc_auc_score, average_precision_score, f1_score, precision_score, recall_score, classification_report
)

class MLflowUtils:
    def __init__(self):
        self.uri = None
        self.experiment = None
         
    def init_mlflow(self, uri, experiment: str = "default"):
        mlflow.set_tracking_uri(uri)
        print(f'Create or use experiment: {experiment}')
        mlflow.set_experiment(experiment)

    def log_basic_metrics(self,**metrics):
        for k, v in metrics.items():
            mlflow.log_metric(k, float(v))

    def safe_log_params(prefix, estimator):
        """Loguea parámetros del estimador filtrando tipos serializables."""
        try:
            params = estimator.get_params()
            flat = {f"{prefix}{k}": v for k, v in params.items()
                    if isinstance(v, (int, float, str, bool, type(None)))}
            if flat:
                mlflow.log_params(flat)
        except Exception:
            pass
    
    def log_confusion_matrix(self,y_true, y_pred, model_name):
        from sklearn.metrics import confusion_matrix
        import matplotlib.pyplot as plt
        import seaborn as sns
        cm = confusion_matrix(y_true, y_pred)
        plt.figure(figsize=(4, 3))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title(f'Confusion Matrix - {model_name}')
        tmp = tempfile.mkdtemp()
        cm_path = os.path.join(tmp, "confusion_matrix.png")
        plt.savefig(cm_path)
        mlflow.log_artifact(cm_path, artifact_path="confusion_matrix")
        plt.close()

    def plot_confusion_matrix(self,y_true, y_pred, title="Matriz de confusión"):
        cm = confusion_matrix(y_true, y_pred)
        fig, ax = plt.subplots()
        im = ax.imshow(cm, interpolation="nearest")
        ax.set_title(title)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Real")
        for (i, j), v in np.ndenumerate(cm):
            ax.text(j, i, str(v), ha="center", va="center")
        fig.colorbar(im, ax=ax)
        return fig, ax

    # ---------------------- Métricas + reporte ----------------------
    def evaluate_and_log(self, pipe, X_train, y_train, X_test, y_test, run_tags):
        # fit
        pipe.fit(X_train, y_train)
    
        # proba vs decisión
        if hasattr(pipe, "predict_proba"):
            y_proba = pipe.predict_proba(X_test)[:, 1]
        elif hasattr(pipe, "decision_function"):
            # escalamos a [0,1] para comparabilidad mínima
            df = pipe.decision_function(X_test)
            df_min, df_max = df.min(), df.max()
            y_proba = (df - df_min) / (df_max - df_min + 1e-12)
        else:
            # fallback súper básico
            y_proba = pipe.predict(X_test).astype(float)
    
        y_pred_05 = (y_proba >= 0.5).astype(int)
    
        metrics = {
            "roc_auc_test": float(roc_auc_score(y_test, y_proba)),
            "avg_precision_test": float(average_precision_score(y_test, y_proba)),
            "f1_0p5_test": float(f1_score(y_test, y_pred_05)),
            "precision_0p5_test": float(precision_score(y_test, y_pred_05)),
            "recall_0p5_test": float(recall_score(y_test, y_pred_05)),
        }
        mlflow.log_metrics(metrics)
    
        # umbral óptimo por F1 (usa helper del usuario si existe)
        try:
            thr_opt, f1_opt, p_opt, r_opt = best_threshold_by_f1(y_test, y_proba)
        except Exception:
            # fallback simple: barrido fino
            thrs = np.linspace(0.01, 0.99, 99)
            scores = []
            for t in thrs:
                yp = (y_proba >= t).astype(int)
                scores.append((t, f1_score(y_test, yp), precision_score(y_test, yp), recall_score(y_test, yp)))
            thr_opt, f1_opt, p_opt, r_opt = max(scores, key=lambda z: z[1])
    
        mlflow.log_metrics({
            "opt_threshold": float(thr_opt),
            "f1_at_opt_thr": float(f1_opt),
            "precision_at_opt_thr": float(p_opt),
            "recall_at_opt_thr": float(r_opt),
        })
    
        # classification report al umbral óptimo
        y_pred_opt = (y_proba >= thr_opt).astype(int)
        report = classification_report(y_test, y_pred_opt, digits=4)
        with open("classification_report.txt", "w", encoding="utf-8") as f:
            f.write(report)
        mlflow.log_artifact("classification_report.txt")
    
        # figuras del usuario si las tenés
        try:
            log_roc_pr_cm_to_mlflow(y_test, y_proba, y_pred_opt, name_prefix=f"{run_tags.get('model_name','')}_")
        except Exception:
            pass
    
        # importancias/coefs si aplica
        try:
            try_log_importances(pipe)
        except Exception:
            pass
    
        return {
            **run_tags,
            **metrics,
            "f1_at_opt_thr": float(f1_opt),
            "precision_at_opt_thr": float(p_opt),
            "recall_at_opt_thr": float(r_opt),
            "opt_threshold": float(thr_opt),
        }

    #====================== HELPERS MÉTRICAS/FIGURAS ======================
    def best_threshold_by_f1(y_true, y_proba):
        p, r, thr = precision_recall_curve(y_true, y_proba)
        f1 = 2 * (p[:-1]*r[:-1]) / (p[:-1]+r[:-1] + 1e-12)
        idx = int(np.nanargmax(f1))
        return float(thr[idx]), float(f1[idx]), float(p[idx]), float(r[idx])

    def log_roc_pr_cm_to_mlflow(y_true, y_proba, y_pred, name_prefix=""):
        # ROC
        fpr, tpr, _ = roc_curve(y_true, y_proba)
        fig = plt.figure()
        plt.plot(fpr, tpr, label=f"AUC={roc_auc_score(y_true, y_proba):.3f}")
        plt.plot([0,1],[0,1],'--')
        plt.xlabel("FPR"); plt.ylabel("TPR"); plt.legend(); plt.title("ROC Curve")
        mlflow.log_figure(fig, f"{name_prefix}roc_curve.png")
        plt.close(fig)

        # PR
        p, r, _ = precision_recall_curve(y_true, y_proba)
        baseline = y_true.mean()
        fig = plt.figure()
        plt.plot(r, p, label=f"AP={average_precision_score(y_true, y_proba):.3f}")
        plt.hlines(baseline, 0, 1, linestyles='--', label=f"baseline={baseline:.3f}")
        plt.xlabel("Recall"); plt.ylabel("Precision"); plt.legend(); plt.title("PR Curve")
        mlflow.log_figure(fig, f"{name_prefix}pr_curve.png")
        plt.close(fig)

        # Confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        fig = plt.figure()
        plt.imshow(cm, interpolation='nearest')
        plt.title("Confusion Matrix"); plt.xlabel("Predicted"); plt.ylabel("True")
        for (i, j), val in np.ndenumerate(cm):
            plt.text(j, i, int(val), ha='center', va='center')
        mlflow.log_figure(fig, f"{name_prefix}confusion_matrix.png")
        plt.close(fig)

    def try_log_importances(pipeline):
        try:
            model = pipeline.named_steps["classifier"]
            # nombres de features si están disponibles
            feat_names = None
            try:
                feat_names = pipeline.named_steps["preprocessor"].get_feature_names_out()
            except Exception:
                pass

            if hasattr(model, "feature_importances_"):
                imp = model.feature_importances_
                df_imp = pd.DataFrame({"feature": feat_names if feat_names is not None else np.arange(len(imp)),
                                    "importance": imp}).sort_values("importance", ascending=False)
                df_imp.to_csv("feature_importances.csv", index=False)
                mlflow.log_artifact("feature_importances.csv")
            elif hasattr(model, "coef_"):
                coefs = model.coef_.ravel()
                df_coef = pd.DataFrame({"feature": feat_names if feat_names is not None else np.arange(len(coefs)),
                                        "coef": coefs}).sort_values("coef", key=lambda s: s.abs(), ascending=False)
                df_coef.to_csv("coefficients.csv", index=False)
                mlflow.log_artifact("coefficients.csv")
        except Exception as e:
            mlflow.log_param("explainability_note", f"No FI/coef logged: {type(e).__name__}")

    def train_and_log_multiclass_model(
        self, model, model_name, X_train, X_test, y_train, y_test, *,
        dataset_name="unknown", params=None, targets_str_col_name="species_name",
        problem_type="auto", target_names=None
    ):
        """
        Entrena y registra métricas en MLflow. Soporta problemas binarios y multiclase.
        - problem_type: "binary" | "multiclass" | "auto"
        - target_names: lista opcional con nombres de clase (en el orden de labels)
        - params puede incluir {"label_encoder": <sklearn.preprocessing.LabelEncoder>}
        """
        with mlflow.start_run(run_name=f"{dataset_name} - {model_name}"):
            # ----- Parámetros -----
            # n_classes tentativo (puede ajustarse después del fit)
            tentative_n_classes = len(np.unique(y_train))
            mlflow.log_param("model_type", model_name)
            mlflow.log_param("dataset", dataset_name)
            mlflow.log_param("problem_type_requested", problem_type)
            mlflow.log_param("train_samples", len(X_train))
            mlflow.log_param("test_samples", len(X_test))
            mlflow.log_param("n_features", X_train.shape[1])
            mlflow.log_param("n_classes_tentative", tentative_n_classes)
    
            if params:
                for key, value in params.items():
                    # Evita loguear objetos pesados literalmente
                    mlflow.log_param(str(key), str(value))
    
            # ----- Entrenamiento -----
            model.fit(X_train, y_train)
    
            # Determinar tipo de problema final
            classes_attr = getattr(model, "classes_", None)
            inferred_n_classes = len(classes_attr) if classes_attr is not None else len(np.unique(y_train))
            if problem_type == "auto":
                final_problem = "binary" if inferred_n_classes == 2 else "multiclass"
            else:
                final_problem = problem_type
            mlflow.log_param("problem_type_effective", final_problem)
            mlflow.log_param("n_classes", inferred_n_classes)
    
            # ----- Predicciones -----
            y_pred = model.predict(X_test)
            has_proba = hasattr(model, "predict_proba")
            has_decfunc = hasattr(model, "decision_function")
    
            # ----- Métricas globales -----
            accuracy = accuracy_score(y_test, y_pred)
            precision_w = precision_score(y_test, y_pred, average='weighted', zero_division=0)
            recall_w = recall_score(y_test, y_pred, average='weighted', zero_division=0)
            f1_w = f1_score(y_test, y_pred, average='weighted', zero_division=0)
    
            precision_m = precision_score(y_test, y_pred, average='macro', zero_division=0)
            recall_m = recall_score(y_test, y_pred, average='macro', zero_division=0)
            f1_m = f1_score(y_test, y_pred, average='macro', zero_division=0)
    
            mlflow.log_metric("accuracy", float(accuracy))
            mlflow.log_metric("precision_weighted", float(precision_w))
            mlflow.log_metric("recall_weighted", float(recall_w))
            mlflow.log_metric("f1_weighted", float(f1_w))
            mlflow.log_metric("precision_macro", float(precision_m))
            mlflow.log_metric("recall_macro", float(recall_m))
            mlflow.log_metric("f1_macro", float(f1_m))
    
            # ----- AUC (binaria o multiclase) -----
            if final_problem == "binary":
                if has_proba or has_decfunc:
                    # Positiva = última clase en model.classes_ (convención sklearn)
                    pos_class = classes_attr[-1] if classes_attr is not None else 1
                    if has_proba:
                        proba = model.predict_proba(X_test)
                        if proba.ndim == 2:
                            pos_idx = list(classes_attr).index(pos_class) if classes_attr is not None else 1
                            y_score = proba[:, pos_idx]
                        else:
                            y_score = proba
                    else:
                        y_score = model.decision_function(X_test)
                    auc = roc_auc_score(y_test, y_score, pos_label=pos_class)
                    mlflow.log_metric("auc", float(auc))
            else:
                # multiclase
                if has_proba:
                    labels_for_auc = classes_attr  # alinear columnas de proba
                    proba = model.predict_proba(X_test)
                    auc_ovr = roc_auc_score(y_test, proba, multi_class='ovr', labels=labels_for_auc)
                    auc_ovo = roc_auc_score(y_test, proba, multi_class='ovo', labels=labels_for_auc)
                    mlflow.log_metric("auc_ovr", float(auc_ovr))
                    mlflow.log_metric("auc_ovo", float(auc_ovo))
    
            # ----- Cross-validation (accuracy) -----
            cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='accuracy')
            mlflow.log_metric("cv_mean", float(cv_scores.mean()))
            mlflow.log_metric("cv_std", float(cv_scores.std()))
    
            # ----- Registrar modelo -----
            mlflow.sklearn.log_model(model, "model")
    
            # ====== Etiquetas y nombres de clase ======
            # Orden base para métricas por-clase y gráficos
            if classes_attr is not None:
                labels = np.array(classes_attr)
            else:
                labels = np.unique(np.concatenate([np.asarray(y_train), np.asarray(y_test), np.asarray(y_pred)]))
    
            # target_names prioridad:
            # 1) target_names pasado por parámetro (validamos largo)
            # 2) params["label_encoder"].classes_ si coincide
            # 3) si y es string/objeto, usar esos nombres
            # 4) str(labels) como fallback
            if target_names is not None and len(target_names) == len(labels):
                display_names = list(map(str, target_names))
            else:
                le = (params or {}).get("label_encoder", None)
                if le is not None and hasattr(le, "classes_") and len(le.classes_) == len(labels):
                    name_map = {lab: str(name) for lab, name in zip(labels, le.classes_)}
                    display_names = [name_map.get(lab, str(lab)) for lab in labels]
                else:
                    yk = np.asarray(y_train).dtype.kind
                    if yk in ("U", "S", "O"):
                        uniq = pd.unique(pd.Series(y_train, name=targets_str_col_name)).tolist()
                        name_map = {u: str(u) for u in uniq}
                        display_names = [name_map.get(lab, str(lab)) for lab in labels]
                    else:
                        display_names = [str(lab) for lab in labels]
    
            # ----- Matriz de confusión -----
            cm = confusion_matrix(y_test, y_pred, labels=labels)
            disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=display_names)
            fig = disp.plot(values_format='d').figure_
            plt.title(f"Matriz de Confusión - {model_name} ({dataset_name})")
            plt.tight_layout()
    
            confusion_file = f"{dataset_name}_confusion_matrix_{model_name.replace(' ', '_')}.png"
            fig.savefig(confusion_file)
            mlflow.log_artifact(confusion_file)
            plt.show()
    
            # ----- Reporte de clasificación -----
            report = classification_report(
                y_test, y_pred, labels=labels, target_names=display_names, zero_division=0
            )
            report_file = f"{dataset_name}_classification_report_{model_name.replace(' ', '_')}.txt"
            with open(report_file, "w", encoding="utf-8") as f:
                f.write(report)
            mlflow.log_artifact(report_file)
    
            # ----- Consola -----
            print(f"🌸 {model_name} entrenado en dataset {dataset_name}")
            print(f"📊 Accuracy: {accuracy:.4f}")
            print(f"📊 F1-Score (weighted): {f1_w:.4f}")
            print(f"📊 F1-Score (macro): {f1_m:.4f}")
            if final_problem == "binary" and (has_proba or has_decfunc):
                print(f"📊 AUC (binario): {auc:.4f}")
            elif final_problem == "multiclase" and has_proba:
                print(f"📊 AUC (OvR): {auc_ovr:.4f} | AUC (OvO): {auc_ovo:.4f}")
            print(f"📊 CV Score: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
            print("-" * 60)
    
            return model

