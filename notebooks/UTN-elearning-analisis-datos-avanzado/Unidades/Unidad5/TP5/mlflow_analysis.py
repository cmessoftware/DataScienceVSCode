#!/usr/bin/env python3
"""
MLflow Experiment Analysis Dashboard
====================================

Script para analizar y comparar experimentos de MLflow de manera interactiva.
Uso: python mlflow_analysis.py
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import mlflow
import numpy as np
import pandas as pd
import seaborn as sns


def setup_mlflow():
    """Configurar MLflow tracking"""
    mlflow_tracking_dir = Path.cwd() / "mlruns"
    mlflow.set_tracking_uri(f"file://{mlflow_tracking_dir}")
    return mlflow_tracking_dir


def get_experiment_data(experiment_name="Churn_Prediction_TP5"):
    """Obtener datos del experimento"""
    try:
        experiment = mlflow.get_experiment_by_name(experiment_name)
        if experiment is None:
            print(f"❌ Experimento '{experiment_name}' no encontrado")
            return None

        runs = mlflow.search_runs(experiment_ids=[experiment.experiment_id])
        return experiment, runs
    except Exception as e:
        print(f"❌ Error obteniendo datos del experimento: {e}")
        return None


def create_comparison_dashboard(runs_df):
    """Crear dashboard de comparación visual"""
    if runs_df.empty:
        print("⚠️ No hay runs para mostrar")
        return

    # Preparar datos
    metrics_cols = [col for col in runs_df.columns if col.startswith("metrics.")]
    if not metrics_cols:
        print("⚠️ No se encontraron métricas en los runs")
        return

    # Extraer información relevante
    model_col = "tags.model_type" if "tags.model_type" in runs_df.columns else None
    if model_col is None:
        print("⚠️ No se encontró información de tipos de modelo")
        return

    # Crear figura
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle(
        "📊 Dashboard de Comparación de Modelos ML", fontsize=16, fontweight="bold"
    )

    # Métricas principales
    main_metrics = [
        "metrics.roc_auc",
        "metrics.accuracy",
        "metrics.f1_score",
        "metrics.precision",
        "metrics.recall",
    ]

    available_metrics = [m for m in main_metrics if m in runs_df.columns]

    for i, metric in enumerate(available_metrics[:6]):  # Máximo 6 gráficos
        row, col = divmod(i, 3)
        ax = axes[row, col]

        # Preparar datos para el gráfico
        data = runs_df[[model_col, metric]].dropna()
        if data.empty:
            ax.text(
                0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes
            )
            continue

        # Crear gráfico de barras
        metric_name = metric.replace("metrics.", "").upper()
        models = data[model_col].values
        values = data[metric].values

        # Truncar nombres largos
        models_short = [m[:12] + "..." if len(m) > 15 else m for m in models]

        bars = ax.bar(models_short, values, alpha=0.7)
        ax.set_title(f"{metric_name}", fontweight="bold")
        ax.set_ylabel(metric_name)
        ax.tick_params(axis="x", rotation=45)
        ax.grid(axis="y", alpha=0.3)

        # Highlight del mejor modelo
        if len(values) > 0:
            best_idx = np.argmax(values)
            bars[best_idx].set_color("gold")
            bars[best_idx].set_edgecolor("orange")
            bars[best_idx].set_linewidth(2)

    # Ocultar subplots vacíos
    for i in range(len(available_metrics), 6):
        row, col = divmod(i, 3)
        axes[row, col].set_visible(False)

    plt.tight_layout()
    plt.show()


def print_summary_table(runs_df):
    """Imprimir tabla resumen de experimentos"""
    if runs_df.empty:
        print("⚠️ No hay runs para mostrar")
        return

    # Seleccionar columnas relevantes
    summary_cols = []

    # Modelo
    if "tags.model_type" in runs_df.columns:
        summary_cols.append("tags.model_type")

    # Métricas principales
    metric_mapping = {
        "metrics.roc_auc": "ROC_AUC",
        "metrics.accuracy": "Accuracy",
        "metrics.precision": "Precision",
        "metrics.recall": "Recall",
        "metrics.f1_score": "F1_Score",
    }

    for mlflow_col, display_col in metric_mapping.items():
        if mlflow_col in runs_df.columns:
            summary_cols.append(mlflow_col)

    if not summary_cols:
        print("⚠️ No se encontraron columnas relevantes para el resumen")
        return

    # Crear tabla resumen
    summary_df = runs_df[summary_cols].copy()

    # Renombrar columnas
    rename_dict = {"tags.model_type": "Model"}
    rename_dict.update(
        {k: v for k, v in metric_mapping.items() if k in summary_df.columns}
    )
    summary_df = summary_df.rename(columns=rename_dict)

    # Redondear métricas
    metric_cols = [col for col in summary_df.columns if col != "Model"]
    for col in metric_cols:
        if col in summary_df.columns:
            summary_df[col] = summary_df[col].round(4)

    # Ordenar por ROC AUC si está disponible
    if "ROC_AUC" in summary_df.columns:
        summary_df = summary_df.sort_values("ROC_AUC", ascending=False)

    print("\n📋 Tabla Resumen de Experimentos:")
    print("=" * 80)
    print(summary_df.to_string(index=False))
    print("=" * 80)


def find_best_models(runs_df, top_n=3):
    """Encontrar los mejores modelos"""
    if runs_df.empty or "metrics.roc_auc" not in runs_df.columns:
        print("⚠️ No se puede determinar los mejores modelos")
        return

    best_models = runs_df.nlargest(top_n, "metrics.roc_auc")

    print(f"\n🏆 Top {top_n} Mejores Modelos:")
    print("=" * 60)

    for i, (idx, row) in enumerate(best_models.iterrows(), 1):
        model_name = row.get("tags.model_type", "Unknown")
        roc_auc = row.get("metrics.roc_auc", 0)
        accuracy = row.get("metrics.accuracy", 0)
        f1_score = row.get("metrics.f1_score", 0)
        run_id = row.get("run_id", "Unknown")

        print(f"\n{i}. 🥇 {model_name}")
        print(f"   📊 ROC AUC: {roc_auc:.4f}")
        print(f"   📊 Accuracy: {accuracy:.4f}")
        print(f"   📊 F1-Score: {f1_score:.4f}")
        print(f"   🔗 Run ID: {run_id[:8]}...")

    # ========================================================================
    # FUNCION HELPER PARA LOGGING DE METRICAS
    # ========================================================================
    def log_model_metrics(
        model_name: str,
        y_true,
        y_pred,
        y_pred_proba=None,
        model=None,
        X_test=None,
        hyperparams=None,
        cv_scores=None,
    ):
        """
        Funcion para registrar metricas de modelo en MLflow.
        """
        # Calcular metricas basicas
        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred, average="weighted", zero_division=0)
        recall = recall_score(y_true, y_pred, average="weighted", zero_division=0)
        f1 = f1_score(y_true, y_pred, average="weighted", zero_division=0)

        # Calcular ROC AUC si tenemos probabilidades
        roc_auc = None
        if y_pred_proba is not None:
            try:
                if y_pred_proba.ndim > 1 and y_pred_proba.shape[1] > 1:
                    roc_auc = roc_auc_score(y_true, y_pred_proba[:, 1])
                else:
                    roc_auc = roc_auc_score(y_true, y_pred_proba)
            except Exception as e:
                print(f"⚠️ Error calculando ROC AUC: {e}")

        if MLFLOW_AVAILABLE and MLFLOW_TRACKING_ENABLED:
            try:
                with mlflow.start_run(run_name=model_name, nested=True):
                    # Log mtricas en MLflow
                    mlflow.log_metric("accuracy", float(accuracy))
                    mlflow.log_metric("precision", float(precision))
                    mlflow.log_metric("recall", float(recall))
                    mlflow.log_metric("f1_score", float(f1))

                    if roc_auc is not None:
                        mlflow.log_metric("roc_auc", float(roc_auc))

                    # Cross-validation scores si estn disponibles
                    if cv_scores is not None:
                        mlflow.log_metric("cv_mean", float(cv_scores.mean()))
                        mlflow.log_metric("cv_std", float(cv_scores.std()))

                    # Hiperparmetros
                    if hyperparams:
                        mlflow.log_params(hyperparams)

                    # Registrar el modelo si est disponible
                    if model is not None and X_test is not None:
                        try:
                            signature = infer_signature(X_test, y_pred)
                            mlflow_sklearn.log_model(
                                model, "model", signature=signature
                            )
                            print(f"   *** Modelo guardado exitosamente")
                        except Exception as e:
                            print(
                                f"   ⚠️ Error guardando modelo: {str(e)[:50]}... (continuando)"
                            )

                    # Tags tiles
                    mlflow.set_tag("model_type", model_name)
                    mlflow.set_tag("problem_type", "binary_classification")
                    mlflow.set_tag("dataset", "telecom_churn")
                    mlflow.set_tag("framework", "scikit-learn")

                    run_info = mlflow.active_run()
                    if run_info is not None:
                        run_id = run_info.info.run_id
                        print(
                            f"✅ Mtricas registradas en MLflow para {model_name} (Run ID: {run_id[:8]}...)"
                        )
                        return run_id
                    else:
                        print(f"✅ Mtricas registradas en MLflow para {model_name}")
                        return "no_run_id"

            except Exception as e:
                print(f"⚠️ Error con MLflow para {model_name}: {str(e)[:80]}...")
                print(" Continuando con mtricas en consola...")

        # Mostrar mtricas en consola (siempre o como fallback)
        print(f"\n {model_name} Metrics:")
        print(f"   Accuracy: {accuracy:.4f}")
        print(f"   Precision: {precision:.4f}")
        print(f"   Recall: {recall:.4f}")
        print(f"   F1-Score: {f1:.4f}")

        if roc_auc is not None:
            print(f"   ROC AUC: {roc_auc:.4f}")

        if cv_scores is not None:
            print(
                f"   CV Score: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})"
            )

        if hyperparams:
            print(f"   Hyperparameters: {hyperparams}")

        print("-" * 50)

        return roc_auc  # Retornar AUC para comparaciones

    print("✅  Funcion de logging de metricas configurada")


def analyze_optimization_impact(runs_df):
    """Analizar el impacto de la optimización"""
    # Buscar modelos originales y optimizados
    original_runs = runs_df[
        ~runs_df["tags.model_type"].str.contains("OPTIMIZED", na=False)
    ]
    optimized_runs = runs_df[
        runs_df["tags.model_type"].str.contains("OPTIMIZED", na=False)
    ]

    if original_runs.empty or optimized_runs.empty:
        print(
            "⚠️ No se encontraron pares de modelos originales/optimizados para comparar"
        )
        return

    print("\n🔬 Análisis de Impacto de Optimización:")
    print("=" * 50)

    # Comparar métricas
    metrics_to_compare = ["metrics.roc_auc", "metrics.accuracy", "metrics.f1_score"]

    for metric in metrics_to_compare:
        if metric in runs_df.columns:
            original_avg = original_runs[metric].mean()
            optimized_avg = optimized_runs[metric].mean()
            improvement = optimized_avg - original_avg

            metric_name = metric.replace("metrics.", "").upper()
            print(f"\n📈 {metric_name}:")
            print(f"   - Original promedio: {original_avg:.4f}")
            print(f"   - Optimizado promedio: {optimized_avg:.4f}")
            print(
                f"   - Mejora: {improvement:+.4f} ({improvement/original_avg*100:+.2f}%)"
            )


def main():
    """Función principal"""
    parser = argparse.ArgumentParser(description="Analizar experimentos MLflow")
    parser.add_argument(
        "--experiment",
        "-e",
        default="Churn_Prediction_TP5",
        help="Nombre del experimento a analizar",
    )
    parser.add_argument("--no-plots", action="store_true", help="No mostrar gráficos")
    args = parser.parse_args()

    print("🔧 MLflow Experiment Analysis Dashboard")
    print("=" * 50)

    # Configurar MLflow
    tracking_dir = setup_mlflow()
    print(f"📁 MLflow Tracking URI: {mlflow.get_tracking_uri()}")

    # Obtener datos del experimento
    result = get_experiment_data(args.experiment)
    if result is None:
        return

    experiment, runs_df = result
    print(f"🧪 Experimento: {experiment.name}")
    print(f"📊 Total de runs: {len(runs_df)}")

    if runs_df.empty:
        print("⚠️ No hay runs en este experimento")
        return

    # Análisis detallado
    print_summary_table(runs_df)
    find_best_models(runs_df)
    analyze_optimization_impact(runs_df)

    # Gráficos (si no está deshabilitado)
    if not args.no_plots:
        create_comparison_dashboard(runs_df)

    # Instrucciones finales
    print(f"\n🌐 Para acceder al MLflow UI completo:")
    print(f"   1. Ejecuta: mlflow ui")
    print(f"   2. Abre: http://localhost:5000")
    print(f"   3. Busca el experimento: '{experiment.name}'")

    print("\n✅ Análisis completado")


if __name__ == "__main__":
    main()
