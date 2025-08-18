#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🏆 Proyecto de Competencia Kaggle: Predicción de Abandono de Clientes 🎯

📊 Diplomado de Ciencia de Datos y Análisis Avanzado
📖 Unidad 5: Modelado Predictivo I - Regresión y Clasificación

🔄"""

from print_utils import USE_EMOJIS, safe_print, supports_emojis


def main():
    # Mostrar configuración de terminal
    if USE_EMOJIS:
        print("✨ Terminal con soporte completo de emojis detectado")
    else:
        print("[*] Terminal sin soporte de emojis - usando modo compatibilidad")

    safe_print("🚀 *** Iniciando script de predicción de churn...")

    # ========================================================================
    # 0. CONFIGURACIÓN INICIAL E IMPORTACIÓN DE LIBRERÍAS
    # ========================================================================
    safe_print("\n📦 *** Importando librerías...")

    # Importaciones bsicas
    import importlib
    import pathlib
    from datetime import datetime
    from typing import Any, Dict, List, Tuple

    import matplotlib.gridspec as gridspec
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    import seaborn as sns
    from sklearn.compose import ColumnTransformer
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import (
        accuracy_score,
        f1_score,
        precision_score,
        recall_score,
        roc_auc_score,
    )
    from sklearn.model_selection import GridSearchCV, cross_val_score, train_test_split
    from sklearn.naive_bayes import GaussianNB
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler

    # MLflow para tracking de experimentos (con importacin condicional)
    try:
        import mlflow
        import mlflow.sklearn as mlflow_sklearn
        from mlflow.models import infer_signature

        MLFLOW_AVAILABLE = True
        print("✅  MLflow importado correctamente")
    except ImportError:
        MLFLOW_AVAILABLE = False
        print("⚠️  MLflow no disponible - funcionará en modo fallback")

    # Configurar visualizaciones
    plt.style.use("default")
    sns.set_palette("husl")
    # Para script, usar backend no interactivo
    import matplotlib

    matplotlib.use("Agg")  # Backend no interactivo para scripts

    # 🔧 Importar módulos del proyecto (si existen)
    try:
        import data_loader
        import dataset_splitter
        import eda
        import metrics
        import mlflow_analysis
        import models
        import models_stable
        import submission

        print("✅  Módulos del proyecto importados correctamente")
        print("INFO: Módulo models_stable disponible")
    except ImportError as e:
        print(f"⚠️  Algunos módulos del proyecto no están disponibles: {e}")
        print("💡  Continuando con funcionalidad básica...")

    # ========================================================================
    # 📊 CONFIGURACIÓN DE MLFLOW
    # ========================================================================
    print("\n⚙️  Configurando MLflow...")

    if MLFLOW_AVAILABLE:
        # Configurar el directorio de tracking (Windows compatible)
        mlflow_tracking_dir = os.path.join(os.getcwd(), "mlruns")
        os.makedirs(mlflow_tracking_dir, exist_ok=True)

        # Usar URI con scheme file:// para compatibilidad completa
        tracking_uri = pathlib.Path(mlflow_tracking_dir).as_uri()

        print(f"[FOLDER] Configurando tracking URI: {tracking_uri}")
        mlflow.set_tracking_uri(tracking_uri)

        # Configurar variables de entorno para evitar warnings del Model Registry
        os.environ["MLFLOW_DISABLE_ENV_MANAGER_CONDA_WARNING"] = "TRUE"

        experiment_name = "Churn_Prediction_TP5"

        # Crear o obtener el experimento con configuracin optimizada
        try:
            # Verificar si el experimento ya existe primero
            existing_experiment = mlflow.get_experiment_by_name(experiment_name)
            if existing_experiment is not None:
                experiment_id = existing_experiment.experiment_id
                print(
                    f"✅ Experimento '{experiment_name}' encontrado con ID: {experiment_id}"
                )
            else:
                experiment_id = mlflow.create_experiment(experiment_name)
                print(
                    f"✅ Experimento '{experiment_name}' creado con ID: {experiment_id}"
                )
        except Exception as e:
            print(f"⚠️ Configuracin de experimento con limitaciones: {str(e)[:100]}...")
            print(" Continuando con experimento por defecto")
            experiment_id = "0"  # Usar experimento por defecto

        # Configurar experimento activo
        try:
            mlflow.set_experiment(experiment_name)
            print(f"🎯 Experimento activo: {experiment_name}")
            MLFLOW_TRACKING_ENABLED = True
        except Exception as e:
            print(f" Usando experimento por defecto - tracking bsico disponible")
            MLFLOW_TRACKING_ENABLED = True  # Tracking sigue funcionando

        print("✅ MLflow configurado exitosamente")

        # Iniciar run principal para toda la ejecucion
        main_run_name = (
            f"churn_prediction_run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        )
        if MLFLOW_AVAILABLE and MLFLOW_TRACKING_ENABLED:
            try:
                mlflow.start_run(run_name=main_run_name)
                print("✅ Run principal iniciado: {main_run_name}")
                active_run = mlflow.active_run()
                if active_run is not None:
                    main_run_id = active_run.info.run_id
                    print(f" Run ID: {main_run_id}")
                else:
                    print("⚠️ No se pudo obtener run activo")
                    main_run_id = "fallback"
            except Exception as e:
                print(f"⚠️ Error iniciando run principal: {e}")
                main_run_id = "fallback"

    else:
        print("⚠️ MLflow no disponible - usando configuracin fallback")
        MLFLOW_TRACKING_ENABLED = False
        main_run_id = f"fallback_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        experiment_id = "fallback"

    # ========================================================================
    # 📊 1. CARGA DE DATOS
    # ========================================================================
    print("\n📁 1.  Cargando datos...")

    try:
        X_train = pd.read_csv("train.csv")
        X_test = pd.read_csv("test.csv")
        sample_submission_df = pd.read_csv("sample_submission.csv")
    except FileNotFoundError:
        print(
            " ❌ Error: Asegúrate de que los archivos .csv están en el directorio actual"
        )
        print("📋   Archivos requeridos: train.csv, test.csv, sample_submission.csv")
        return False

    print(f"✅  Datos cargados exitosamente:")
    print(f"📊   - Dataset de entrenamiento: {X_train.shape}")
    print(f"🧪   - Dataset de prueba: {X_test.shape}")
    print(f"📝   - Sample submission: {sample_submission_df.shape}")

    # ========================================================================
    # 📈 2. ANÁLISIS EXPLORATORIO DE DATOS (EDA)
    # ========================================================================
    print("\n📊 2. Realizando análisis exploratorio...")

    # Información general del dataset
    print("ℹ️  INFORMACIÓN GENERAL DEL DATASET")
    print("=" * 50)
    print(f"📐 Dimensiones: {X_train.shape}")
    print(f"📋 Columnas: {list(X_train.columns)}")

    # Información sobre valores faltantes
    print("\n🔍 VALORES FALTANTES:")
    missing_values = X_train.isnull().sum()
    if missing_values.sum() == 0:
        print("✅  No hay valores faltantes")
    else:
        print(missing_values[missing_values > 0])

    # Distribución de la variable objetivo
    print("\n🎯 🎯 DISTRIBUCIÓN DE LA VARIABLE OBJETIVO (Churn):")
    y_train = X_train["Churn"]
    churn_counts = y_train.value_counts()
    churn_pct = y_train.value_counts(normalize=True) * 100
    print(f"No Churn: {churn_counts.get('No', 0)} ({churn_pct.get('No', 0):.1f}%)")
    print(f"Churn: {churn_counts.get('Yes', 0)} ({churn_pct.get('Yes', 0):.1f}%)")

    # Guardar gráficos en lugar de mostrarlos
    try:
        if not os.path.exists("img"):
            os.makedirs("img")
        plt.figure(figsize=(12, 4))
        plt.subplot(1, 2, 1)
        y_train.value_counts().plot(kind="bar", color=["lightblue", "salmon"])
        plt.title("Distribución de Churn")
        plt.xlabel("Churn")
        plt.ylabel("Frecuencia")
        plt.xticks(rotation=0)

        plt.subplot(1, 2, 2)
        y_train.value_counts(normalize=True).plot(
            kind="pie", autopct="%1.1f%%", colors=["lightblue", "salmon"]
        )
        plt.title("Proporcin de Churn")
        plt.ylabel("")

        plt.tight_layout()
        plt.savefig("churn_distribution.png", dpi=150, bbox_inches="tight")
        plt.close()
        print("✅ Gráfico de distribución guardado: churn_distribution.png")
    except Exception as e:
        print(f"⚠️ Error guardando gráfico: {e}")

    # ========================================================================
    # 3. PREPROCESAMIENTO DE DATOS
    # ========================================================================
    print("\n Iniciando preprocesamiento...")

    try:
        from models_stable import ChurnPredictor  # Usar versión estable sin emojis

        # Preparar datos para la división train/validation
        print("⚙️  Preparando datos para división train/validation...")

        # Separar features y target
        y = X_train["Churn"]
        print(f" Variable objetivo extrada: {y.shape}")

        # Extraer caractersticas (X) - remover Churn y customerID
        columns_to_drop = ["Churn"]
        if "customerID" in X_train.columns:
            columns_to_drop.append("customerID")

        X = X_train.drop(columns_to_drop, axis=1)
        print(f" Caractersticas extradas: {X.shape}")
        print(f"[REPORT] Columnas removidas: {columns_to_drop}")

        # Dividir datos en entrenamiento y validacin interna
        print("\n Dividiendo datos en train/validation interno...")
        X_train_split, X_val, y_train_split, y_val = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        print(f"✅ Divisin completada:")
        print(f"   - Train: {X_train_split.shape[0]} muestras")
        print(f"   - Validation: {X_val.shape[0]} muestras")

    except ImportError:
        print("❌ ❌ Error: No se pudo importar el módulo 'models_stable'")
        print("💡  Asegúrate de que el archivo models_stable.py está en el directorio")
        return False

    # ========================================================================
    # 🤖 4. MODELADO Y EVALUACIÓN
    # ========================================================================
    print("\n🚀 4. [ROBOT] Iniciando entrenamiento de modelos...")

    # Recargar módulos para asegurar versión más reciente
    import importlib
    import sys

    module_name = "models_stable"  # Usar versión estable sin emojis
    if module_name in sys.modules:
        del sys.modules[module_name]

    import models_stable

    importlib.reload(models_stable)

    # Instanciar la clase
    predictor = models_stable.ChurnPredictor(random_state=42)

    # Mapear datos para entrenamiento y evaluacin
    print(" Mapeando datos para entrenamiento y evaluacin...")
    label_mapping = {"No": 0, "Yes": 1}

    # Mapear los datos de entrenamiento y validacin
    y_train_split_mapped = y_train_split.map(label_mapping)
    y_val_mapped = y_val.map(label_mapping)

    print(f"✅ Datos mapeados exitosamente:")
    print(f"   - y_train_split_mapped valores nicos: {y_train_split_mapped.unique()}")
    print(f"   - y_val_mapped valores nicos: {y_val_mapped.unique()}")

    # Configurar preprocesador
    print("\n Configurando preprocesador...")
    if predictor.preprocessor is None:
        preprocessor = predictor.create_preprocessor(X_train_split)
        print(f"✅ Preprocessor creado: {type(preprocessor)}")

    # Crear modelos con el preprocesador
    print("\n Creando modelos con preprocesador...")
    models_dict = predictor.create_models()
    print(
        f"✅ Modelos creados con pipeline de preprocesamiento: {list(models_dict.keys())}"
    )

    # Entrenar modelos
    print("\n🎯 Entrenando modelos con datos mapeados...")
    try:
        predictor.train_models(X_train_split, y_train_split_mapped)
        print("\n✅ Entrenamiento completado para todos los modelos:")
        for name in predictor.models.keys():
            print(f"   ✅ {name}")
    except Exception as e:
        print(f"❌ Error durante entrenamiento: {e}")

        # Intentar solución con reinicialización completa
        print("\n🔧  Reinicializando pipeline completo...")
        predictor_fixed = models_stable.ChurnPredictor(random_state=42)
        preprocessor_fixed = predictor_fixed.create_preprocessor(X_train_split)
        preprocessor_fixed.fit(X_train_split)

        models_fixed = predictor_fixed.create_models()
        predictor_fixed.train_models(X_train_split, y_train_split_mapped)

        predictor = predictor_fixed
        models_dict = predictor.models
        print("✅ Pipeline reinicializado exitosamente")

    # ========================================================================
    # 📊 5. EVALUACIÓN DE MODELOS
    # ========================================================================
    print("\n📈 5.  Evaluando modelos...")

    try:
        # Evaluar modelos
        results = predictor.evaluate_models(X_val, y_val_mapped)
        best_model_name, best_model = predictor.get_best_model("ROC_AUC", results)

        print(f"\n🏆 Mejor modelo seleccionado: {best_model_name}")

        # Mostrar resultados
        print(f"\n📊  Resultados de evaluación:")
        for model_name, metrics in results.items():
            print(f"   {model_name}:")
            for metric_name, value in metrics.items():
                print(f"      - {metric_name}: {value:.4f}")
            print()

        # Generar reporte de modelos si est disponible
        try:
            predictor.generate_model_report(X_val, y_val_mapped)
        except Exception as e:
            print(f"⚠️ No se pudo generar reporte completo: {e}")

    except Exception as e:
        print(f"❌ Error durante evaluacin: {e}")
        return False

    # ========================================================================
    # 🎯 6. ENTRENAMIENTO FINAL Y GENERACIÓN DE SUBMISSION
    # ========================================================================
    print("\n🎯 6. Preparando modelo final para submission...")

    # Preparar datos limpios para entrenamiento final
    print("Verificando datos antes del preprocesamiento final...")

    # Remover customerID de los datos si existe
    if "customerID" in X_train.columns:
        X_train_clean = X_train.drop(["customerID"], axis=1)
    else:
        X_train_clean = X_train.copy()

    # Guardar customerIDs para el archivo de submission
    customer_ids = X_test["customerID"]

    # Remover customerID de X_test si existe
    if "customerID" in X_test.columns:
        X_test_clean = X_test.drop(["customerID"], axis=1)
    else:
        X_test_clean = X_test.copy()

    # Sincronizar datos
    if X_train_clean.shape[0] != y_train.shape[0]:
        print(f"⚠️ Sincronizando datos finales:")
        min_samples = min(X_train_clean.shape[0], y_train.shape[0])
        X_train_clean = X_train_clean.iloc[:min_samples]
        y_train_sync = (
            y_train.iloc[:min_samples]
            if hasattr(y_train, "iloc")
            else y_train[:min_samples]
        )
        print(f"Sincronizados a {min_samples} muestras")
    else:
        y_train_sync = y_train
        print("✅ Datos ya estn sincronizados")

    # Mapear y_train_sync para consistencia de tipos
    print(f"\n Mapeando y_train_sync a formato numrico...")
    y_train_sync = predictor.map_target(y_train_sync)

    # Crear el preprocesador con los datos originales
    preprocessor = predictor.create_preprocessor(X_train_clean)

    print("✅ Preprocesador final configurado exitosamente")
    print(f" Caractersticas procesadas: {X_train_clean.shape[1]}")
    print(f" Muestras para entrenamiento: {X_train_clean.shape[0]}")

    # Crear y entrenar modelos finales
    models_final = predictor.create_models()
    print("\n🎯 Iniciando entrenamiento final con datos completos...")
    print(f"🔍 DEBUG: Iniciando train_models con {len(predictor.models)} modelos...")

    predictor.train_models(X_train_clean, y_train_sync)

    print("\n✅ Entrenamiento final completado para todos los modelos:")
    for model_name in predictor.models.keys():
        print(f"   ✅ {model_name}")

    # ========================================================================
    # 📁 7. GENERACIÓN DE PREDICCIONES Y SUBMISSION
    # ========================================================================
    print("\n💾 7. [FILE] Generando predicciones finales...")

    try:
        from submission import create_submission_file

        print(f" Datos para entrenamiento final: {len(X_train_clean):,} muestras")

        if not os.path.exists("submissions"):
            os.makedirs("submissions", exist_ok=True)

        # Generar timestamp unico para evitar sobrescribir archivos
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        submission_file = (
            f"submissions\\submission_grupoM_{timestamp}_{main_run_id[:8]}.csv"
        )
        print(f"[FILE] Archivo de submission: {submission_file}")

        # Crear archivo de submission
        submission_df = create_submission_file(
            final_model=best_model,
            X_train_full=X_train_clean,
            y_train_full=y_train_sync,
            X_test_full=X_test_clean,
            customer_ids=customer_ids,
            filename=submission_file,
        )

        # Estadísticas de las predicciones
        predictions = np.array(submission_df.iloc[:, 1].values, dtype=float)
        print(f"\n Estadsticas de predicciones:")
        print(
            f"- Predicciones de churn (>0.5): {np.sum(predictions > 0.5):,} ({np.mean(predictions > 0.5)*100:.1f}%)"
        )
        print(
            f"- Predicciones de no churn (0.5): {np.sum(predictions <= 0.5):,} ({np.mean(predictions <= 0.5)*100:.1f}%)"
        )
        print(f"- Rango: [{predictions.min():.4f}, {predictions.max():.4f}]")

        print(
            f"\n✅ Archivo de submission 'submission_grupoM_script.csv' creado exitosamente"
        )

    except ImportError:
        print("⚠️ Mdulo submission no disponible, creando submission bsico...")

        # Crear submission bsico
        print("Entrenando el modelo final con todos los datos de entrenamiento...")
        final_model = best_model.fit(X_train_clean, y_train_sync)

        print("Generando predicciones de probabilidad sobre el conjunto de prueba...")
        y_pred_proba_final = final_model.predict_proba(X_test_clean)[:, 1]

        # Crear DataFrame de submission
        submission_basic = pd.DataFrame(
            {"customerID": customer_ids, "Churn": y_pred_proba_final}
        )

        submission_basic.to_csv("submission_grupoM_script_basic.csv", index=False)
        print("✅ Archivo 'submission_grupoM_script_basic.csv' generado exitosamente")

    # ========================================================================
    # ⚙️ 8. OPTIMIZACIÓN DE HIPERPARÁMETROS (OPCIONAL)
    # ========================================================================
    print("\n🎛️  Realizando optimización de hiperparámetros...")

    try:
        from models_stable import (
            hyperparameter_tuning,
        )  # Usar versión estable sin emojis

        print(f"🎯 Optimizando hiperparámetros para {best_model_name}...")

        # Definir grillas de parmetros segn el modelo
        if best_model_name and "Logistic" in best_model_name:
            param_grid = {
                "classifier__C": [0.01, 0.1, 1.0, 10.0, 100.0, 1000.0],
                "classifier__penalty": ["l1", "l2", "elasticnet"],
                "classifier__solver": [
                    "liblinear",
                    "lbfgs",
                    "saga",
                    "newton-cg",
                    "sag",
                    "sparse_cg",
                ],
                "classifier__max_iter": [100, 500, 1000, 5000, 10000],
                "classifier__tol": [1e-5, 1e-4, 1e-3, 1e-2],
            }
        elif best_model_name and "KNN" in best_model_name:
            param_grid = {
                "classifier__n_neighbors": [3, 5, 7, 9, 11, 15],
                "classifier__weights": [
                    "uniform",
                    "distance",
                    "kernel",
                    "precomputed",
                    "ball_tree",
                    "kd_tree",
                    "brute",
                    "auto",
                    "nearest_neighbors",
                    "radius_neighbors",
                    "nearest_centroid",
                    "nearest_centroid_classifier",
                    "radius_neighbors_classifier",
                ],
            }
        elif best_model_name and "Random" in best_model_name:
            param_grid = {
                "classifier__n_estimators": [50, 100, 200],
                "classifier__max_depth": [None, 10, 20],
                "classifier__min_samples_split": [2, 5],
            }
        else:
            # Para Naive Bayes u otros
            param_grid = {"classifier__var_smoothing": [1e-9, 1e-8, 1e-7]}

        # Realizar búsqueda de hiperparámetros con configuración optimizada para Windows
        grid_search = hyperparameter_tuning(
            best_model, param_grid, X_train_clean, y_train_sync, cv=3, scoring="roc_auc"
        )

        print(f"\n🎯 Hiperparmetros optimizados:")
        print(f" 🏆 - Mejor score CV: {grid_search.best_score_:.4f}")
        print(f" 🏆 - Mejores parmetros: {grid_search.best_params_}")

        # Actualizar el mejor modelo con los parmetros optimizados
        optimized_model = grid_search.best_estimator_

        # Evaluar modelo optimizado
        y_pred_opt = optimized_model.predict(X_val)
        y_pred_proba_opt = optimized_model.predict_proba(X_val)[:, 1]

        # Corregir tipos de datos si es necesario
        if y_pred_opt.dtype == "object" or (
            len(y_pred_opt) > 0 and isinstance(y_pred_opt[0], str)
        ):
            y_pred_opt = np.where(y_pred_opt == "Yes", 1, 0)

        if y_val.dtype == "object" or (
            len(y_val) > 0
            and isinstance(y_val.iloc[0] if hasattr(y_val, "iloc") else y_val[0], str)
        ):
            if hasattr(y_val, "map"):
                y_val_numeric = y_val.map({"Yes": 1, "No": 0})
            else:
                y_val_numeric = np.where(y_val == "Yes", 1, 0)
            y_val = y_val_numeric

        # Calcular mtricas del modelo optimizado
        opt_auc = roc_auc_score(y_val, y_pred_proba_opt)
        opt_acc = accuracy_score(y_val, y_pred_opt)
        opt_precision = precision_score(y_val, y_pred_opt)
        opt_recall = recall_score(y_val, y_pred_opt)
        opt_f1 = f1_score(y_val, y_pred_opt)

        print(f"\n🎯 Mtricas del modelo optimizado:")
        print(f"   - ROC AUC: {opt_auc:.4f}")
        print(f"   - Accuracy: {opt_acc:.4f}")
        print(f"   - Precision: {opt_precision:.4f}")
        print(f"   - Recall: {opt_recall:.4f}")
        print(f"   - F1-Score: {opt_f1:.4f}")

        # Comparar con modelo original
        if "results" in locals():
            original_auc = results[best_model_name]["ROC_AUC"]
            improvement = opt_auc - original_auc

            print(f"\n📊 Comparacin con modelo original:")
            print(f"   - ROC AUC original: {original_auc:.4f}")
            print(f"   - ROC AUC optimizado: {opt_auc:.4f}")
            print(f"   - Mejora: {improvement:.4f} ({improvement*100:.2f}%)")

            if improvement > 0.001:
                print("✅ Optimizacin exitosa! El modelo mejor significativamente")
                best_model = optimized_model
                print("✅ Mejor modelo actualizado con la versin optimizada")
            else:
                print(" La optimizacin no produjo mejoras significativas")

        print("\n✅ Optimizacin de hiperparmetros completada")

    except Exception as e:
        print(f"⚠️ Error durante optimizacin de hiperparmetros: {e}")
        print(" Continuando con el modelo original...")

    # =======================================================================
    # 9. MLflow: Dashboard de Experimentos y Comparación de Modelos
    # =======================================================================

    print("📊 Generando dashboard de experimentos MLflow...")

    # Obtener todos los runs del experimento actual
    experiment = mlflow.get_experiment_by_name("Churn_Prediction_TP5")

    try:
        # Intentar obtener runs como DataFrame
        runs_data = mlflow.search_runs(experiment_ids=[experiment.experiment_id])

        print(f"\n🧪 Resumen del Experimento: {experiment.name}")
        print(f"   - Total de runs: {len(runs_data)}")
        print(f"   - Tracking URI: {mlflow.get_tracking_uri()}")

        # Verificar si runs_data es DataFrame o lista de Run objects
        if isinstance(runs_data, list):
            # Convertir lista de Run objects a DataFrame
            runs_list = []
            for run in runs_data:
                run_dict = {
                    "run_id": run.info.run_id,
                    "status": run.info.status,
                    "tags.model_type": run.data.tags.get("model_type", "Unknown"),
                    "metrics.roc_auc": run.data.metrics.get("roc_auc", 0),
                    "metrics.accuracy": run.data.metrics.get("accuracy", 0),
                    "metrics.f1_score": run.data.metrics.get("f1_score", 0),
                    "metrics.precision": run.data.metrics.get("precision", 0),
                    "metrics.recall": run.data.metrics.get("recall", 0),
                }
                runs_list.append(run_dict)
            runs_df = pd.DataFrame(runs_list)
        else:
            runs_df = runs_data

        # Mostrar top 5 modelos por ROC AUC
        if len(runs_df) > 0 and "metrics.roc_auc" in runs_df.columns:
            print("\n🏆 Top 5 Modelos por ROC AUC:")
            top_models = runs_df.nlargest(5, "metrics.roc_auc")[
                [
                    "tags.model_type",
                    "metrics.roc_auc",
                    "metrics.accuracy",
                    "metrics.f1_score",
                ]
            ]

            for i, (idx, row) in enumerate(top_models.iterrows(), 1):
                model_name = row.get("tags.model_type", "Unknown")
                roc_auc = row.get("metrics.roc_auc", 0)
                accuracy = row.get("metrics.accuracy", 0)
                f1_score = row.get("metrics.f1_score", 0)

                print(f"   {i}. {model_name}")
                print(f"      - ROC AUC: {roc_auc:.4f}")
                print(f"      - Accuracy: {accuracy:.4f}")
                print(f"      - F1-Score: {f1_score:.4f}")

            # Visualización comparativa
            plt.figure(figsize=(15, 5))

            # Subplot 1: ROC AUC Comparison
            plt.subplot(1, 3, 1)
            model_names = [
                name if len(name) <= 15 else name[:12] + "..."
                for name in top_models["tags.model_type"]
            ]
            plt.bar(model_names, top_models["metrics.roc_auc"], color="skyblue")
            plt.title("Comparación ROC AUC")
            plt.ylabel("ROC AUC Score")
            plt.xticks(rotation=45)
            plt.grid(axis="y", alpha=0.3)

            # Subplot 2: Accuracy Comparison
            plt.subplot(1, 3, 2)
            plt.bar(model_names, top_models["metrics.accuracy"], color="lightgreen")
            plt.title("Comparación Accuracy")
            plt.ylabel("Accuracy Score")
            plt.xticks(rotation=45)
            plt.grid(axis="y", alpha=0.3)

            # Subplot 3: F1-Score Comparison
            plt.subplot(1, 3, 3)
            plt.bar(model_names, top_models["metrics.f1_score"], color="salmon")
            plt.title("Comparación F1-Score")
            plt.ylabel("F1-Score")
            plt.xticks(rotation=45)
            plt.grid(axis="y", alpha=0.3)

            plt.tight_layout()
            plt.savefig("mlflow_model_comparison.png", dpi=150, bbox_inches="tight")
            plt.close()

            # Tabla comparativa detallada
            print("\n📋 Tabla Comparativa Detallada:")
            try:
                # Verificar que las columnas existan antes de seleccionarlas
                available_columns = [
                    "tags.model_type",
                    "metrics.roc_auc",
                    "metrics.accuracy",
                    "metrics.precision",
                    "metrics.recall",
                    "metrics.f1_score",
                ]
                existing_columns = [
                    col for col in available_columns if col in runs_df.columns
                ]

                if existing_columns:
                    comparison_df = runs_df[existing_columns].round(4)
                    # Crear nombres de columnas más legibles
                    column_mapping = {
                        "tags.model_type": "Model",
                        "metrics.roc_auc": "ROC_AUC",
                        "metrics.accuracy": "Accuracy",
                        "metrics.precision": "Precision",
                        "metrics.recall": "Recall",
                        "metrics.f1_score": "F1_Score",
                    }
                    comparison_df.columns = [
                        column_mapping.get(col, col) for col in existing_columns
                    ]
                    print(
                        comparison_df.sort_values("ROC_AUC", ascending=False)
                        if "ROC_AUC" in comparison_df.columns
                        else comparison_df
                    )
                else:
                    print("⚠️ No se encontraron columnas de métricas en los runs")
            except Exception as e:
                print(f"⚠️ Error generando tabla comparativa: {e}")

        else:
            print("⚠️ No se encontraron métricas de ROC AUC en los runs")
            if len(runs_df) > 0 and hasattr(runs_df, "columns"):
                print("📋 Columnas disponibles:")
                print(runs_df.columns.tolist())
            else:
                print("📋 Los runs no están en formato DataFrame o están vacíos")

    except Exception as e:
        print(f"⚠️ Error accediendo a los runs de MLflow: {e}")
        print("📋 Continuando sin dashboard de comparación...")

    # Información para acceder a MLflow UI
    print(f"\n🌐 Para acceder al dashboard completo de MLflow:")
    print(f"   1. Abre una terminal en la carpeta del proyecto")
    print(f"   2. Ejecuta: mlflow ui --backend-store-uri {mlflow.get_tracking_uri()}")
    print(f"   3. Abre tu navegador en: http://localhost:5000")
    print(f"   4. Busca el experimento: '{experiment.name}'")

    print("\n✅ Dashboard de experimentos generado correctamente")
    print("🎯 Ahora puedes comparar fácilmente todos tus modelos y experimentos!")

    # ========================================================================
    # 📋 10. RESUMEN FINAL
    # ========================================================================
    print("\n" + "=" * 60)
    print("✅ SCRIPT COMPLETADO EXITOSAMENTE")
    print("=" * 60)

    print(f"\n📊  Resumen de resultados:")
    print(f"   🏆 Mejor modelo: {best_model_name}")
    if "results" in locals():
        print(f"   📊 ROC AUC: {results[best_model_name]['ROC_AUC']:.4f}")
        print(f"   🎯 Accuracy: {results[best_model_name]['Accuracy']:.4f}")

    print(f"\n[FOLDER] Archivos generados:")
    print(f"   ✅ {submission_file} (archivo de submission)")
    print(f"   ✅ churn_distribution.png")
    if MLFLOW_AVAILABLE:
        print(f"   ✅ MLflow tracking data en: mlruns/")

    print(f"\n*** Prximos pasos:")
    print(f"   1. Revisa el archivo de submission generado")
    print(f"   2. Sube el archivo a Kaggle para evaluacin")
    if MLFLOW_AVAILABLE:
        print(f"   3. Ejecuta 'mlflow ui' para ver el dashboard de experimentos")

    # Cerrar el run principal de MLflow
    if MLFLOW_AVAILABLE and MLFLOW_TRACKING_ENABLED:
        try:
            # Log metricas finales del mejor modelo
            if "results" in locals() and best_model_name in results:
                mlflow.log_metric("final_roc_auc", results[best_model_name]["ROC_AUC"])
                mlflow.log_metric(
                    "final_accuracy", results[best_model_name]["Accuracy"]
                )
                mlflow.log_param("best_model", best_model_name)
                mlflow.log_param("submission_file", submission_file)

            mlflow.end_run()
            print(f"   ✅ Run principal cerrado exitosamente")
        except Exception as e:
            print(f"   ⚠️  Error cerrando run principal: {e}")

    print(f"\n✅ Script ejecutado exitosamente!")
    return True


if __name__ == "__main__":
    import os

    print("Ejecutando desde:", os.path.abspath(__file__))

    success = main()
    if success:
        print("\n🎯 Ejecución exitosa - Listo para Kaggle!")
    else:
        print("\n❌ Error en la ejecución - Revisa los logs")
        sys.exit(1)
