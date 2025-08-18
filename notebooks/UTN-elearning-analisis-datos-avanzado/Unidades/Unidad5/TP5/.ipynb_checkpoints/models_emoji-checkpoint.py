#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🤖 Módulo de Modelos con Emojis para Predicción de Churn

✨ Versión mejorada del módulo models.py con emojis y visualización optimizada
📊 Incluye ChurnPredictor y funciones de optimización de hiperparámetros
"""

import warnings

warnings.filterwarnings("ignore")

import sys
import time

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler

# Importar tqdm para barra de progreso
# try:
#     from tqdm import tqdm

#     TQDM_AVAILABLE = True
# except ImportError:
#     TQDM_AVAILABLE = False
#     print("⚠️ tqdm no disponible - usando progress básico")


class ChurnPredictor:
    """
    🤖 Predictor de Churn con Emojis y Visualización Mejorada

    ✨ Clase principal para el modelado de predicción de abandono de clientes
    📊 Incluye preprocesamiento, entrenamiento y evaluación de modelos
    """

    def __init__(self, random_state=42):
        """
        🚀 Inicializar el predictor de churn

        Args:
            random_state (int): Semilla para reproducibilidad
        """
        self.random_state = random_state
        self.preprocessor = None
        self.models = {}
        self.results = {}

        print(f"🤖 ChurnPredictor inicializado con random_state={random_state}")

    def create_preprocessor(self, X_train):
        """
        ⚙️ Crear el preprocesador para las características

        Args:
            X_train: Dataset de entrenamiento

        Returns:
            ColumnTransformer: Preprocesador configurado
        """
        print("⚙️  Creando preprocesador...")

        # Identificar columnas categóricas y numéricas
        categorical_features = []
        numerical_features = []

        for col in X_train.columns:
            if X_train[col].dtype == "object":
                categorical_features.append(col)
            else:
                numerical_features.append(col)

        print(f"📋 Características categóricas: {categorical_features}")
        print(f"📊 Características numéricas: {numerical_features}")

        # Crear transformadores
        preprocessor = ColumnTransformer(
            transformers=[
                ("num", StandardScaler(), numerical_features),
                (
                    "cat",
                    OneHotEncoder(drop="first", handle_unknown="ignore"),
                    categorical_features,
                ),
            ]
        )

        self.preprocessor = preprocessor
        print("✅ Preprocesador creado exitosamente")
        return preprocessor

    def create_models(self):
        """
        🤖 Crear diccionario de modelos con emojis

        Returns:
            dict: Diccionario de modelos con pipelines
        """
        print("🤖  Creando modelos con emojis...")

        models = {
            "🔮 Logistic Regression": Pipeline(
                [
                    ("preprocessor", self.preprocessor),
                    (
                        "classifier",
                        LogisticRegression(
                            random_state=self.random_state, max_iter=1000
                        ),
                    ),
                ]
            ),
            "🌲 Random Forest": Pipeline(
                [
                    ("preprocessor", self.preprocessor),
                    (
                        "classifier",
                        RandomForestClassifier(
                            random_state=self.random_state, n_estimators=100
                        ),
                    ),
                ]
            ),
            "🧠 Naive Bayes": Pipeline(
                [("preprocessor", self.preprocessor), ("classifier", GaussianNB())]
            ),
            "👥 K-Nearest Neighbors": Pipeline(
                [
                    ("preprocessor", self.preprocessor),
                    ("classifier", KNeighborsClassifier(n_neighbors=5)),
                ]
            ),
        }

        self.models = models
        print(f"✅ {len(models)} modelos creados:")
        for name in models.keys():
            print(f"   🤖 {name}")

        return models

    def train_models(self, X_train, y_train):
        """
        🚂 Entrenar todos los modelos con barra de progreso

        Args:
            X_train: Características de entrenamiento
            y_train: Variable objetivo de entrenamiento
        """
        print("🚂 Iniciando entrenamiento de modelos...")
        print(f"📊 Total de modelos a entrenar: {len(self.models)}")

        # Variables para tracking de tiempo
        start_time = time.time()
        model_times = []
        successful_models = 0
        failed_models = 0

        # Configurar progress bar
        models_list = list(self.models.items())

        # if TQDM_AVAILABLE:
        #     progress_iterator = tqdm(
        #         models_list,
        #         desc="Entrenando modelos",  # SIN emojis en la descripción
        #         unit="modelo",
        #         bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]",
        #         ncols=80,  # Ancho fijo para mejor compatibilidad
        #         ascii=True,  # Usar caracteres ASCII para la barra
        #         leave=False,  # No dejar la barra al terminar
        #     )
        #     use_tqdm = True
        # else:
        #     progress_iterator = models_list
        #     use_tqdm = False

        for i, (name, model) in enumerate(progress_iterator, 1):
            model_start_time = time.time()

            try:
                # Actualizar descripción si usamos tqdm - SIN emojis
                if use_tqdm and hasattr(progress_iterator, "set_description"):
                    # Limpiar el nombre del modelo de emojis para tqdm
                    clean_name = (
                        name.replace("🔮", "")
                        .replace("🌲", "")
                        .replace("🧠", "")
                        .replace("👥", "")
                        .strip()
                    )
                    progress_iterator.set_description(
                        f"Entrenando {clean_name[:25]}..."
                    )
                else:
                    # Progress básico sin tqdm
                    elapsed = time.time() - start_time
                    if model_times:
                        avg_time = sum(model_times) / len(model_times)
                        remaining_models = len(self.models) - i + 1
                        estimated_remaining = avg_time * remaining_models
                        print(
                            f"🎯 [{i}/{len(self.models)}] Entrenando {name} (ETA: {estimated_remaining:.1f}s)"
                        )
                    else:
                        print(f"🎯 [{i}/{len(self.models)}] Entrenando {name}...")

                # Entrenar el modelo
                model.fit(X_train, y_train)

                # Calcular tiempo de entrenamiento
                model_time = time.time() - model_start_time
                model_times.append(model_time)
                successful_models += 1

                if use_tqdm and hasattr(progress_iterator, "set_postfix"):
                    progress_iterator.set_postfix(
                        {
                            "Último": f"{model_time:.1f}s",
                            "Exitosos": f"{successful_models}/{i}",
                        }
                    )
                else:
                    print(f"✅ {name} entrenado exitosamente en {model_time:.1f}s")

            except Exception as e:
                failed_models += 1
                model_time = time.time() - model_start_time
                model_times.append(model_time)

                if use_tqdm and hasattr(progress_iterator, "set_postfix"):
                    progress_iterator.set_postfix(
                        {
                            "Último": f"{model_time:.1f}s (ERROR)",
                            "Exitosos": f"{successful_models}/{i}",
                        }
                    )
                else:
                    print(f"❌ Error entrenando {name} en {model_time:.1f}s: {e}")

        # Resumen final
        total_time = time.time() - start_time
        avg_time = sum(model_times) / len(model_times) if model_times else 0

        # Forzar flush de la salida antes del resumen para limpiar tqdm
        if use_tqdm and TQDM_AVAILABLE:
            # Pequeña pausa para que tqdm termine completamente
            import time as time_module

            time_module.sleep(0.2)  # Pausa más larga para PowerShell

        # Limpiar completamente la salida
        sys.stdout.flush()
        sys.stderr.flush()

        # Limpiar la pantalla de cualquier residuo de tqdm
        print("\r" + " " * 100 + "\r", end="")  # Limpiar línea

        # Agregar línea en blanco para separar de tqdm
        print()

        print(f"🎉 ¡Entrenamiento completado!")
        print(f"⏱️ Tiempo total: {total_time:.1f}s")
        print(f"📊 Tiempo promedio por modelo: {avg_time:.1f}s")
        print(f"✅ Modelos exitosos: {successful_models}")
        if failed_models > 0:
            print(f"❌ Modelos fallidos: {failed_models}")
        print(f"🏆 Tasa de éxito: {(successful_models/len(self.models)*100):.1f}%")

        # Flush final agresivo
        sys.stdout.flush()
        sys.stderr.flush()

    def evaluate_models(self, X_test, y_test):
        """
        📊 Evaluar todos los modelos entrenados

        Args:
            X_test: Características de prueba
            y_test: Variable objetivo de prueba

        Returns:
            dict: Resultados de evaluación
        """
        print("📊Evaluando modelos...")

        results = {}

        for name, model in self.models.items():
            try:
                print(f"🔍  Evaluando {name}...")

                # Predicciones
                y_pred = model.predict(X_test)
                y_pred_proba = model.predict_proba(X_test)[:, 1]

                # Métricas
                accuracy = accuracy_score(y_test, y_pred)
                precision = precision_score(
                    y_test, y_pred, average="weighted", zero_division=0
                )
                recall = recall_score(
                    y_test, y_pred, average="weighted", zero_division=0
                )
                f1 = f1_score(y_test, y_pred, average="weighted", zero_division=0)
                roc_auc = roc_auc_score(y_test, y_pred_proba)

                results[name] = {
                    "Accuracy": accuracy,
                    "Precision": precision,
                    "Recall": recall,
                    "F1_Score": f1,
                    "ROC_AUC": roc_auc,
                }

                print(f"✅ {name} evaluado - ROC AUC: {roc_auc:.4f}")

            except Exception as e:
                print(f"❌  Error evaluando {name}: {e}")
                results[name] = {
                    "Accuracy": 0.0,
                    "Precision": 0.0,
                    "Recall": 0.0,
                    "F1_Score": 0.0,
                    "ROC_AUC": 0.0,
                }

        self.results = results
        print("📊¡Evaluación completada!")
        return results

    def get_best_model(self, metric="ROC_AUC", results=None):
        """
        🏆 Obtener el mejor modelo basado en una métrica

        Args:
            metric (str): Métrica para seleccionar el mejor modelo
            results (dict): Resultados de evaluación (opcional)

        Returns:
            tuple: (nombre_modelo, modelo)
        """
        if results is None:
            results = self.results

        print(f"🏆  Seleccionando mejor modelo por {metric}...")

        best_score = -1
        best_model_name = None

        for name, metrics in results.items():
            if metrics[metric] > best_score:
                best_score = metrics[metric]
                best_model_name = name

        if best_model_name is None:
            print("❌  No se encontró ningún modelo válido")
            return None, None

        best_model = self.models[best_model_name]

        print(f"🎉  Mejor modelo: {best_model_name} ({metric}: {best_score:.4f})")

        return best_model_name, best_model

    def generate_model_report(self, X_test, y_test):
        """
        📋 Generar reporte detallado de modelos

        Args:
            X_test: Características de prueba
            y_test: Variable objetivo de prueba
        """
        print("📋 Generando reporte de modelos...")

        print("\n" + "═" * 60)
        print("📊 REPORTE DETALLADO DE MODELOS")
        print("═" * 60)

        for name, model in self.models.items():
            try:
                print(f"\n🤖 MODELO: {name}")
                print("─" * 40)

                y_pred = model.predict(X_test)
                y_pred_proba = model.predict_proba(X_test)[:, 1]

                # Métricas básicas
                accuracy = accuracy_score(y_test, y_pred)
                precision = precision_score(
                    y_test, y_pred, average="weighted", zero_division=0
                )
                recall = recall_score(
                    y_test, y_pred, average="weighted", zero_division=0
                )
                f1 = f1_score(y_test, y_pred, average="weighted", zero_division=0)
                roc_auc = roc_auc_score(y_test, y_pred_proba)

                print(f"🎯 Accuracy:  {accuracy:.4f}")
                print(f"🔍 Precision: {precision:.4f}")
                print(f"📈 Recall:    {recall:.4f}")
                print(f"⚖️ F1-Score:  {f1:.4f}")
                print(f"🏆 ROC AUC:   {roc_auc:.4f}")

                # Matriz de confusión
                cm = confusion_matrix(y_test, y_pred)
                print(f"\n📊 Matriz de Confusión:")
                print(f"   TN: {cm[0,0]:4d} | FP: {cm[0,1]:4d}")
                print(f"   FN: {cm[1,0]:4d} | TP: {cm[1,1]:4d}")

            except Exception as e:
                print(f"❌  Error generando reporte para {name}: {e}")

        print("\n" + "═" * 60)
        print("✅ Reporte completado")

    def map_target(self, y):
        """
        🗺️ Mapear variable objetivo a formato numérico

        Args:
            y: Variable objetivo (puede ser 'Yes'/'No' o 0/1)

        Returns:
            Series/Array: Variable objetivo mapeada a 0/1
        """
        print("🗺️  Mapeando variable objetivo...")

        if hasattr(y, "dtype") and y.dtype == "object":
            # Si es texto, mapear a números
            if hasattr(y, "map"):
                y_mapped = y.map({"No": 0, "Yes": 1})
            else:
                y_mapped = np.where(y == "Yes", 1, 0)
            print("✅ Variable objetivo mapeada: 'No'→0, 'Yes'→1")
        else:
            # Si ya es numérico, mantener como está
            y_mapped = y
            print("✅ Variable objetivo ya es numérica")

        return y_mapped


def hyperparameter_tuning(model, param_grid, X_train, y_train, cv=5, scoring="roc_auc"):
    """
    🔧 Optimización de hiperparámetros con emojis

    Args:
        model: Modelo a optimizar
        param_grid (dict): Grilla de parámetros
        X_train: Características de entrenamiento
        y_train: Variable objetivo de entrenamiento
        cv (int): Número de folds para cross-validation
        scoring (str): Métrica de evaluación

    Returns:
        GridSearchCV: Objeto con resultados de búsqueda
    """
    print("🔧 Iniciando optimización de hiperparámetros...")
    print(f"📊 CV folds: {cv}, Scoring: {scoring}")
    print(f"⚙️ Grilla de parámetros: {len(param_grid)} parámetros")

    # Configurar GridSearchCV con n_jobs=1 para Windows
    grid_search = GridSearchCV(
        model,
        param_grid,
        cv=cv,
        scoring=scoring,
        n_jobs=1,  # Optimizado para Windows
        verbose=0,
        return_train_score=False,
    )

    print("🚂  Ejecutando búsqueda en grilla...")
    grid_search.fit(X_train, y_train)

    print(f"🏆 Mejor score: {grid_search.best_score_:.4f}")
    print(f"⚙️ Mejores parámetros: {grid_search.best_params_}")
    print("✅ Optimización completada")

    return grid_search


# Función de utilidad para mostrar información del módulo
def show_module_info():
    """
    ℹ️ Mostrar información del módulo
    """
    print("═" * 60)
    print("🤖 MÓDULO MODELS_EMOJI")
    print("═" * 60)
    print("✨ Versión mejorada con emojis y visualización optimizada")
    print("📊 Incluye:")
    print("   🤖 ChurnPredictor - Clase principal para modelado")
    print("   🔧 hyperparameter_tuning - Optimización de hiperparámetros")
    print("   📋 Funciones de reporte y evaluación")
    print("═" * 60)


if __name__ == "__main__":
    show_module_info()
