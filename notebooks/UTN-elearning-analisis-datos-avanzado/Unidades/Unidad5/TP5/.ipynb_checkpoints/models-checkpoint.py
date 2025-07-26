"""
Módulo para crear y evaluar modelos de machine learning para predicción de churn.
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.metrics import (roc_auc_score, accuracy_score, precision_score, 
                           recall_score, f1_score, classification_report, 
                           confusion_matrix, roc_curve)
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Any
import warnings
warnings.filterwarnings('ignore')

class ChurnPredictor:
    """
    Clase principal para entrenar y evaluar modelos de predicción de churn.
    """
    
    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        self.models = {}
        self.preprocessor = None
        self.results = {}
        
    def create_preprocessor(self, X: pd.DataFrame) -> ColumnTransformer:
        """
        Crea un preprocesador para las características.
        
        Args:
            X (pd.DataFrame): Dataset de características
            
        Returns:
            ColumnTransformer: Preprocesador configurado
        """
        
        # Identificar tipos de columnas
        numeric_features = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
        categorical_features = X.select_dtypes(include=['object']).columns.tolist()
        
        # Remover ID si existe
        id_cols = [col for col in X.columns if 'id' in col.lower()]
        for col in id_cols:
            if col in numeric_features:
                numeric_features.remove(col)
            if col in categorical_features:
                categorical_features.remove(col)
        
        print(f"📊 Preprocesador configurado:")
        print(f"   - Características numéricas: {len(numeric_features)}")
        print(f"   - Características categóricas: {len(categorical_features)}")
        
        # Crear transformadores
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', StandardScaler(), numeric_features),
                ('cat', OneHotEncoder(drop='first', handle_unknown='ignore'), categorical_features)
            ],
            remainder='drop'  # Eliminar columnas no especificadas
        )
        
        self.preprocessor = preprocessor
        return preprocessor
    
    def create_models(self) -> Dict[str, Pipeline]:
        """
        Crea los modelos de machine learning.
        
        Returns:
            Dict[str, Pipeline]: Diccionario con los modelos
        """
        
        models = {
            'Logistic_Regression': Pipeline([
                ('preprocessor', self.preprocessor),
                ('classifier', LogisticRegression(random_state=self.random_state, max_iter=1000))
            ]),
            
            'KNN': Pipeline([
                ('preprocessor', self.preprocessor),
                ('classifier', KNeighborsClassifier())
            ]),
            
            'Naive_Bayes': Pipeline([
                ('preprocessor', self.preprocessor),
                ('classifier', GaussianNB())
            ]),
            
            'Random_Forest': Pipeline([
                ('preprocessor', self.preprocessor),
                ('classifier', RandomForestClassifier(random_state=self.random_state, n_estimators=100))
            ])
        }
        
        self.models = models
        print(f"🤖 Modelos creados: {list(models.keys())}")
        return models
    
    def train_models(self, X_train: pd.DataFrame, y_train: pd.Series) -> None:
        """
        Entrena todos los modelos.
        
        Args:
            X_train (pd.DataFrame): Características de entrenamiento
            y_train (pd.Series): Variable objetivo de entrenamiento
        """
        
        print(f"🎯 Entrenando modelos...")
        print(f"   - Muestras de entrenamiento: {len(X_train):,}")
        print(f"   - Características: {X_train.shape[1]}")
        
        for name, model in self.models.items():
            print(f"   🔄 Entrenando {name}...")
            model.fit(X_train, y_train)
            print(f"   ✅ {name} entrenado")
        
        print(f"🎉 Todos los modelos entrenados exitosamente")
    
    def evaluate_models(self, X_val: pd.DataFrame, y_val: pd.Series) -> Dict[str, Dict[str, float]]:
        """
        Evalúa todos los modelos entrenados.
        
        Args:
            X_val (pd.DataFrame): Características de validación
            y_val (pd.Series): Variable objetivo de validación
            
        Returns:
            Dict[str, Dict[str, float]]: Métricas de evaluación
        """
        
        results = {}
        
        print(f"📊 Evaluando modelos...")
        print(f"   - Muestras de validación: {len(X_val):,}")
        
        for name, model in self.models.items():
            print(f"   🔍 Evaluando {name}...")
            
            # Predicciones
            y_pred = model.predict(X_val)
            y_pred_proba = model.predict_proba(X_val)[:, 1]
            
            # Calcular métricas
            metrics = {
                'ROC_AUC': roc_auc_score(y_val, y_pred_proba),
                'Accuracy': accuracy_score(y_val, y_pred),
                'Precision': precision_score(y_val, y_pred),
                'Recall': recall_score(y_val, y_pred),
                'F1_Score': f1_score(y_val, y_pred)
            }
            
            results[name] = metrics
            
            print(f"   ✅ {name} - ROC AUC: {metrics['ROC_AUC']:.4f}")
        
        self.results = results
        return results
    
    def get_best_model(self, metric: str = 'ROC_AUC') -> Tuple[str, Pipeline]:
        """
        Obtiene el mejor modelo basado en una métrica.
        
        Args:
            metric (str): Métrica para seleccionar el mejor modelo
            
        Returns:
            Tuple[str, Pipeline]: Nombre y modelo del mejor
        """
        
        if not self.results:
            raise ValueError("No hay resultados disponibles. Ejecuta evaluate_models() primero.")
        
        best_name = max(self.results.keys(), key=lambda x: self.results[x][metric])
        best_model = self.models[best_name]
        
        print(f"🏆 Mejor modelo según {metric}: {best_name}")
        print(f"   - {metric}: {self.results[best_name][metric]:.4f}")
        
        return best_name, best_model
    
    def plot_model_comparison(self) -> None:
        """
        Crea gráficos comparativos de los modelos.
        """
        
        if not self.results:
            print("❌ No hay resultados para plotear")
            return
        
        # Convertir resultados a DataFrame
        results_df = pd.DataFrame(self.results).T
        
        # Crear subplots
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        axes = axes.ravel()
        
        metrics = ['ROC_AUC', 'Accuracy', 'Precision', 'Recall', 'F1_Score']
        
        for i, metric in enumerate(metrics):
            ax = axes[i]
            
            # Ordenar por métrica
            sorted_results = results_df[metric].sort_values(ascending=True)
            
            # Crear gráfico de barras horizontal
            bars = ax.barh(range(len(sorted_results)), sorted_results.values)
            ax.set_yticks(range(len(sorted_results)))
            ax.set_yticklabels(sorted_results.index)
            ax.set_xlabel(metric)
            ax.set_title(f'Comparación - {metric}')
            
            # Añadir valores en las barras
            for j, (bar, value) in enumerate(zip(bars, sorted_results.values)):
                ax.text(value + 0.01, j, f'{value:.3f}', 
                       va='center', ha='left', fontsize=9)
            
            ax.set_xlim(0, 1.1)
            ax.grid(axis='x', alpha=0.3)
        
        # Ocultar subplot extra
        axes[5].set_visible(False)
        
        plt.tight_layout()
        plt.show()
    
    def plot_roc_curves(self, X_val: pd.DataFrame, y_val: pd.Series) -> None:
        """
        Plotea las curvas ROC de todos los modelos.
        
        Args:
            X_val (pd.DataFrame): Características de validación
            y_val (pd.Series): Variable objetivo de validación
        """
        
        plt.figure(figsize=(10, 8))
        
        for name, model in self.models.items():
            # Obtener probabilidades
            y_pred_proba = model.predict_proba(X_val)[:, 1]
            
            # Calcular curva ROC
            fpr, tpr, _ = roc_curve(y_val, y_pred_proba)
            auc_score = roc_auc_score(y_val, y_pred_proba)
            
            # Plotear
            plt.plot(fpr, tpr, label=f'{name} (AUC = {auc_score:.3f})', linewidth=2)
        
        # Línea diagonal (modelo aleatorio)
        plt.plot([0, 1], [0, 1], 'k--', alpha=0.5, label='Modelo Aleatorio')
        
        plt.xlabel('Tasa de Falsos Positivos')
        plt.ylabel('Tasa de Verdaderos Positivos')
        plt.title('Curvas ROC - Comparación de Modelos')
        plt.legend()
        plt.grid(alpha=0.3)
        plt.show()
    
    def generate_model_report(self, X_val: pd.DataFrame, y_val: pd.Series) -> None:
        """
        Genera un reporte completo de los modelos.
        
        Args:
            X_val (pd.DataFrame): Características de validación
            y_val (pd.Series): Variable objetivo de validación
        """
        
        print("📋 REPORTE COMPLETO DE MODELOS")
        print("=" * 80)
        
        # Tabla de resultados
        if self.results:
            results_df = pd.DataFrame(self.results).T
            print("\n📊 Métricas de Evaluación:")
            print(results_df.round(4))
            
            # Mejor modelo por métrica
            print(f"\n🏆 Mejores modelos por métrica:")
            for metric in results_df.columns:
                best_model = results_df[metric].idxmax()
                best_score = results_df[metric].max()
                print(f"   - {metric}: {best_model} ({best_score:.4f})")
        
        # Visualizaciones
        print(f"\n📈 Generando visualizaciones...")
        self.plot_model_comparison()
        self.plot_roc_curves(X_val, y_val)
        
        print(f"\n✅ Reporte completo generado")

def hyperparameter_tuning(model: Pipeline, 
                         param_grid: Dict[str, List[Any]], 
                         X_train: pd.DataFrame, 
                         y_train: pd.Series,
                         cv: int = 5,
                         scoring: str = 'roc_auc') -> GridSearchCV:
    """
    Realiza búsqueda de hiperparámetros.
    
    Args:
        model (Pipeline): Modelo a optimizar
        param_grid (Dict): Grilla de parámetros
        X_train (pd.DataFrame): Datos de entrenamiento
        y_train (pd.Series): Variable objetivo
        cv (int): Número de folds para cross-validation
        scoring (str): Métrica de evaluación
        
    Returns:
        GridSearchCV: Objeto con los mejores parámetros
    """
    
    print(f"🔧 Iniciando búsqueda de hiperparámetros...")
    print(f"   - Métrica: {scoring}")
    print(f"   - CV folds: {cv}")
    print(f"   - Combinaciones a probar: {np.prod([len(v) for v in param_grid.values()])}")
    
    grid_search = GridSearchCV(
        model, 
        param_grid, 
        cv=cv, 
        scoring=scoring, 
        n_jobs=-1,
        verbose=1
    )
    
    grid_search.fit(X_train, y_train)
    
    print(f"✅ Búsqueda completada")
    print(f"   - Mejor score: {grid_search.best_score_:.4f}")
    print(f"   - Mejores parámetros: {grid_search.best_params_}")
    
    return grid_search

def create_submission_file(model: Pipeline,
                          X_train: pd.DataFrame,
                          y_train: pd.Series,
                          X_test: pd.DataFrame,
                          test_ids: pd.Series,
                          filename: str = "submission.csv") -> pd.DataFrame:
    """
    Crea archivo de submission para Kaggle.
    
    Args:
        model (Pipeline): Modelo entrenado
        X_train (pd.DataFrame): Datos de entrenamiento completos
        y_train (pd.Series): Variable objetivo completa
        X_test (pd.DataFrame): Datos de prueba
        test_ids (pd.Series): IDs de los datos de prueba
        filename (str): Nombre del archivo de salida
        
    Returns:
        pd.DataFrame: DataFrame con las predicciones
    """
    
    print(f"📄 Creando archivo de submission...")
    
    # Re-entrenar con todos los datos
    print(f"   🔄 Re-entrenando modelo con datos completos...")
    model.fit(X_train, y_train)
    
    # Hacer predicciones
    print(f"   🎯 Generando predicciones...")
    predictions = model.predict_proba(X_test)[:, 1]
    
    # Crear DataFrame de submission
    submission_df = pd.DataFrame({
        'customerID': test_ids,
        'Churn': predictions
    })
    
    # Guardar archivo
    submission_df.to_csv(filename, index=False)
    
    print(f"   ✅ Archivo '{filename}' creado exitosamente")
    print(f"   📊 Estadísticas de predicciones:")
    print(f"      - Mínimo: {predictions.min():.4f}")
    print(f"      - Máximo: {predictions.max():.4f}")
    print(f"      - Media: {predictions.mean():.4f}")
    print(f"      - Mediana: {np.median(predictions):.4f}")
    
    return submission_df

if __name__ == "__main__":
    # Ejemplo de uso
    print("🔄 Probando módulo de modelos...")
    
    # Crear datos de ejemplo
    np.random.seed(42)
    n_samples = 1000
    
    # Simular dataset
    X = pd.DataFrame({
        'feature1': np.random.normal(0, 1, n_samples),
        'feature2': np.random.normal(0, 1, n_samples),
        'category1': np.random.choice(['A', 'B', 'C'], n_samples),
        'category2': np.random.choice(['X', 'Y'], n_samples)
    })
    
    y = np.random.choice([0, 1], n_samples, p=[0.7, 0.3])
    
    # Dividir datos
    split_idx = int(0.8 * len(X))
    X_train, X_val = X[:split_idx], X[split_idx:]
    y_train, y_val = y[:split_idx], y[split_idx:]
    
    # Crear y evaluar modelos
    predictor = ChurnPredictor()
    predictor.create_preprocessor(X_train)
    predictor.create_models()
    predictor.train_models(X_train, y_train)
    predictor.evaluate_models(X_val, y_val)
    predictor.generate_model_report(X_val, y_val)
