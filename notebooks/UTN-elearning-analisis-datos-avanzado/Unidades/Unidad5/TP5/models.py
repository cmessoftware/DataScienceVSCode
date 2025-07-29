"""
Módulo para crear y evaluar modelos de machine learning para predicción de churn.
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder,FunctionTransformer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
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

    def diagnose_missing_values(self, X: pd.DataFrame) -> None:
        """
        Diagnostica valores faltantes en el dataset.
        
        Args:
            X (pd.DataFrame): Dataset a analizar
        """
        print("🔍 DIAGNÓSTICO DE VALORES FALTANTES")
        print("=" * 50)
        
        missing_count = X.isnull().sum()
        missing_percent = (X.isnull().sum() / len(X)) * 100
        
        missing_info = pd.DataFrame({
            'Columna': X.columns,
            'Valores Faltantes': missing_count,
            'Porcentaje': missing_percent.round(2)
        })
        
        # Solo mostrar columnas con valores faltantes
        missing_info = missing_info[missing_info['Valores Faltantes'] > 0]
        
        if len(missing_info) == 0:
            print("✅ ¡Excelente! No hay valores faltantes en el dataset")
        else:
            print("⚠️ Se encontraron valores faltantes:")
            print(missing_info.to_string(index=False))
            print(f"\n📊 Total de filas: {len(X)}")
            print(f"📊 Filas con algún valor faltante: {X.isnull().any(axis=1).sum()}")
            print(f"📊 Filas completas: {len(X) - X.isnull().any(axis=1).sum()}")
            
        print("\n💡 El preprocesador maneja automáticamente estos valores faltantes:")
        print("   - Columnas numéricas: imputa con la mediana")
        print("   - Columnas categóricas: imputa con el valor más frecuente")

    def _normalize_service_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Reemplaza valores como 'No internet service' por 'No' en columnas de servicios opcionales.
        """
        df = df.copy()
        
        # Columnas que pueden tener 'No internet service'
        internet_service_cols = [
            'OnlineSecurity', 'OnlineBackup', 'DeviceProtection',
            'TechSupport', 'StreamingTV', 'StreamingMovies'
        ]
        
        # Columnas que pueden tener 'No phone service'  
        phone_service_cols = [
            'MultipleLines'
        ]
        
        # Normalizar servicios de internet
        for col in internet_service_cols:
            if col in df.columns:
                df[col] = df[col].replace('No internet service', 'No')
        
        # Normalizar servicios de teléfono
        for col in phone_service_cols:
            if col in df.columns:
                df[col] = df[col].replace('No phone service', 'No')
                
        return df

    def map_target(self, y, positive='Yes', negative='No', verbose=True):
        """
        Transforma una serie objetivo de 'Yes'/'No' a 1/0 si es necesario.
        """
        if verbose:
            print(f"🔍 Valores únicos antes del mapeo: {y.unique()}")
    
        # Si ya está mapeado, devolver tal cual
        if set(y.unique()) <= {0, 1}:
            if verbose:
                print("⚠️ La serie ya está mapeada. No se aplica transformación.")
            return y.astype(int)
    
        mapped = y.map({negative: 0, positive: 1})
    
        if mapped.isnull().any():
            raise ValueError(f"❌ Error: Se encontraron valores no mapeables: {y[mapped.isnull()].unique()}")
    
        if verbose:
            print(f"✅ Valores únicos después del mapeo: {mapped.unique()}")
            
        return mapped



    def _transform_yes_no(self, X):
        """
        Método privado que transforma columnas Yes/No a 1/0 desde numpy array, 
        manejando casos especiales.
        
        Esta función es usada por FunctionTransformer dentro del pipeline de preprocesamiento.
        Maneja casos especiales como 'No internet service' y 'No phone service'.
        """
        if isinstance(X, pd.DataFrame):
            X_transformed = X.copy()
            for col in X_transformed.columns:
                # Manejar casos especiales antes del mapeo
                X_transformed[col] = X_transformed[col].replace('No internet service', 'No')
                X_transformed[col] = X_transformed[col].replace('No phone service', 'No')
                # Mapear Yes/No → 1/0
                X_transformed[col] = X_transformed[col].map({'No': 0, 'Yes': 1}).fillna(0).astype('float64')
            return X_transformed
        else:
            # Si es numpy array, convertir cada columna
            X_transformed = X.copy()
            for col_idx in range(X.shape[1]):
                col_data = X[:, col_idx].copy()
                
                # Normalizar valores especiales a 'No'
                col_data = np.where(col_data == 'No internet service', 'No', col_data)
                col_data = np.where(col_data == 'No phone service', 'No', col_data)
                
                # Mapear Yes/No → 1/0, cualquier otro valor → 0
                transformed_col = np.where(col_data == 'Yes', 1.0,
                                         np.where(col_data == 'No', 0.0, 0.0))
                X_transformed[:, col_idx] = transformed_col.astype('float64')
            return X_transformed

    def inspect_transformed_columns(self, X_original: pd.DataFrame, columns: list, fit=True):
        """
        Muestra comparativa entre las columnas originales y sus transformaciones numéricas.

        Args:
            X_original (pd.DataFrame): Dataset original sin transformar.
            columns (list): Lista de columnas a inspeccionar.
            fit (bool): Si True, hace fit_transform; si False, solo transform.
        """
        print(f"🔍 Inspeccionando transformación de columnas: {columns}")
        
        # Asegurarse de usar solo las columnas seleccionadas
        X_subset = X_original.copy()
    
        # Fit o solo transform (según contexto)
        if fit:
            X_transformed = self.preprocessor.fit_transform(X_subset)
        else:
            X_transformed = self.preprocessor.transform(X_subset)
    
        # Obtener nombres de columnas transformadas con la nueva estructura de pipeline
        try:
            # Obtener las características de cada transformer
            num_transformer = self.preprocessor.named_transformers_['num']
            bin_transformer = self.preprocessor.named_transformers_['bin']
            cat_transformer = self.preprocessor.named_transformers_['cat']
            
            # Obtener las columnas de cada tipo
            num_cols = list(self.preprocessor.transformers_[0][2])
            bin_cols = list(self.preprocessor.transformers_[1][2])
            cat_cols = list(self.preprocessor.transformers_[2][2])
            
            # Para categorical features, obtener nombres después de one-hot encoding
            if len(cat_cols) > 0:
                cat_encoder = cat_transformer.named_steps['encoder']
                cat_names = cat_encoder.get_feature_names_out(cat_cols)
            else:
                cat_names = []
    
            final_cols = num_cols + bin_cols + list(cat_names)
        except Exception as e:
            print(f"⚠️ No se pudieron obtener nombres de columnas: {e}")
            # Fallback con nombres genéricos
            final_cols = [f"feature_{i}" for i in range(X_transformed.shape[1])]
    
        # Convertir a DataFrame
        df_transformed = pd.DataFrame(X_transformed, columns=final_cols)
    
        # Buscar todas las columnas transformadas que derivan de las columnas originales seleccionadas
        matched_cols = []
        for col in columns:
            # Coincidencias exactas o one-hot
            matched = [c for c in df_transformed.columns if col == c or c.startswith(col + "_")]
            matched_cols.extend(matched)
    
        # Mostrar comparación lado a lado
        print("\n🗂️  Valores originales:")
        print(X_subset[columns].head())
    
        print("⚙️  Valores transformados:")
        if matched_cols:
            print(df_transformed[matched_cols].head())
        else:
            print("⚠️ No se encontraron columnas coincidentes en los datos transformados")
            print(f"Columnas disponibles: {list(df_transformed.columns[:10])}...")  # Mostrar primeras 10
   
        
    def create_preprocessor(self, X: pd.DataFrame) -> ColumnTransformer:
        """
        Crea un preprocesador para las características del dataset Telco.
        
        Args:
            X (pd.DataFrame): Dataset de características
            
        Returns:
            ColumnTransformer: Preprocesador configurado
        """
        # 🔄 Normalizar valores de servicios
        X = self._normalize_service_values(X)

        # Identificar columnas
        id_cols = [col for col in X.columns if 'id' in col.lower()]
        numeric_features = X.select_dtypes(include=['int64', 'float64']).columns.difference(id_cols).tolist()
        object_features = X.select_dtypes(include='object').columns.difference(id_cols).tolist()

        # Detectar columnas binarias (solo Yes/No)
        binary_features = [
            col for col in object_features
            if set(X[col].dropna().unique()) <= {'Yes', 'No'}
        ]

        # Otras categóricas (multiclase)
        categorical_features = [col for col in object_features if col not in binary_features]

        # Usar el método privado para transformación binaria
        yes_no_transformer = FunctionTransformer(
            self._transform_yes_no,
            validate=False
        )

        # Construir el preprocesador con manejo de valores faltantes
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', Pipeline([
                    ('imputer', SimpleImputer(strategy='median')),
                    ('scaler', StandardScaler())
                ]), numeric_features),
                ('bin', Pipeline([
                    ('imputer', SimpleImputer(strategy='most_frequent')),
                    ('transformer', yes_no_transformer)
                ]), binary_features),
                ('cat', Pipeline([
                    ('imputer', SimpleImputer(strategy='most_frequent')),
                    ('encoder', OneHotEncoder(drop='first', handle_unknown='ignore'))
                ]), categorical_features),
            ],
            remainder='drop'
        )

        print(f"📊 Preprocesador configurado:")
        print(f"   - Numéricas: {numeric_features}")
        print(f"   - Binarias: {binary_features}")
        print(f"   - Categóricas: {categorical_features}")

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

    def train_model(self,model, X_train : pd.DataFrame, y_train: pd.Series) ->None:
        """
        Entrena un modelo en particular
        
        Args:
            model: modelo a entrenar
            X_train (pd.DataFrame): Características de entrenamiento
            y_train (pd.Series): Variable objetivo de entrenamiento
        """
        print(f"🎯 Entrenando modelos...")
        print(f"   - Muestras de entrenamiento: {len(X_train):,}")
        print(f"   - Características: {X_train.shape[1]}")

        print(f"   🔄 Entrenando {model}...")
        model.fit(X_train, y_train)
        print(f"   ✅ Modelo entrenado")
                
     
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

    def get_best_model(self, metric: str = 'ROC_AUC', results = None) -> Tuple[str, Pipeline]:
        """
        Obtiene el mejor modelo basado en una métrica.
        
        Args:
            metric (str): Métrica para seleccionar el mejor modelo
            
        Returns:
            Tuple[str, Pipeline]: Nombre y modelo del mejor
        """
        if not results:
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

# if __name__ == "__main__":
#     # Ejemplo de uso
#     print("🔄 Probando módulo de modelos...")
    
#     # Crear datos de ejemplo
#     np.random.seed(42)
#     n_samples = 1000
    
#     # Simular dataset
#     X = pd.DataFrame({
#         'feature1': np.random.normal(0, 1, n_samples),
#         'feature2': np.random.normal(0, 1, n_samples),
#         'category1': np.random.choice(['A', 'B', 'C'], n_samples),
#         'category2': np.random.choice(['X', 'Y'], n_samples)
#     })
    
#     y = np.random.choice([0, 1], n_samples, p=[0.7, 0.3])
    
#     # Dividir datos
#     split_idx = int(0.8 * len(X))
#     X_train, X_val = X[:split_idx], X[split_idx:]
#     y_train, y_val = y[:split_idx], y[split_idx:]
    
#     # Crear y evaluar modelos
#     predictor = ChurnPredictor()
#     predictor.create_preprocessor(X_train)
#     predictor.create_models()
#     predictor.train_models(X_train, y_train)
#     predictor.evaluate_models(X_val, y_val)
#     predictor.generate_model_report(X_val, y_val)
