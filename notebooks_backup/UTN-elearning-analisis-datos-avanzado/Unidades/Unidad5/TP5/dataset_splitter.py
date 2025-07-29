"""
Módulo para dividir datasets en conjuntos de entrenamiento, validación y prueba.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from typing import Tuple, Optional

def split_data(df: pd.DataFrame, 
               target_col: str = 'Churn',
               test_size: float = 0.2,
               val_size: float = 0.2,
               random_state: int = 42,
               stratify: bool = True) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series]:
    """
    Divide el dataset en conjuntos de entrenamiento, validación y prueba.
    
    Args:
        df (pd.DataFrame): Dataset completo
        target_col (str): Nombre de la columna objetivo
        test_size (float): Proporción para el conjunto de prueba
        val_size (float): Proporción para el conjunto de validación (del total)
        random_state (int): Semilla para reproducibilidad
        stratify (bool): Si usar estratificación basada en el target
        
    Returns:
        tuple: (X_train, X_val, X_test, y_train, y_val, y_test)
    """
    
    print(f"📊 Dividiendo dataset...")
    print(f"   - Dataset original: {df.shape}")
    print(f"   - Test size: {test_size:.1%}")
    print(f"   - Validation size: {val_size:.1%}")
    print(f"   - Train size: {1 - test_size - val_size:.1%}")
    
    # Separar características y target
    X = df.drop(columns=[target_col])
    y = df[target_col]
    
    # Configurar estratificación
    stratify_param = y if stratify else None
    
    # Primera división: separar conjunto de prueba
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        stratify=stratify_param
    )
    
    # Segunda división: separar entrenamiento y validación
    val_size_adjusted = val_size / (1 - test_size)  # Ajustar proporción
    stratify_temp = y_temp if stratify else None
    
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp,
        test_size=val_size_adjusted,
        random_state=random_state,
        stratify=stratify_temp
    )
    
    print(f"\n✅ División completada:")
    print(f"   - Train: {X_train.shape[0]} muestras ({X_train.shape[0]/len(df):.1%})")
    print(f"   - Validation: {X_val.shape[0]} muestras ({X_val.shape[0]/len(df):.1%})")
    print(f"   - Test: {X_test.shape[0]} muestras ({X_test.shape[0]/len(df):.1%})")
    
    # Verificar distribución del target si es estratificado
    if stratify:
        print(f"\n🎯 Distribución del target:")
        print(f"   - Original: {y.value_counts(normalize=True).round(3).to_dict()}")
        print(f"   - Train: {y_train.value_counts(normalize=True).round(3).to_dict()}")
        print(f"   - Validation: {y_val.value_counts(normalize=True).round(3).to_dict()}")
        print(f"   - Test: {y_test.value_counts(normalize=True).round(3).to_dict()}")
    
    return X_train, X_val, X_test, y_train, y_val, y_test

def create_time_based_split(df: pd.DataFrame,
                           date_col: str,
                           target_col: str = 'Churn',
                           train_months: int = 18,
                           val_months: int = 3,
                           test_months: int = 3) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series]:
    """
    Crea una división basada en tiempo para datos temporales.
    
    Args:
        df (pd.DataFrame): Dataset con columna de fecha
        date_col (str): Nombre de la columna de fecha
        target_col (str): Nombre de la columna objetivo
        train_months (int): Meses para entrenamiento
        val_months (int): Meses para validación
        test_months (int): Meses para prueba
        
    Returns:
        tuple: (X_train, X_val, X_test, y_train, y_val, y_test)
    """
    
    print(f"📅 División temporal del dataset...")
    
    # Convertir a datetime si no lo es
    df[date_col] = pd.to_datetime(df[date_col])
    df_sorted = df.sort_values(date_col)
    
    # Calcular puntos de corte
    total_months = train_months + val_months + test_months
    train_cutoff = len(df_sorted) * train_months // total_months
    val_cutoff = len(df_sorted) * (train_months + val_months) // total_months
    
    # Dividir datasets
    train_data = df_sorted.iloc[:train_cutoff]
    val_data = df_sorted.iloc[train_cutoff:val_cutoff]
    test_data = df_sorted.iloc[val_cutoff:]
    
    # Separar características y target
    X_train = train_data.drop(columns=[target_col])
    y_train = train_data[target_col]
    
    X_val = val_data.drop(columns=[target_col])
    y_val = val_data[target_col]
    
    X_test = test_data.drop(columns=[target_col])
    y_test = test_data[target_col]
    
    print(f"✅ División temporal completada:")
    print(f"   - Train: {len(X_train)} muestras")
    print(f"   - Validation: {len(X_val)} muestras")
    print(f"   - Test: {len(X_test)} muestras")
    
    return X_train, X_val, X_test, y_train, y_val, y_test

def get_split_info(X_train, X_val, X_test, y_train, y_val, y_test):
    """
    Muestra información detallada sobre la división de datos.
    
    Args:
        X_train, X_val, X_test: Conjuntos de características
        y_train, y_val, y_test: Conjuntos de targets
    """
    
    print("\n📊 INFORMACIÓN DETALLADA DE LA DIVISIÓN")
    print("=" * 50)
    
    total_samples = len(X_train) + len(X_val) + len(X_test)
    
    datasets = [
        ("Entrenamiento", X_train, y_train),
        ("Validación", X_val, y_val),
        ("Prueba", X_test, y_test)
    ]
    
    for name, X, y in datasets:
        print(f"\n🔍 {name}:")
        print(f"   - Muestras: {len(X):,} ({len(X)/total_samples:.1%})")
        print(f"   - Características: {X.shape[1]}")
        
        if len(y.unique()) <= 10:  # Solo para variables categóricas
            dist = y.value_counts(normalize=True).round(3)
            print(f"   - Distribución target: {dist.to_dict()}")
        
        # Información sobre tipos de datos
        numeric_cols = X.select_dtypes(include=[np.number]).shape[1]
        categorical_cols = X.select_dtypes(include=['object']).shape[1]
        print(f"   - Columnas numéricas: {numeric_cols}")
        print(f"   - Columnas categóricas: {categorical_cols}")

class DataSplitter:
    """
    Clase para manejar múltiples estrategias de división de datos.
    """
    
    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        self.splits_history = []
    
    def random_split(self, df: pd.DataFrame, **kwargs) -> Tuple:
        """Ejecuta división aleatoria."""
        result = split_data(df, random_state=self.random_state, **kwargs)
        self.splits_history.append(("random", kwargs))
        return result
    
    def temporal_split(self, df: pd.DataFrame, **kwargs) -> Tuple:
        """Ejecuta división temporal."""
        result = create_time_based_split(df, **kwargs)
        self.splits_history.append(("temporal", kwargs))
        return result
    
    def get_history(self):
        """Retorna el historial de divisiones realizadas."""
        return self.splits_history

if __name__ == "__main__":
    # Ejemplo de uso
    from data_loader import create_sample_data
    
    print("🔄 Probando división de datos...")
    
    # Crear datos de ejemplo
    train_df, _, _ = create_sample_data(n_train=1000)
    
    # Probar división aleatoria
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(
        train_df, 
        target_col='Churn',
        test_size=0.2,
        val_size=0.2
    )
    
    # Mostrar información
    get_split_info(X_train, X_val, X_test, y_train, y_val, y_test)
