"""
Módulo para análisis exploratorio de datos (EDA) del proyecto de churn.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Optional, Tuple
import warnings
warnings.filterwarnings('ignore')

# Configuración global para visualizaciones
plt.style.use('default')
sns.set_palette("husl")

def basic_info(df: pd.DataFrame, name: str = "Dataset") -> None:
    """
    Muestra información básica del dataset.
    
    Args:
        df (pd.DataFrame): Dataset a analizar
        name (str): Nombre descriptivo del dataset
    """
    print(f"📊 INFORMACIÓN BÁSICA DE {name.upper()}")
    print("=" * 60)
    
    print(f"Dimensiones: {df.shape[0]:,} filas × {df.shape[1]} columnas")
    print(f"Memoria utilizada: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
    
    print(f"\n🔍 Tipos de datos:")
    dtype_counts = df.dtypes.value_counts()
    for dtype, count in dtype_counts.items():
        print(f"   - {dtype}: {count} columnas")
    
    print(f"\n⚠️ Valores faltantes:")
    missing = df.isnull().sum()
    total_missing = missing.sum()
    
    if total_missing > 0:
        missing_pct = (missing / len(df)) * 100
        missing_df = pd.DataFrame({
            'Columna': missing.index,
            'Faltantes': missing.values,
            'Porcentaje': missing_pct.values
        })
        missing_df = missing_df[missing_df['Faltantes'] > 0].sort_values('Faltantes', ascending=False)
        print(missing_df.to_string(index=False))
    else:
        print("   ✅ No hay valores faltantes")
    
    print(f"\n📋 Resumen de columnas:")
    print(f"   - Numéricas: {len(df.select_dtypes(include=[np.number]).columns)}")
    print(f"   - Categóricas: {len(df.select_dtypes(include=['object']).columns)}")
    print(f"   - Booleanas: {len(df.select_dtypes(include=['bool']).columns)}")

def analyze_target(df: pd.DataFrame, target_col: str = 'Churn') -> None:
    """
    Analiza la variable objetivo.
    
    Args:
        df (pd.DataFrame): Dataset con la variable objetivo
        target_col (str): Nombre de la columna objetivo
    """
    print(f"🎯 ANÁLISIS DE VARIABLE OBJETIVO: {target_col}")
    print("=" * 60)
    
    if target_col not in df.columns:
        print(f"❌ La columna '{target_col}' no existe en el dataset")
        return
    
    # Contar valores
    value_counts = df[target_col].value_counts().sort_index()
    value_pcts = df[target_col].value_counts(normalize=True).sort_index() * 100
    
    print(f"Distribución de {target_col}:")
    for value, count, pct in zip(value_counts.index, value_counts.values, value_pcts.values):
        print(f"   - {value}: {count:,} ({pct:.1f}%)")
    
    # Calcular desbalance
    if len(value_counts) == 2:
        minority_pct = min(value_pcts.values)
        if minority_pct < 30:
            print(f"\n⚠️ Dataset desbalanceado: clase minoritaria {minority_pct:.1f}%")
        else:
            print(f"\n✅ Dataset balanceado: diferencia {abs(value_pcts.iloc[0] - value_pcts.iloc[1]):.1f}%")
    
    # Visualización
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    
    # Gráfico de barras
    value_counts.plot(kind='bar', ax=axes[0], color=['lightblue', 'salmon'])
    axes[0].set_title(f'Distribución de {target_col}')
    axes[0].set_xlabel(target_col)
    axes[0].set_ylabel('Frecuencia')
    axes[0].tick_params(axis='x', rotation=0)
    
    # Gráfico de torta
    value_pcts.plot(kind='pie', ax=axes[1], autopct='%1.1f%%', colors=['lightblue', 'salmon'])
    axes[1].set_title(f'Proporción de {target_col}')
    axes[1].set_ylabel('')
    
    plt.tight_layout()
    plt.show()

def analyze_numerical_features(df: pd.DataFrame, target_col: Optional[str] = None) -> None:
    """
    Analiza las características numéricas del dataset.
    
    Args:
        df (pd.DataFrame): Dataset a analizar
        target_col (str, optional): Columna objetivo para análisis bivariado
    """
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if target_col and target_col in numeric_cols:
        numeric_cols.remove(target_col)
    
    if not numeric_cols:
        print("❌ No hay columnas numéricas para analizar")
        return
    
    print(f"📈 ANÁLISIS DE CARACTERÍSTICAS NUMÉRICAS")
    print("=" * 60)
    print(f"Columnas numéricas encontradas: {len(numeric_cols)}")
    
    # Estadísticas descriptivas
    print(f"\n📊 Estadísticas descriptivas:")
    stats = df[numeric_cols].describe()
    print(stats.round(2))
    
    # Detectar outliers usando IQR
    print(f"\n🔍 Detección de outliers (método IQR):")
    for col in numeric_cols:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)]
        outlier_pct = len(outliers) / len(df) * 100
        
        print(f"   - {col}: {len(outliers)} outliers ({outlier_pct:.1f}%)")
    
    # Visualizaciones
    n_cols = min(len(numeric_cols), 4)
    n_rows = (len(numeric_cols) + n_cols - 1) // n_cols
    
    if target_col and target_col in df.columns:
        # Boxplots por target
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(4*n_cols, 4*n_rows))
        if n_rows == 1:
            axes = axes.reshape(1, -1) if n_cols > 1 else [axes]
        
        for i, col in enumerate(numeric_cols):
            row, col_idx = divmod(i, n_cols)
            ax = axes[row][col_idx] if n_cols > 1 else axes[row]
            
            df.boxplot(column=col, by=target_col, ax=ax)
            ax.set_title(f'{col} por {target_col}')
            ax.set_xlabel(target_col)
        
        # Ocultar subplots vacíos
        for i in range(len(numeric_cols), n_rows * n_cols):
            row, col_idx = divmod(i, n_cols)
            axes[row][col_idx].set_visible(False)
        
        plt.tight_layout()
        plt.show()
    
    # Histogramas
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4*n_cols, 3*n_rows))
    if n_rows == 1:
        axes = axes.reshape(1, -1) if n_cols > 1 else [axes]
    
    for i, col in enumerate(numeric_cols):
        row, col_idx = divmod(i, n_cols)
        ax = axes[row][col_idx] if n_cols > 1 else axes[row]
        
        df[col].hist(bins=30, ax=ax, alpha=0.7)
        ax.set_title(f'Distribución de {col}')
        ax.set_xlabel(col)
        ax.set_ylabel('Frecuencia')
    
    # Ocultar subplots vacíos
    for i in range(len(numeric_cols), n_rows * n_cols):
        row, col_idx = divmod(i, n_cols)
        axes[row][col_idx].set_visible(False)
    
    plt.tight_layout()
    plt.show()

def analyze_categorical_features(df: pd.DataFrame, target_col: Optional[str] = None) -> None:
    """
    Analiza las características categóricas del dataset.
    
    Args:
        df (pd.DataFrame): Dataset a analizar
        target_col (str, optional): Columna objetivo para análisis bivariado
    """
    categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
    if target_col and target_col in categorical_cols:
        categorical_cols.remove(target_col)
    
    if not categorical_cols:
        print("❌ No hay columnas categóricas para analizar")
        return
    
    print(f"📊 ANÁLISIS DE CARACTERÍSTICAS CATEGÓRICAS")
    print("=" * 60)
    print(f"Columnas categóricas encontradas: {len(categorical_cols)}")
    
    # Análisis de cardinalidad
    print(f"\n🔢 Cardinalidad de variables categóricas:")
    for col in categorical_cols:
        unique_count = df[col].nunique()
        unique_pct = unique_count / len(df) * 100
        most_frequent = df[col].mode().iloc[0] if len(df[col].mode()) > 0 else "N/A"
        most_frequent_count = df[col].value_counts().iloc[0] if len(df) > 0 else 0
        most_frequent_pct = most_frequent_count / len(df) * 100
        
        print(f"   - {col}: {unique_count} valores únicos ({unique_pct:.1f}%)")
        print(f"     Más frecuente: '{most_frequent}' ({most_frequent_count}, {most_frequent_pct:.1f}%)")
    
    # Visualizaciones
    n_cols = min(len(categorical_cols), 2)
    n_rows = (len(categorical_cols) + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(6*n_cols, 4*n_rows))
    if n_rows == 1:
        axes = axes.reshape(1, -1) if n_cols > 1 else [axes]
    
    for i, col in enumerate(categorical_cols):
        row, col_idx = divmod(i, n_cols)
        ax = axes[row][col_idx] if n_cols > 1 else axes[row]
        
        # Limitar a top 10 categorías si hay muchas
        value_counts = df[col].value_counts()
        if len(value_counts) > 10:
            value_counts = value_counts.head(10)
            title = f'Top 10 categorías de {col}'
        else:
            title = f'Distribución de {col}'
        
        value_counts.plot(kind='bar', ax=ax)
        ax.set_title(title)
        ax.set_xlabel('')
        ax.set_ylabel('Frecuencia')
        ax.tick_params(axis='x', rotation=45)
    
    # Ocultar subplots vacíos
    for i in range(len(categorical_cols), n_rows * n_cols):
        row, col_idx = divmod(i, n_cols)
        axes[row][col_idx].set_visible(False)
    
    plt.tight_layout()
    plt.show()
    
    # Análisis bivariado con target si está disponible
    if target_col and target_col in df.columns:
        print(f"\n🎯 Relación con variable objetivo ({target_col}):")
        
        for col in categorical_cols[:5]:  # Limitar a 5 variables para no saturar
            print(f"\n   📊 {col} vs {target_col}:")
            
            # Tabla de contingencia
            crosstab = pd.crosstab(df[col], df[target_col], normalize='index') * 100
            print(crosstab.round(1))

def correlation_analysis(df: pd.DataFrame, target_col: Optional[str] = None, threshold: float = 0.7) -> None:
    """
    Analiza correlaciones entre variables numéricas.
    
    Args:
        df (pd.DataFrame): Dataset a analizar
        target_col (str, optional): Columna objetivo
        threshold (float): Umbral para detectar correlaciones altas
    """
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    
    if len(numeric_cols) < 2:
        print("❌ Se necesitan al menos 2 columnas numéricas para análisis de correlación")
        return
    
    print(f"🔗 ANÁLISIS DE CORRELACIONES")
    print("=" * 60)
    
    # Calcular matriz de correlación
    corr_matrix = df[numeric_cols].corr()
    
    # Detectar correlaciones altas
    high_corr_pairs = []
    for i in range(len(corr_matrix.columns)):
        for j in range(i+1, len(corr_matrix.columns)):
            corr_val = abs(corr_matrix.iloc[i, j])
            if corr_val >= threshold:
                high_corr_pairs.append((
                    corr_matrix.columns[i],
                    corr_matrix.columns[j],
                    corr_matrix.iloc[i, j]
                ))
    
    if high_corr_pairs:
        print(f"⚠️ Correlaciones altas (|r| >= {threshold}):")
        for var1, var2, corr in high_corr_pairs:
            print(f"   - {var1} ↔ {var2}: {corr:.3f}")
    else:
        print(f"✅ No se encontraron correlaciones altas (|r| >= {threshold})")
    
    # Correlaciones con variable objetivo
    if target_col and target_col in numeric_cols:
        target_corrs = corr_matrix[target_col].drop(target_col).abs().sort_values(ascending=False)
        print(f"\n🎯 Correlaciones con {target_col} (ordenadas por magnitud):")
        for var, corr in target_corrs.items():
            print(f"   - {var}: {corr:.3f}")
    
    # Visualización del mapa de calor
    plt.figure(figsize=(10, 8))
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
    sns.heatmap(corr_matrix, 
                mask=mask,
                annot=True, 
                cmap='coolwarm', 
                center=0,
                square=True,
                fmt='.2f',
                cbar_kws={"shrink": .8})
    plt.title('Matriz de Correlaciones')
    plt.tight_layout()
    plt.show()

def generate_eda_report(df: pd.DataFrame, target_col: str = 'Churn') -> None:
    """
    Genera un reporte completo de EDA.
    
    Args:
        df (pd.DataFrame): Dataset a analizar
        target_col (str): Nombre de la columna objetivo
    """
    print("🔍 GENERANDO REPORTE COMPLETO DE EDA")
    print("=" * 80)
    
    # Información básica
    basic_info(df, "Dataset Principal")
    print("\n" + "="*80 + "\n")
    
    # Análisis de variable objetivo
    analyze_target(df, target_col)
    print("\n" + "="*80 + "\n")
    
    # Análisis de características numéricas
    analyze_numerical_features(df, target_col)
    print("\n" + "="*80 + "\n")
    
    # Análisis de características categóricas
    analyze_categorical_features(df, target_col)
    print("\n" + "="*80 + "\n")
    
    # Análisis de correlaciones
    correlation_analysis(df, target_col)
    print("\n" + "="*80 + "\n")
    
    print("✅ Reporte de EDA completado")

if __name__ == "__main__":
    # Ejemplo de uso
    print("🔄 Probando módulo de EDA...")
    
    # Crear datos de ejemplo (normalmente vendrían de data_loader)
    np.random.seed(42)
    n_samples = 1000
    
    sample_df = pd.DataFrame({
        'customerID': [f'C{i:04d}' for i in range(n_samples)],
        'tenure': np.random.randint(0, 73, n_samples),
        'MonthlyCharges': np.random.uniform(18, 120, n_samples),
        'TotalCharges': np.random.uniform(18, 8000, n_samples),
        'Contract': np.random.choice(['Month-to-month', 'One year', 'Two year'], n_samples),
        'InternetService': np.random.choice(['DSL', 'Fiber optic', 'No'], n_samples),
        'Churn': np.random.choice([0, 1], n_samples, p=[0.7, 0.3])
    })
    
    # Ejecutar EDA
    generate_eda_report(sample_df, 'Churn')
