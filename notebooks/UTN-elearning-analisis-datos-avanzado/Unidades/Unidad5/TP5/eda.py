"""
Módulo para análisis exploratorio de datos (EDA) del proyecto de churn.
"""

import warnings
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

warnings.filterwarnings("ignore")

# Configuración global para visualizaciones
plt.style.use("default")
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

    print("\n🔍 Tipos de datos:")
    dtype_counts = df.dtypes.value_counts()
    for dtype, count in dtype_counts.items():
        print(f"   - {dtype}: {count} columnas")

    print("\n⚠️ Valores faltantes:")
    missing = df.isnull().sum()
    total_missing = missing.sum()

    if total_missing > 0:
        missing_pct = (missing / len(df)) * 100
        missing_df = pd.DataFrame(
            {
                "Columna": missing.index,
                "Faltantes": missing.values,
                "Porcentaje": missing_pct.values,
            }
        )
        missing_df = missing_df[missing_df["Faltantes"] > 0].sort_values(
            "Faltantes", ascending=False
        )
        print(missing_df.to_string(index=False))
    else:
        print("   ✅ No hay valores faltantes")

    print("\n📋 Resumen de columnas:")
    print(f"   - Numéricas: {len(df.select_dtypes(include=[np.number]).columns)}")
    print(f"   - Categóricas: {len(df.select_dtypes(include=['object']).columns)}")
    print(f"   - Booleanas: {len(df.select_dtypes(include=['bool']).columns)}")

    # Información general del dataset
    print("📊 INFORMACIÓN GENERAL DEL DATASET")
    print("=" * 50)
    print(f"Dimensiones: {df.shape}")
    print(f"Columnas: {list(df.columns)}")
    print("\n📋 Tipos de datos:")
    print(df.dtypes)

    # Información sobre valores faltantes
    print("\n🔍 VALORES FALTANTES:")
    print("=" * 30)
    missing_values = df.isnull().sum()
    print(missing_values[missing_values > 0])
    if missing_values.sum() == 0:
        print("✅ No hay valores faltantes")


def analyze_target(df: pd.DataFrame, target_col: str = "Churn") -> None:
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
    for value, count, pct in zip(
        value_counts.index, value_counts.values, value_pcts.values
    ):
        print(f"   - {value}: {count:,} ({pct:.1f}%)")

    # Calcular desbalance
    if len(value_counts) == 2:
        minority_pct = min(value_pcts.values)
        if minority_pct < 30:
            print(f"\n⚠️ Dataset desbalanceado: clase minoritaria {minority_pct:.1f}%")
        else:
            print(
                f"\n✅ Dataset balanceado: diferencia {abs(value_pcts.iloc[0] - value_pcts.iloc[1]):.1f}%"
            )

    # Visualización
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # Gráfico de barras
    value_counts.plot(kind="bar", ax=axes[0], color=["lightblue", "salmon"])
    axes[0].set_title(f"Distribución de {target_col}")
    axes[0].set_xlabel(target_col)
    axes[0].set_ylabel("Frecuencia")
    axes[0].tick_params(axis="x", rotation=0)

    # Gráfico de torta
    value_pcts.plot(
        kind="pie", ax=axes[1], autopct="%1.1f%%", colors=["lightblue", "salmon"]
    )
    axes[1].set_title(f"Proporción de {target_col}")
    axes[1].set_ylabel("")

    plt.tight_layout()
    plt.show()


def analyze_numerical_features(
    df: pd.DataFrame, target_col: Optional[str] = None
) -> None:
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

    print("📈 ANÁLISIS DE CARACTERÍSTICAS NUMÉRICAS")
    print("=" * 60)
    print(f"Columnas numéricas encontradas: {len(numeric_cols)}")

    # Estadísticas descriptivas
    print("\n📊 Estadísticas descriptivas:")
    stats = df[numeric_cols].describe()
    print(stats.round(2))

    # Detectar outliers usando IQR
    print("\n🔍 Detección de outliers (método IQR):")
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
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 4 * n_rows))
        if n_rows == 1:
            axes = axes.reshape(1, -1) if n_cols > 1 else [axes]

        for i, col in enumerate(numeric_cols):
            row, col_idx = divmod(i, n_cols)
            ax = axes[row][col_idx] if n_cols > 1 else axes[row]

            df.boxplot(column=col, by=target_col, ax=ax)
            ax.set_title(f"{col} por {target_col}")
            ax.set_xlabel(target_col)

        # Ocultar subplots vacíos
        for i in range(len(numeric_cols), n_rows * n_cols):
            row, col_idx = divmod(i, n_cols)
            axes[row][col_idx].set_visible(False)

        plt.tight_layout()
        plt.show()

    # Histogramas
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 3 * n_rows))
    if n_rows == 1:
        axes = axes.reshape(1, -1) if n_cols > 1 else [axes]

    for i, col in enumerate(numeric_cols):
        row, col_idx = divmod(i, n_cols)
        ax = axes[row][col_idx] if n_cols > 1 else axes[row]

        df[col].hist(bins=30, ax=ax, alpha=0.7)
        ax.set_title(f"Distribución de {col}")
        ax.set_xlabel(col)
        ax.set_ylabel("Frecuencia")

    # Ocultar subplots vacíos
    for i in range(len(numeric_cols), n_rows * n_cols):
        row, col_idx = divmod(i, n_cols)
        axes[row][col_idx].set_visible(False)

    plt.tight_layout()
    plt.show()


def analyze_categorical_features(
    df: pd.DataFrame, target_col: Optional[str] = None
) -> None:
    """
    Analiza las características categóricas del dataset.

    Args:
        df (pd.DataFrame): Dataset a analizar
        target_col (str, optional): Columna objetivo para análisis bivariado
    """
    categorical_cols = df.select_dtypes(include=["object"]).columns.tolist()
    if target_col and target_col in categorical_cols:
        categorical_cols.remove(target_col)

    if not categorical_cols:
        print("❌ No hay columnas categóricas para analizar")
        return

    print("📊 ANÁLISIS DE CARACTERÍSTICAS CATEGÓRICAS")
    print("=" * 60)
    print(f"Columnas categóricas encontradas: {len(categorical_cols)}")

    # Análisis de cardinalidad
    print("\n🔢 Cardinalidad de variables categóricas:")
    for col in categorical_cols:
        unique_count = df[col].nunique()
        unique_pct = unique_count / len(df) * 100
        most_frequent = df[col].mode().iloc[0] if len(df[col].mode()) > 0 else "N/A"
        most_frequent_count = df[col].value_counts().iloc[0] if len(df) > 0 else 0
        most_frequent_pct = most_frequent_count / len(df) * 100

        print(f"   - {col}: {unique_count} valores únicos ({unique_pct:.1f}%)")
        print(
            f"     Más frecuente: '{most_frequent}' ({most_frequent_count}, {most_frequent_pct:.1f}%)"
        )

    # Visualizaciones
    n_cols = min(len(categorical_cols), 2)
    n_rows = (len(categorical_cols) + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 4 * n_rows))
    if n_rows == 1:
        axes = axes.reshape(1, -1) if n_cols > 1 else [axes]

    for i, col in enumerate(categorical_cols):
        row, col_idx = divmod(i, n_cols)
        ax = axes[row][col_idx] if n_cols > 1 else axes[row]

        # Limitar a top 10 categorías si hay muchas
        value_counts = df[col].value_counts()
        if len(value_counts) > 10:
            value_counts = value_counts.head(10)
            title = f"Top 10 categorías de {col}"
        else:
            title = f"Distribución de {col}"

        value_counts.plot(kind="bar", ax=ax)
        ax.set_title(title)
        ax.set_xlabel("")
        ax.set_ylabel("Frecuencia")
        ax.tick_params(axis="x", rotation=45)

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
            crosstab = pd.crosstab(df[col], df[target_col], normalize="index") * 100
            print(crosstab.round(1))


def correlation_analysis(
    df: pd.DataFrame, target_col: Optional[str] = None, threshold: float = 0.7
) -> None:
    """
    Analiza correlaciones entre variables numéricas.

    Args:
        df (pd.DataFrame): Dataset a analizar
        target_col (str, optional): Columna objetivo
        threshold (float): Umbral para detectar correlaciones altas
    """
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

    if len(numeric_cols) < 2:
        print(
            "❌ Se necesitan al menos 2 columnas numéricas para análisis de correlación"
        )
        return

    print("🔗 ANÁLISIS DE CORRELACIONES")
    print("=" * 60)

    # Calcular matriz de correlación
    correlation_matrix = df[numeric_cols].corr()

    # Detectar correlaciones altas
    high_corr_pairs = []
    for i in range(len(correlation_matrix.columns)):
        for j in range(i + 1, len(correlation_matrix.columns)):
            corr_val = correlation_matrix.iloc[i, j]
            if pd.notna(corr_val) and abs(float(corr_val)) >= threshold:
                high_corr_pairs.append(
                    (
                        correlation_matrix.columns[i],
                        correlation_matrix.columns[j],
                        correlation_matrix.iloc[i, j],
                    )
                )

    if high_corr_pairs:
        print(f"⚠️ Correlaciones altas (|r| >= {threshold}):")
        for var1, var2, corr in high_corr_pairs:
            print(f"   - {var1} ↔ {var2}: {corr:.3f}")
    else:
        print(f"✅ No se encontraron correlaciones altas (|r| >= {threshold})")

    # Correlaciones con variable objetivo
    if target_col and target_col in numeric_cols:
        target_corrs = (
            correlation_matrix[target_col]
            .drop(target_col)
            .abs()
            .sort_values(ascending=False)
        )
        print(f"\n🎯 Correlaciones con {target_col} (ordenadas por magnitud):")
        for var, corr in target_corrs.items():
            print(f"   - {var}: {corr:.3f}")

    # Visualización del mapa de calor
    plt.figure(figsize=(10, 8))
    mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))
    sns.heatmap(
        correlation_matrix,
        mask=mask,
        annot=True,
        cmap="coolwarm",
        center=0,
        square=True,
        fmt=".2f",
        cbar_kws={"shrink": 0.8},
    )
    plt.title("Matriz de Correlaciones")
    plt.tight_layout()
    plt.show()

    # Mapa de calor de la correlación entre la variable objetivo y las otras características
    if target_col and target_col in numeric_cols:
        target_correlation = correlation_matrix[target_col].sort_values(ascending=False)

        plt.figure(figsize=(8, 10))
        sns.heatmap(
            target_correlation.to_frame(), annot=True, cmap="coolwarm", fmt=".2f"
        )
        plt.title(f"Correlation with {target_col}")
        plt.show()
    else:
        print(
            f"⚠️ No se puede mostrar correlaciones con variable objetivo: '{target_col}' no está en las columnas numéricas o no se especificó."
        )


# Función para analizar patrones de churn por variable
def analyze_churn_patterns(data, variable, title):
    """Analiza los patrones de churn para una variable específica"""

    # Crear tabla de contingencia
    crosstab = pd.crosstab(data[variable], data["Churn"], normalize="index") * 100
    crosstab_counts = pd.crosstab(data[variable], data["Churn"])

    # Crear figura con subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    # Gráfico de barras con porcentajes
    crosstab.plot(kind="bar", ax=ax1, color=["lightblue", "salmon"])
    ax1.set_title(f"{title} - Tasa de Churn (%)")
    ax1.set_xlabel(variable)
    ax1.set_ylabel("Porcentaje")
    ax1.legend(["No Churn", "Churn"])
    ax1.set_xticklabels(ax1.get_xticklabels(), rotation=45, ha="right")

    # Agregar etiquetas de porcentaje
    for container in ax1.containers:
        ax1.bar_label(container, fmt="%.1f%%")

    # Gráfico de barras apiladas con conteos
    crosstab_counts.plot(
        kind="bar", stacked=True, ax=ax2, color=["lightblue", "salmon"]
    )
    ax2.set_title(f"{title} - Distribución Total")
    ax2.set_xlabel(variable)
    ax2.set_ylabel("Cantidad de Clientes")
    ax2.legend(["No Churn", "Churn"])
    ax2.set_xticklabels(ax2.get_xticklabels(), rotation=45, ha="right")

    plt.tight_layout()
    plt.show()

    # Mostrar estadísticas detalladas
    print(f"\n📊 ANÁLISIS DETALLADO: {title.upper()}")
    print("=" * 60)

    # Tabla de contingencia con conteos
    print(f"\n📋 Conteos por {variable}:")
    print(crosstab_counts)

    # Tabla de contingencia con porcentajes
    print(f"\n📈 Tasa de Churn por {variable} (%):")
    churn_rates = crosstab["Yes"].sort_values(ascending=False)
    for category in churn_rates.index:
        total = crosstab_counts.loc[category].sum()
        churn_count = crosstab_counts.loc[category, "Yes"]
        churn_rate = churn_rates[category]
        print(f"   {category}: {churn_rate:.1f}% ({churn_count}/{total} clientes)")

    # Identificar patrones clave
    highest_churn = churn_rates.index[0]
    lowest_churn = churn_rates.index[-1]

    print(f"\n🔍 PATRONES IDENTIFICADOS:")
    print(
        f"   ⚠️  MAYOR RIESGO: {highest_churn} ({churn_rates[highest_churn]:.1f}% churn)"
    )
    print(
        f"   ✅ MENOR RIESGO: {lowest_churn} ({churn_rates[lowest_churn]:.1f}% churn)"
    )
    print(
        f"   📊 DIFERENCIA: {churn_rates[highest_churn] - churn_rates[lowest_churn]:.1f} puntos porcentuales"
    )

    return churn_rates


def show_correlation_respect_to_feature(df: pd.DataFrame, feature = 'Churn_Yes'):
    # Copia del dataframe preprocesado
    df_corr = df.copy()
    
    # Verifica que 'Churn_Yes' existe
    if 'Churn_Yes' not in df_corr.columns:
        raise ValueError("⚠️ La columna 'Churn_Yes' no está presente en X_train_clean.")
    
    # Seleccionar columnas numéricas + booleanas
    numeric_df = df_corr.select_dtypes(include=['int64', 'float64', 'bool']).astype(float)
    
    # Calcular matriz de correlación
    correlation_matrix = numeric_df.corr()
    
    # Extraer correlación con respecto a Churn_Yes (sin autocorrelación)
    churn_corr = correlation_matrix['Churn_Yes'].drop('Churn_Yes').sort_values()
    
    # Mostrar heatmap vertical
    plt.figure(figsize=(6, len(churn_corr) * 0.5))  # Altura dinámica según cantidad de filas
    sns.heatmap(churn_corr.to_frame(), annot=True, cmap='coolwarm', center=0, cbar=True)
    plt.title('📊 Correlación con Churn_Yes', fontsize=14)
    plt.xlabel('Correlación')
    plt.ylabel('Variable')
    plt.tight_layout()
    plt.show()



def generate_eda_report(df: pd.DataFrame, target_col: str = "Churn") -> None:
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
    print("\n" + "=" * 80 + "\n")

    # Análisis de variable objetivo
    analyze_target(df, target_col)
    print("\n" + "=" * 80 + "\n")

    # Análisis de características numéricas
    analyze_numerical_features(df, target_col)
    print("\n" + "=" * 80 + "\n")

    # Análisis de características categóricas
    analyze_categorical_features(df, target_col)
    print("\n" + "=" * 80 + "\n")

    # Análisis de correlaciones
    correlation_analysis(df, target_col)
    print("\n" + "=" * 80 + "\n")

    print("✅ Reporte de EDA completado")


if __name__ == "__main__":
    # Ejemplo de uso
    print("🔄 Probando módulo de EDA...")

    # Crear datos de ejemplo (normalmente vendrían de data_loader)
    np.random.seed(42)
    n_samples = 1000

    sample_df = pd.DataFrame(
        {
            "customerID": [f"C{i:04d}" for i in range(n_samples)],
            "tenure": np.random.randint(0, 73, n_samples),
            "MonthlyCharges": np.random.uniform(18, 120, n_samples),
            "TotalCharges": np.random.uniform(18, 8000, n_samples),
            "Contract": np.random.choice(
                ["Month-to-month", "One year", "Two year"], n_samples
            ),
            "InternetService": np.random.choice(
                ["DSL", "Fiber optic", "No"], n_samples
            ),
            "Churn": np.random.choice([0, 1], n_samples, p=[0.7, 0.3]),
        }
    )

    # Ejecutar EDA
    generate_eda_report(sample_df, "Churn")
