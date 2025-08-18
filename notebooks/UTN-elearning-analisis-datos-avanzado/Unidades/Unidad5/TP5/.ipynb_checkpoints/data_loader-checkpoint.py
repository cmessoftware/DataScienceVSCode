"""
Módulo para la carga y gestión de datos del proyecto de predicción de churn.
"""

import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")


def load_competition_data(data_dir="."):
    """
    Carga los archivos de datos de la competencia de Kaggle.

    Args:
        data_dir (str): Directorio donde se encuentran los archivos CSV

    Returns:
        tuple: (train_df, test_df, sample_submission_df)
    """
    data_path = Path(data_dir)

    try:
        train_df = pd.read_csv(data_path / "train.csv")
        test_df = pd.read_csv(data_path / "test.csv")
        sample_submission_df = pd.read_csv(data_path / "sample_submission.csv")

        print("✅ Datos cargados exitosamente:")
        print(f"   - Train: {train_df.shape}")
        print(f"   - Test: {test_df.shape}")
        print(f"   - Sample submission: {sample_submission_df.shape}")

        return train_df, test_df, sample_submission_df

    except FileNotFoundError as e:
        print(f"❌ Error: No se encontraron los archivos de datos: {e}")
        print(
            "💡 Asegúrate de que los archivos train.csv, test.csv y sample_submission.csv estén en el directorio especificado"
        )
        return None, None, None


def create_sample_data(n_train=1000, n_test=200, random_state=42):
    """
    Crea datos de ejemplo para desarrollo y pruebas.

    Args:
        n_train (int): Número de muestras para entrenamiento
        n_test (int): Número de muestras para prueba
        random_state (int): Semilla para reproducibilidad

    Returns:
        tuple: (train_df, test_df, sample_submission_df)
    """
    np.random.seed(random_state)

    # Simular datos de entrenamiento
    train_df = pd.DataFrame(
        {
            "customerID": [f"C{i:04d}" for i in range(n_train)],
            "gender": np.random.choice(["Male", "Female"], n_train),
            "SeniorCitizen": np.random.choice([0, 1], n_train, p=[0.84, 0.16]),
            "Partner": np.random.choice(["Yes", "No"], n_train),
            "Dependents": np.random.choice(["Yes", "No"], n_train, p=[0.3, 0.7]),
            "tenure": np.random.randint(0, 73, n_train),
            "PhoneService": np.random.choice(["Yes", "No"], n_train, p=[0.9, 0.1]),
            "MultipleLines": np.random.choice(
                ["Yes", "No", "No phone service"], n_train
            ),
            "InternetService": np.random.choice(
                ["DSL", "Fiber optic", "No"], n_train, p=[0.35, 0.45, 0.2]
            ),
            "OnlineSecurity": np.random.choice(
                ["Yes", "No", "No internet service"], n_train
            ),
            "OnlineBackup": np.random.choice(
                ["Yes", "No", "No internet service"], n_train
            ),
            "DeviceProtection": np.random.choice(
                ["Yes", "No", "No internet service"], n_train
            ),
            "TechSupport": np.random.choice(
                ["Yes", "No", "No internet service"], n_train
            ),
            "StreamingTV": np.random.choice(
                ["Yes", "No", "No internet service"], n_train
            ),
            "StreamingMovies": np.random.choice(
                ["Yes", "No", "No internet service"], n_train
            ),
            "Contract": np.random.choice(
                ["Month-to-month", "One year", "Two year"], n_train, p=[0.55, 0.25, 0.2]
            ),
            "PaperlessBilling": np.random.choice(["Yes", "No"], n_train),
            "PaymentMethod": np.random.choice(
                [
                    "Electronic check",
                    "Mailed check",
                    "Bank transfer (automatic)",
                    "Credit card (automatic)",
                ],
                n_train,
            ),
            "MonthlyCharges": np.random.uniform(18.25, 118.75, n_train),
            "TotalCharges": np.random.uniform(18.8, 8684.8, n_train),
        }
    )

    # Crear variable objetivo con cierta lógica
    churn_prob = np.random.random(n_train)
    # Los clientes con contratos más largos tienen menor probabilidad de churn
    contract_effect = np.where(
        train_df["Contract"] == "Month-to-month",
        0.3,
        np.where(train_df["Contract"] == "One year", 0.15, 0.05),
    )
    # Los clientes con tenure más alto tienen menor probabilidad de churn
    tenure_effect = 0.3 * (1 - train_df["tenure"] / 72)

    final_prob = np.clip(churn_prob + contract_effect + tenure_effect, 0, 1)
    train_df["Churn"] = np.where(final_prob > 0.6, 1, 0)

    # Simular datos de prueba (sin target)
    test_df = (
        train_df.drop("Churn", axis=1)
        .sample(n_test, random_state=random_state)
        .reset_index(drop=True)
    )
    test_df["customerID"] = [f"T{i:04d}" for i in range(len(test_df))]

    # Simular archivo de submission
    sample_submission_df = pd.DataFrame(
        {"customerID": test_df["customerID"], "Churn": 0.5}  # Probabilidad base
    )

    print("📝 Datos de ejemplo creados:")
    print(f"   - Train: {train_df.shape}")
    print(f"   - Test: {test_df.shape}")
    print(f"   - Sample submission: {sample_submission_df.shape}")
    print(f"   - Distribución de Churn: {train_df['Churn'].value_counts().to_dict()}")

    return train_df, test_df, sample_submission_df


def get_data_info(df, name="Dataset"):
    """
    Muestra información básica sobre un DataFrame.

    Args:
        df (pd.DataFrame): DataFrame a analizar
        name (str): Nombre descriptivo del dataset
    """
    print(f"\n📊 INFORMACIÓN DE {name.upper()}")
    print("=" * 50)
    print(f"Dimensiones: {df.shape}")
    print(f"Memoria utilizada: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")

    print("\n🔍 Tipos de datos:")
    print(df.dtypes.value_counts())

    print("\n⚠️ Valores faltantes:")
    missing = df.isnull().sum()
    if missing.sum() > 0:
        print(missing[missing > 0])
    else:
        print("✅ No hay valores faltantes")

    print("\n📈 Estadísticas numéricas:")
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) > 0:
        print(df[numeric_cols].describe())
    else:
        print("No hay columnas numéricas")


if __name__ == "__main__":
    # Ejemplo de uso
    print("🔄 Probando carga de datos...")

    # Intentar cargar datos reales
    train, test, submission = load_competition_data()

    if train is None:
        print("\n🔄 Creando datos de ejemplo...")
        train, test, submission = create_sample_data()

    # Mostrar información
    get_data_info(train, "Train")
    get_data_info(test, "Test")
