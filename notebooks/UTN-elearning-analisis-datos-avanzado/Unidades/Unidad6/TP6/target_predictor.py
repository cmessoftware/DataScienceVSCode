#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Models Module for Target Prediction - Stable Version

Version without emojis optimized for maximum compatibility
Includes TargetPredictor and hyperparameter optimization functions
"""

import warnings

warnings.filterwarnings("ignore")

import time
from datetime import datetime
import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV, StratifiedKFold

# Import display for Jupyter environments
try:
    from IPython.display import display
except ImportError:
    # Fallback for non-Jupyter environments
    def display(obj, **kwargs):
        print(obj)
        return None


from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, StratifiedKFold
from tqdm.auto import tqdm
from datetime import datetime
from joblib import parallel_backend
from scipy.stats import rv_continuous


# Import tqdm for progress bar
# try:
#     from tqdm import tqdm

#     TQDM_AVAILABLE = True
# except ImportError:
#     TQDM_AVAILABLE = False
#     print("Warning: tqdm not available - using basic progress")


class FeatureEngineer(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        # Safe casting
        if "TotalCharges" in X.columns:
            X["TotalCharges"] = pd.to_numeric(X["TotalCharges"], errors="coerce")
        if "tenure" in X.columns:
            X["avg_monthly"] = X["TotalCharges"] / (X["tenure"] + 1)
            X["tenure_group"] = pd.cut(
                X["tenure"],
                bins=[0, 6, 24, 60, 100],
                labels=["0-6", "7-24", "25-60", "60+"],
                include_lowest=True,
            )
            X["new_customer"] = (X["tenure"] < 6).astype(int)
        if "MonthlyCharges" in X.columns:
            X["high_monthly_charge"] = (X["MonthlyCharges"] > 80).astype(int)
        
        return X
    
    def _categorize_features(self, X: pd.DataFrame) -> tuple:
        """
        Categorize features into numeric, binary, and categorical types.
        
        Args:
            X (pd.DataFrame): Input dataframe
            
        Returns:
            tuple: (numeric_features, binary_features, categorical_features)
        """
        id_cols = [col for col in X.columns if "id" in col.lower()]
        numeric_features = (
            X.select_dtypes(include=["int64", "float64"])
            .columns.difference(id_cols)
            .tolist()
        )
        object_features = (
            X.select_dtypes(include="object").columns.difference(id_cols).tolist()
        )

        binary_features = [
            col
            for col in object_features
            if set(X[col].dropna().unique()) <= {"Yes", "No"}
        ]
        categorical_features = [
            col for col in object_features if col not in binary_features
        ]
        
        return numeric_features, binary_features, categorical_features



class TargetPredictor:
    """
    Target Predictor - Stable Version

    Main class for customer churn prediction modeling
    Includes preprocessing, training and model evaluation
    """

    def __init__(self, random_state=42):
        """
        Initialize the Target predictor

        Args:
            random_state (int): Seed for reproducibility
        """
        self.random_state = random_state
        self.preprocessor = None
        self.models = {}
        self.results = {}

        print(f"🗺️  TargetPredictor initialized with random_state={random_state}")
        
    def ensure_dense(self):
        """Transformer para convertir sparse->dense antes de samplers que lo requieren."""
        from sklearn.preprocessing import FunctionTransformer
        return FunctionTransformer(
            lambda X: X.toarray() if hasattr(X, "toarray") else X, validate=False
        )

    def _step_name(self, step):
        # Returns readable step name (class or function)
        if isinstance(step, tuple) and len(step) == 2:
            name, obj = step
        else:
            name, obj = None, step
        try:
            cls = obj.__class__.__name__
        except Exception:
            cls = str(type(obj))
        # For FunctionTransformer, show the function
        if isinstance(obj, FunctionTransformer) and obj.func is not None:
            return f"{cls}({getattr(obj.func, '__name__', str(obj.func))})"
        return cls

    def _describe_transformer(self, trf):
        """
        Returns (transformer_type, steps_str) to display in table.
        - If Pipeline: list of steps.
        - If ColumnTransformer: name and number of sub-blocks.
        - If other estimator: its class.
        """
        if isinstance(trf, Pipeline):
            steps = [self._step_name(s) for s in trf.steps]
            return ("Pipeline", " -> ".join(steps))
        if isinstance(trf, ColumnTransformer):
            # rare here (nested), but we consider it
            return ("ColumnTransformer", f"{len(trf.transformers)} sub-blocks")
        if trf in ("drop", "passthrough"):
            return (str(trf), "")
        try:
            return (trf.__class__.__name__, "")
        except Exception:
            return (str(type(trf)), "")

    def _column_transformers_to_df(self, models_dict):
        """
        models_dict: dict[str, ColumnTransformer]
          Ex: {'Logistic_Regression': <ColumnTransformer ...>, ...}
        """
        rows = []
        for model_name, coltr in models_dict.items():
            if not isinstance(coltr, ColumnTransformer):
                # If a complete Pipeline came, try to extract the 'preprocessor' step
                if isinstance(coltr, Pipeline) and "preprocessor" in dict(coltr.steps):
                    coltr = dict(coltr.steps)["preprocessor"]
                else:
                    # Not a ColumnTransformer; record and continue
                    rows.append(
                        {
                            "model": model_name,
                            "block": "",
                            "transformer_type": coltr.__class__.__name__,
                            "steps": "",
                            "columns": "",
                        }
                    )
                    continue

            # coltr.transformers is list of tuples (name, transformer, columns)
            for name, trf, cols in coltr.transformers:
                trf_type, steps_desc = self._describe_transformer(trf)

                # columns can come as list, slice, 'drop', 'passthrough', etc.
                if isinstance(cols, (list, tuple)):
                    cols_str = ", ".join(map(str, cols))
                else:
                    cols_str = str(cols)

                rows.append(
                    {
                        "model": model_name,
                        "block": name,
                        "transformer_type": trf_type,
                        "steps": steps_desc,
                        "columns": cols_str,
                    }
                )

        df = pd.DataFrame(
            rows, columns=["model", "block", "transformer_type", "steps", "columns"]
        )
        return df

    def _normalize_service_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Replaces values like 'No internet service' with 'No' in optional service columns.
        """
        df = df.copy()

        # Columns that may have 'No internet service'
        internet_service_cols = [
            "OnlineSecurity",
            "OnlineBackup",
            "DeviceProtection",
            "TechSupport",
            "StreamingTV",
            "StreamingMovies",
        ]

        # Columns that may have 'No phone service'
        phone_service_cols = ["MultipleLines"]

        # Normalize internet services
        for col in internet_service_cols:
            if col in df.columns:
                df[col] = df[col].replace("No internet service", "No")

        # Normalize phone services
        for col in phone_service_cols:
            if col in df.columns:
                df[col] = df[col].replace("No phone service", "No")

        return df

    def _transform_yes_no(self, X):
        """
        Transforms columns with 'Yes'/'No' values to 1/0.

        Args:
            X: Pandas Series or numpy array with 'Yes'/'No' values.

        Returns:
            Array transformed to 1/0.
        """
        if hasattr(X, "map"):
            # It's pandas Series
            return X.map({"Yes": 1, "No": 0}).fillna(0).astype(int)
        else:
            # It's numpy array
            import numpy as np

            result = np.where(X == "Yes", 1, np.where(X == "No", 0, 0))
            return result.astype(int)

    def _transform_yes_no_df(self, X):
        """
        Transforms DataFrame with 'Yes'/'No' columns to 1/0.

        Args:
            X: DataFrame with 'Yes'/'No' columns or 2D numpy array.

        Returns:
            DataFrame or array transformed to 1/0.
        """
        if hasattr(X, "apply"):
            # It's pandas DataFrame
            return X.apply(self._transform_yes_no)
        else:
            # It's 2D numpy array
            import numpy as np

            if len(X.shape) == 2:
                # 2D Array - apply transformation to each column
                result = np.zeros_like(X, dtype=int)
                for i in range(X.shape[1]):
                    result[:, i] = self._transform_yes_no(X[:, i])
                return result
            else:
                # 1D Array
                return self._transform_yes_no(X)

    def _transform_male_female(self, X):
        """
        Transforms columns with 'Male'/'Female' values to 1/0.
        Args:
            X: Pandas Series or numpy array with 'Male'/'Female' values.

        Returns:
            Array transformed to 1/0.
        """
        if hasattr(X, "map"):
            # It's pandas Series
            return X.map({"Male": 1, "Female": 0}).fillna(0).astype(int)
        else:
            # It's numpy array
            import numpy as np

            result = np.where(X == "Male", 1, np.where(X == "Female", 0, 0))
            return result.astype(int)

    def inspect_transformed_columns(
        self,
        X_original: pd.DataFrame,
        columns: list,
        fit=True,
        current_preprocessor=None,
    ):
        """
        Shows comparison between original columns and their numeric transformations.

        Args:
            X_original (pd.DataFrame): Original dataset without transformations.
            columns (list): List of columns to inspect.
            fit (bool): If True, does fit_transform; if False, only transform.
        """
        print(f"🔍 Inspecting column transformation: {columns}")

        if current_preprocessor is not None:
            self.preprocessor = current_preprocessor

        print(f"Using preprocessor: {self.preprocessor}")

        # Make sure to use only selected columns
        X_subset = X_original.copy()

        # Fit or only transform (depending on context)
        if fit:
            X_transformed = self.preprocessor.fit_transform(X_subset)
        else:
            X_transformed = self.preprocessor.transform(X_subset)

        # Get transformed column names with dimension verification
        try:
            print(f"🔍 Debug: X_transformed shape: {X_transformed.shape}")

            # Get columns of each type safely
            num_cols = []
            bin_cols = []
            cat_cols = []

            # Iterate over available transformers
            for name, transformer, cols in self.preprocessor.transformers_:
                if name == "num":
                    num_cols = list(cols)
                elif name == "bin":
                    bin_cols = list(cols)
                elif name == "cat":
                    cat_cols = list(cols)

            print(
                f"🔍 Debug: num_cols={len(num_cols)}, bin_cols={len(bin_cols)}, cat_cols={len(cat_cols)}"
            )

            # For categorical features, get names after one-hot encoding
            cat_names = []
            if cat_cols and "cat" in self.preprocessor.named_transformers_:
                cat_transformer = self.preprocessor.named_transformers_["cat"]
                if (
                    hasattr(cat_transformer, "named_steps")
                    and "encoder" in cat_transformer.named_steps
                ):
                    cat_encoder = cat_transformer.named_steps["encoder"]
                    if hasattr(cat_encoder, "get_feature_names_out"):
                        cat_names = list(cat_encoder.get_feature_names_out(cat_cols))
                        print(f"🔍 Debug: cat_names length: {len(cat_names)}")
                elif hasattr(cat_transformer, "get_feature_names_out"):
                    cat_names = list(cat_transformer.get_feature_names_out(cat_cols))
                    print(f"🔍 Debug: cat_names length: {len(cat_names)}")

            final_cols = num_cols + bin_cols + cat_names
            print(f"🔍 Debug: final_cols length: {len(final_cols)}")

            # Verify dimensions match
            if len(final_cols) != X_transformed.shape[1]:
                print(
                    f"⚠️ Mismatch: final_cols={len(final_cols)}, X_transformed.shape[1]={X_transformed.shape[1]}"
                )
                raise ValueError("Dimensions don't match")

        except Exception as e:
            print(f"⚠️ Could not get column names: {e}")
            # Fallback with generic names based on actual shape
            final_cols = [f"feature_{i}" for i in range(X_transformed.shape[1])]
            print(f"🔧 Using fallback with {len(final_cols)} generic columns")

        # Convert to DataFrame
        df_transformed = pd.DataFrame(X_transformed, columns=final_cols)

        # Search for all transformed columns that derive from selected original columns
        matched_cols = []
        for col in columns:
            # Exact matches or one-hot
            matched = [
                c for c in df_transformed.columns if col == c or c.startswith(col + "_")
            ]
            matched_cols.extend(matched)

        # Show side-by-side comparison
        display("\n🗂️  Original values:")
        print(X_subset[columns].head())

        print("⚙️  Transformed values:")
        if matched_cols:
            print(df_transformed[matched_cols].head())
        else:
            print(
                "⚠️ No matching columns found in transformed data"
            )
            print(
                f"Available columns: {list(df_transformed.columns[:10])}..."
            )  # Show first 10
            
    
    def create_preprocessor_random_forest(self, X: pd.DataFrame) -> ColumnTransformer:
        """
        Justification:
            Random Forest doesn't need scaling.
            Supports numeric and dummified categorical variables well.
            Recommended to remove irrelevant variables but maintain complete encoding.
        """
        X = self._normalize_service_values(X)

        fe = FeatureEngineer()
        numeric_features, binary_features, categorical_features = fe._categorize_features(X)

        yes_no_transformer = FunctionTransformer(self._transform_yes_no, validate=False)

        return ColumnTransformer(
            [
                ("num", SimpleImputer(strategy="median"), numeric_features),
                (
                    "bin",
                    Pipeline(
                        [
                            ("imputer", SimpleImputer(strategy="most_frequent")),
                            ("transformer", yes_no_transformer),
                        ]
                    ),
                    binary_features,
                ),
                (
                    "cat",
                    Pipeline(
                        [
                            ("imputer", SimpleImputer(strategy="most_frequent")),
                            (
                                "encoder",
                                OneHotEncoder(drop="first", handle_unknown="ignore"),
                            ),
                        ]
                    ),
                    categorical_features,
                ),
            ]
        )

    def create_preprocessor_svm(self, X: pd.DataFrame) -> ColumnTransformer:
        """
        Numeric: imputation (median) + StandardScaler.
        Categorical: imputation (most frequent) + One-Hot.
        For SVC(kernel='rbf'), use dense output (sparse=False) to avoid densifying later.
        If you prefer LinearSVC, you can leave sparse=True (lighter in memory).
        Sí, aplica y está bien para un SVC con kernel RBF. Ese preprocesador:
        Imputa y estandariza numéricas → imprescindible para SVC.
        One-Hot en categóricas con sparse=False → entrega matriz densa al clasificador.
        Pros y contras de tu versión (densa)
        Pros
        Setup simple y funciona out-of-the-box con SVC (RBF).
        Evita sorpresas de mezclas sparse/dense.
        Contras
        Puede consumir mucha RAM si el OHE genera muchas columnas (SVC ya es  𝑂(𝑛**2) en muestras).
        Si además usás SMOTE/NearMiss, el denso se vuelve aún más pesado.
        """
        if X is None or X.empty:
            raise ValueError("X cannot be None or empty")

        X = self._normalize_service_values(X)

        num_cols: List[str] = X.select_dtypes(include=[np.number]).columns.tolist()
        cat_cols: List[str] = X.select_dtypes(exclude=[np.number]).columns.tolist()

        numeric = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
            ]
        )

        categorical = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
            ]
        )

        preprocessor = ColumnTransformer(
            transformers=[
                ("num", numeric, num_cols),
                ("cat", categorical, cat_cols),
            ],
            remainder="drop",
            verbose_feature_names_out=False,
        )

        return preprocessor

    def create_preprocessor_svm_sparse(self, X: pd.DataFrame) -> ColumnTransformer:
        """
        Versión parcialmente esparsa y ligera
        Mantiene OHE sparse y “balancea” magnitudes con MaxAbsScaler (no rompe la esparsidad), útil cuando hay muchas dummies.
        Nota: aunque el bloque numérico sale “sparse-like” por with_mean=False, algunos pasos (imputer) pueden producir denso en ese subbloque; 
        aun así, con muchas categóricas OHE el ColumnTransformer suele mantener salida global sparse si pasa sparse_threshold.
        """
        from sklearn.compose import ColumnTransformer
        from sklearn.pipeline import Pipeline
        from sklearn.impute import SimpleImputer
        from sklearn.preprocessing import OneHotEncoder, StandardScaler, MaxAbsScaler
    
        if X is None or X.empty:
            raise ValueError("X cannot be None or empty")
    
        num_cols = X.select_dtypes(include=[np.number]).columns.tolist()
        cat_cols = X.select_dtypes(exclude=[np.number]).columns.tolist()
    
        numeric = Pipeline(steps=[
            ("imputer", SimpleImputer(strategy="median")),
            # with_mean=False para no densificar por centrado (SVC no necesita el centrado estricto)
            ("scaler",  StandardScaler(with_mean=False)),
        ])
    
        categorical = Pipeline(steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot",  OneHotEncoder(handle_unknown="ignore", sparse_output=True)),
        ])
    
        pre = ColumnTransformer(
            transformers=[
                ("num", numeric, num_cols),
                ("cat", categorical, cat_cols),
            ],
            verbose_feature_names_out=False,
            sparse_threshold=0.3,   # deja salida sparse si la fracción es suficientemente dispersa
        )
    
        # Post-escala sin romper esparsidad (igualar rangos num/cat)
        pre_svc_sparse = Pipeline([
            ("pre", pre),
            ("maxabs", MaxAbsScaler())
        ])
        return pre_svc_sparse


    def create_preprocessor_xgboost(self, X: pd.DataFrame) -> ColumnTransformer:
        raise NotImplementedError("Not yet implemented")

    def create_preprocessor_naive_bayes(self, X: pd.DataFrame) -> ColumnTransformer:
        """
        Justification:
            Classic Naive Bayes (GaussianNB) doesn't handle sparse dummified variables well.
            Recommended: LabelEncoder for multiclass categorical, and keep numeric as is.
            Practical alternative: convert multiclass to ordinal.
        """
        from sklearn.preprocessing import OrdinalEncoder

        X = self._normalize_service_values(X)

        fe = FeatureEngineer()
        numeric_features, binary_features, categorical_features = fe._categorize_features(X)

        yes_no_transformer = FunctionTransformer(self._transform_yes_no, validate=False)

        return ColumnTransformer(
            [
                (
                    "num",
                    Pipeline([("imputer", SimpleImputer(strategy="mean"))]),
                    numeric_features,
                ),
                (
                    "bin",
                    Pipeline(
                        [
                            ("imputer", SimpleImputer(strategy="most_frequent")),
                            ("transformer", yes_no_transformer),
                        ]
                    ),
                    binary_features,
                ),
                (
                    "cat",
                    Pipeline(
                        [
                            ("imputer", SimpleImputer(strategy="most_frequent")),
                            (
                                "encoder",
                                OrdinalEncoder(
                                    handle_unknown="use_encoded_value", unknown_value=-1
                                ),
                            ),
                        ]
                    ),
                    categorical_features,
                ),
            ]
        )

    def create_preprocessor_knn(self, X: pd.DataFrame) -> ColumnTransformer:
        """
        Justification:
            KNN is very sensitive to scale, so everything must be scaled.
            One option is to use OneHotEncoder(sparse_output=False) + StandardScaler for everything.
        """
        X = self._normalize_service_values(X)

        fe = FeatureEngineer()
        numeric_features, binary_features, categorical_features = fe._categorize_features(X)

        yes_no_transformer = FunctionTransformer(self._transform_yes_no, validate=False)

        return ColumnTransformer(
            [
                (
                    "num",
                    Pipeline(
                        [
                            ("imputer", SimpleImputer(strategy="median")),
                            ("scaler", StandardScaler()),
                        ]
                    ),
                    numeric_features,
                ),
                (
                    "bin",
                    Pipeline(
                        [
                            ("imputer", SimpleImputer(strategy="most_frequent")),
                            ("transformer", yes_no_transformer),
                            ("scaler", StandardScaler()),
                        ]
                    ),
                    binary_features,
                ),
                (
                    "cat",
                    Pipeline(
                        [
                            ("imputer", SimpleImputer(strategy="most_frequent")),
                            (
                                "encoder",
                                OneHotEncoder(drop="first", sparse_output=False),
                            ),
                            ("scaler", StandardScaler()),
                        ]
                    ),
                    categorical_features,
                ),
            ]
        )

    def create_preprocessor_logistic_regression(
        self, X: pd.DataFrame
    ) -> ColumnTransformer:
        """
        Justification:
            Works better with scaled variables.
            Supports dummified variables well.
            Avoid many low-information columns.
        """
        X = self._normalize_service_values(X)

        fe = FeatureEngineer()
        numeric_features, binary_features, categorical_features = fe._categorize_features(X)

        yes_no_transformer = FunctionTransformer(self._transform_yes_no, validate=False)

        return ColumnTransformer(
            [
                (
                    "num",
                    Pipeline(
                        [
                            ("imputer", SimpleImputer(strategy="median")),
                            ("scaler", StandardScaler()),
                        ]
                    ),
                    numeric_features,
                ),
                (
                    "bin",
                    Pipeline(
                        [
                            ("imputer", SimpleImputer(strategy="most_frequent")),
                            ("transformer", yes_no_transformer),
                        ]
                    ),
                    binary_features,
                ),
                (
                    "cat",
                    Pipeline(
                        [
                            ("imputer", SimpleImputer(strategy="most_frequent")),
                            (
                                "encoder",
                                OneHotEncoder(drop="first", handle_unknown="ignore"),
                            ),
                        ]
                    ),
                    categorical_features,
                ),
            ]
        )

    def create_preprocesor_gradient_boosting(
        self, X: pd.DataFrame
    ) -> ColumnTransformer:
        """
        Justification:
              Doesn't require scaling (trees are insensitive to scale)
              Sensitive to dummies but can handle high-cardinality
              Requires imputation in null handling, but some modern GB do it internally
              GB doesn't need one-hot; can use label encoding or ordinal encoding (or directly if CatBoost or LGBM with native support)

        Parameters for GradientBoostingClassifier:
            | Parameter            | Meaning                                                        | Common justification                                                                                      |
            |----------------------|--------------------------------------------------------------------|-----------------------------------------------------------------------------------------------------------|
            | `n_estimators=150`   | Number of trees in the ensemble                                   | More trees improve performance up to a point. 150 is a reasonable value.                        |
            | `learning_rate=0.1`  | How much each new tree contributes to the final model                 | Classic value (default). Reduces overfitting risk if `n_estimators` is increased.                      |
            | `max_depth=6`        | Maximum depth of each tree                                   | Deeper trees capture more patterns, but with overfitting risk.                             |
            | `random_state=42`    | Seed for reproducibility                                      | Fixes results between runs.                                                                   |
            | `subsample=1.0`      | Proportion of samples to train each tree                    | If < 1.0 introduces randomness that can reduce overfitting. Ex: `0.8` is used in stochastic boosting.  |
            | `min_samples_split=2`| Minimum number of samples to split a node                     | Increasing this value makes the model more conservative.                                                      |
            | `min_samples_leaf=1` | Minimum number of samples in a terminal leaf                     | Useful for smoothing the model and avoiding small leaves that overfit.                                   |
            | `max_features=None`  | Maximum number of features evaluated when splitting a node             | Limiting it (`'sqrt'`, `'log2'`, number or fraction) can reduce overfitting.                          |
            | `loss='log_loss'`    | Loss function to optimize                                     | `'log_loss'` (default) for binary classification, also available: `'exponential'`.                  |
            | `criterion='friedman_mse'` | Function to measure the quality of a split                | `'friedman_mse'` is robust and suitable for boosting.                                                    |
            | `init=None`          | Initial model before applying boosting                           | Can use a base model, but `None` uses initial constant prediction.                              |
            | `warm_start=False`   | Continue training from a previous execution                 | Useful for fitting more trees without retraining from scratch.                                          |


        """
        X = self._normalize_service_values(X)

        fe = FeatureEngineer()
        numeric_features, binary_features, categorical_features = fe._categorize_features(X)

        yes_no_transformer = FunctionTransformer(self._transform_yes_no, validate=False)

        return ColumnTransformer(
            [
                (
                    "num",
                    Pipeline(
                        [
                            ("imputer", SimpleImputer(strategy="median")),
                            # No scaler
                        ]
                    ),
                    numeric_features,
                ),
                (
                    "bin",
                    Pipeline(
                        [
                            ("imputer", SimpleImputer(strategy="most_frequent")),
                            ("transformer", yes_no_transformer),
                        ]
                    ),
                    binary_features,
                ),
                (
                    "cat",
                    Pipeline(
                        [
                            ("imputer", SimpleImputer(strategy="most_frequent")),
                            (
                                "encoder",
                                OneHotEncoder(drop="first", handle_unknown="ignore"),
                            ),
                        ]
                    ),
                    categorical_features,
                ),
            ]
        )

    def create_preprocessor(self, X_train):
        """
        Create the preprocessor for features - IMPROVED VERSION WITH BALANCING

        Args:
            X_train: Training dataset

        Returns:
            ColumnTransformer: Configured preprocessor
        """
        print("🗺️  Creating improved preprocessor...")

        # Remove problematic columns
        X_clean = X_train.copy()
        if "Target" in X_clean.columns:
            X_clean = X_clean.drop("Target", axis=1)
        if "customerID" in X_clean.columns:
            X_clean = X_clean.drop("customerID", axis=1)

        # Identify column types more robustly
        categorical_features = []
        numerical_features = []

        for col in X_clean.columns:
            if X_clean[col].dtype == "object":
                categorical_features.append(col)
            else:
                numerical_features.append(col)

        print(
            f"🗺️  Categorical features ({len(categorical_features)}): {categorical_features}"
        )
        print(
            f"🗺️  Numerical features ({len(numerical_features)}): {numerical_features}"
        )

        # Create improved transformers with imputation
        numerical_pipeline = Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
            ]
        )

        categorical_pipeline = Pipeline(
            [
                ("imputer", SimpleImputer(strategy="constant", fill_value="missing")),
                (
                    "encoder",
                    OneHotEncoder(
                        drop="first", handle_unknown="ignore", sparse_output=False
                    ),
                ),
            ]
        )

        # Create improved preprocessor
        preprocessor = ColumnTransformer(
            transformers=[
                ("num", numerical_pipeline, numerical_features),
                ("cat", categorical_pipeline, categorical_features),
            ],
            remainder="drop",  # Remove any unspecified columns
        )

        self.preprocessor = preprocessor
        print("✅ Improved preprocessor created successfully")
        return preprocessor

    # --- 2) Helper to choose preprocessor by model
    def _pick_preprocessor(self, model_name: str, X: pd.DataFrame) -> ColumnTransformer:
        if model_name == "Logistic_Regression":
            return self.create_preprocessor_logistic_regression(X)
        if model_name == "KNN":
            return self.create_preprocessor_knn(X)
        if model_name == "Naive_Bayes":
            return self.create_preprocessor_naive_bayes(X)
        if model_name == "Random_Forest":
            return self.create_preprocessor_random_forest(X)
        if model_name == "Gradient_":
            return self.create_preprocesor_gradient_boosting(X)
        # Fallback (your generic one)
        return self.create_preprocessor(X)

    
    def create_pipeline(self, classifier):
        """
        Builds a complete pipeline with:
        - Optional feature engineering
        - Numeric and categorical preprocessing
        - Final model injected as parameter
        """
    
        # 🔧 Numeric and categorical columns (adjust according to your data)
        fe = FeatureEngineer()
        numeric_features, binary_features, categorical_features  = fe._categorize_features(X)
        #Ejemplo de categorización para el dataset Telco Churn.
        # num_features = ['SeniorCitizen', 'tenure', 'MonthlyCharges', 'TotalCharges', 'avg_monthly']
        # bin_features = ['Partner', 'Dependents', 'PhoneService', 'PaperlessBilling', 
        #                 'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 
        #                 'TechSupport', 'StreamingTV', 'StreamingMovies']
    
        # cat_features = ['gender', 'InternetService', 'Contract', 'PaymentMethod']
    
      
        def _normalize_and_engineer(Z: pd.DataFrame) -> pd.DataFrame:
            Z = self._normalize_service_values(Z.copy())
            fe = FeatureEngineer()
            return fe.transform(Z)

    
        feature_engineering = FunctionTransformer(_normalize_and_engineer)
    
        # ⚙️ Preprocessing
        num_pipeline = Pipeline(steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler())
        ])
    
        bin_pipeline = Pipeline(steps=[
            ("imputer", SimpleImputer(strategy="most_frequent"))
            # No scaler or encoding, already in 0/1 or Yes/No format
        ])
    
        cat_pipeline = Pipeline(steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(drop="first", handle_unknown="ignore"))
        ])
    
        preprocessor = ColumnTransformer(transformers=[
            ("num", num_pipeline, num_features),
            ("bin", "passthrough", bin_features),  # assume they're already preprocessed as 0/1
            ("cat", cat_pipeline, cat_features)
        ])
    
        # 🧪 Complete pipeline
        pipeline = Pipeline(steps=[
            ("features", feature_engineering),
            ("preprocessor", preprocessor),
            ("classifier", classifier)
        ])
    
        return pipeline

   
    def create_models(self, X: pd.DataFrame):
        """
        Create dictionary of models with algorithm-optimized pipelines.
        Requires X with all columns (features without target).
        """
        print("🗺️  Creating models...")

        # Common step: feature engineering + service string normalization
        def _normalize_and_engineer(Z: pd.DataFrame) -> pd.DataFrame:
            Z = self._normalize_service_values(Z.copy())
            fe = FeatureEngineer()
            return fe.transform(Z)

        feat_step = (
            "features",
            FunctionTransformer(_normalize_and_engineer, validate=False),
        )

        # Specific preprocessors (calculated with columns after feature engineering)
        X_for_schema = _normalize_and_engineer(X)

        preprocessors = {
            "Logistic_Regression": self._pick_preprocessor(
                "Logistic_Regression", X_for_schema
            ),
            "Random_Forest": self._pick_preprocessor("Random_Forest", X_for_schema),
            "Naive_Bayes": self._pick_preprocessor("Naive_Bayes", X_for_schema),
            "KNN": self._pick_preprocessor("KNN", X_for_schema),
            "Gradient_Boosting": self._pick_preprocessor(
                "Gradient_Boosting", X_for_schema
            ),
        }

        df_view = self._column_transformers_to_df(preprocessors)
        if running_context == "jupyter_notebook":
            display(df_view)  # in notebook
        else:
            print(df_view.to_string(index=False))  # in script

        print("✅ Preprocessors were configured successfully")

        models = {
            "Logistic_Regression": Pipeline(
                [
                    feat_step,
                    ("preprocessor", preprocessors["Logistic_Regression"]),
                    (
                        "classifier",
                        LogisticRegression(
                            random_state=self.random_state,
                            max_iter=1000,
                            class_weight="balanced",  # IMPROVEMENT: Class balancing --> IMPORTANT FOR TELCO Target with high imbalance between No Target / Target
                        ),
                    ),
                ]
            ),
            "Random_Forest": Pipeline(
                [
                    feat_step,
                    ("preprocessor", preprocessors["Random_Forest"]),
                    (
                        "classifier",
                        RandomForestClassifier(
                            random_state=self.random_state,
                            n_estimators=200,
                            max_depth=10,
                            min_samples_split=10,
                            min_samples_leaf=5,
                            class_weight="balanced",  # IMPROVEMENT: Class balancing
                        ),
                    ),
                ]
            ),
            "Gradient_Boosting": Pipeline(
                [
                    feat_step,
                    (
                        "preprocessor",
                        preprocessors["Random_Forest"],
                    ),  # Use same preprocessor as RF
                    (
                        "classifier",
                        GradientBoostingClassifier(
                            random_state=self.random_state,
                            n_estimators=150,
                            learning_rate=0.1,
                            max_depth=6,
                        ),
                    ),
                ]
            ),
            "Naive_Bayes": Pipeline(
                [
                    feat_step,
                    ("preprocessor", preprocessors["Naive_Bayes"]),
                    ("classifier", GaussianNB()),
                ]
            ),
            "KNN": Pipeline(
                [
                    feat_step,
                    ("preprocessor", preprocessors["KNN"]),
                    (
                        "classifier",
                        KNeighborsClassifier(n_neighbors=7, weights="distance"),
                    ),
                ]
            ),
        }

        self.models = models
        print(f"✅ {len(models)} models created:")
        for name in models.keys():
            print(f"  - {name}")
        return models

    def train_models(self, X_train, y_train):
        """
        Train all models with progress bar

        Args:
            X_train: Training features
            y_train: Training target variable
        """
        print("🗺️  Starting model training...")
        print(f"🗺️  Total models to train: {len(self.models)}")

        # Variables for time tracking
        start_time = time.time()
        model_times = []
        successful_models = 0
        failed_models = 0

        # Configure progress bar - SIMPLE AND RELIABLE VERSION
        models_list = list(self.models.items())

        # if TQDM_AVAILABLE:
        #     progress_iterator = tqdm(
        #         models_list,
        #         desc="Training models",
        #         unit="model",
        #         ascii=True,  # ASCII for maximum compatibility
        #         ncols=70,  # Fixed width
        #         leave=True,  # Leave visible for debug
        #     )
        #     use_tqdm = True
        # else:
        #     progress_iterator = models_list
        #     use_tqdm = False

        try:
            for i, (name, model) in enumerate(models_list, 1):
                model_start_time = time.time()

                try:
                    print(f"[{i}/{len(self.models)}] Training {name}...")

                    # Train the model
                    model.fit(X_train, y_train)

                    # Calculate training time
                    model_time = time.time() - model_start_time
                    model_times.append(model_time)
                    successful_models += 1

                    print(f"✅ {name} trained in {model_time:.1f}s")

                except Exception as e:
                    failed_models += 1
                    model_time = time.time() - model_start_time
                    model_times.append(model_time)

                    print(f"❌Error training {name}: {e}")

            # Final summary - SIMPLE AND CLEAR
            total_time = time.time() - start_time
            avg_time = sum(model_times) / len(model_times) if model_times else 0

            # Clean output
            # if use_tqdm:
            #     time.sleep(0.1)  # Small pause

            print()  # Blank line
            print("=" * 50)
            print("TRAINING COMPLETED")
            print("=" * 50)
            print(f"Total time: {total_time:.1f}s")
            print(f"Average time per model: {avg_time:.1f}s")
            print(f"✅ Successful models: {successful_models}")
            if failed_models > 0:
                print(f"❌ Failed models: {failed_models}")
            print(f"Success rate: {(successful_models/len(self.models)*100):.1f}%")
            print("=" * 50)
        except Exception as e:
            print(f"❌General training error: {e}")
            print(f"      Error type: {type(e).__name__}")
            if hasattr(e, "__cause__") and e.__cause__:
                print(f"      Original cause: {e.__cause__}")

    def evaluate_models(self, X_test, y_test):
        """
        Evaluate all trained models

        Args:
            X_test: Test features
            y_test: Test target variable

        Returns:
            dict: Evaluation results
        """
        print("🗺️  Evaluating models...")

        results = {}

        for name, model in self.models.items():
            try:
                print(f"🗺️  Evaluating {name}...")

                # Predictions
                y_pred = model.predict(X_test)
                y_pred_proba = model.predict_proba(X_test)[:, 1]

                # Metrics
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

                print(f"✅ {name} evaluated - ROC AUC: {roc_auc:.4f}")

            except Exception as e:
                print(f"❌Error evaluating {name}: {e}")
                results[name] = {
                    "Accuracy": 0.0,
                    "Precision": 0.0,
                    "Recall": 0.0,
                    "F1_Score": 0.0,
                    "ROC_AUC": 0.0,
                }

        self.results = results
        print("✅ Evaluation completed!")
        return results

    def get_best_model(self, metric="ROC_AUC", results=None):
        """
        Get the best model based on a metric

        Args:
            metric (str): Metric to select the best model
            results (dict): Evaluation results (optional)

        Returns:
            tuple: (model_name, model)
        """
        if results is None:
            results = self.results

        print(f"🗺️  Selecting best model by {metric}...")

        best_score = -1
        best_model_name = None

        for name, metrics in results.items():
            if metrics[metric] > best_score:
                best_score = metrics[metric]
                best_model_name = name

        if best_model_name is None:
            print("❌No valid model found")
            return None, None

        best_model = self.models[best_model_name]

        print(f"✅ Best model: {best_model_name} ({metric}: {best_score:.4f})")

        return best_model_name, best_model

    def predict_proba(self, X):
        """Probability predictions"""
        X_processed = self.preprocessor.transform(X)
        return self.model.predict_proba(X_processed)

    def predict(self, X):
        """Binary predictions"""
        X_processed = self.preprocessor.transform(X)
        return self.model.predict(X_processed)

    def generate_model_report(self, X_test, y_test):
        """
        Generate detailed model report

        Args:
            X_test: Test features
            y_test: Test target variable
        """
        print("🗺️  Generating model report...")

        print("\n" + "=" * 60)
        print("DETAILED MODEL REPORT")
        print("=" * 60)

        for name, model in self.models.items():
            try:
                print(f"\nMODEL: {name}")
                print("-" * 40)

                y_pred = model.predict(X_test)
                y_pred_proba = model.predict_proba(X_test)[:, 1]

                # Basic metrics
                # print('Call accuracy_score()')
                accuracy = accuracy_score(y_test, y_pred)
                # print('Call precision_score()')
                precision = precision_score(
                    y_test, y_pred, average="weighted", zero_division=0
                )
                # print('Call recall_score()')
                recall = recall_score(
                    y_test, y_pred, average="weighted", zero_division=0
                )
                # print('Call f1_score()')
                f1 = f1_score(y_test, y_pred, average="weighted", zero_division=0)
                # print('Call roc_auc_score()')
                roc_auc = roc_auc_score(y_test, y_pred_proba)

                print(f"Accuracy:  {accuracy:.4f}")
                print(f"Precision: {precision:.4f}")
                print(f"Recall:    {recall:.4f}")
                print(f"F1-Score:  {f1:.4f}")
                print(f"ROC AUC:   {roc_auc:.4f}")

                # Confusion matrix
                self.show_confusion_matrix(y_test, y_pred, name)

            except Exception as e:
                print(f"❌Error generating report for {name}: {e}")

        print("\n" + "=" * 60)
        print("✅ Report completed")

    def show_confusion_matrix(self, y_test, y_pred, model_name):
        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        print(f"\nConfusion Matrix for {model_name}")
        print(f"   TN: {cm[0,0]:4d} | FP: {cm[0,1]:4d}")
        print(f"   FN: {cm[1,0]:4d} | TP: {cm[1,1]:4d}")

    def prepare_data(self, df,target_col):
        """
        Prepare data by separating features (X) and target variable (y)

        Args:
            df: DataFrame with complete data including target

        Returns:
            tuple: (X, y) - features and target variable
        """
        print("🗺️  Preparing data...")

        # Verify DataFrame is not empty
        if df is None or df.empty:
            raise ValueError("DataFrame cannot be None or empty")

        # Make a copy to not modify the original
        df_copy = df.copy()

        # Identify target column
        if target_col not in df_copy.columns:
            raise ValueError(
                f"Target column '{target_col}' not found in DataFrame"
            )

        # Separate features and target
        y = df_copy[target_col]

        # Remove columns that are not features
        columns_to_drop = [target_col]
        if "customerID" in df_copy.columns:
            columns_to_drop.append("customerID")

        X = df_copy.drop(columns=columns_to_drop, axis=1)

        # Map target variable if necessary
        y_mapped = self.map_target(y)

        print(f"✅ Data prepared: X{X.shape}, y{y_mapped.shape}")
        print(f"📊 Features: {list(X.columns)}")
        print(f"📊 Target distribution: {y_mapped.value_counts().to_dict()}")

        return X, y_mapped

    def create_advanced_features(self, df):
        """
        IMPROVEMENT: Advanced feature engineering function for Target prediction

        Creates more predictive features based on data analysis:
        - Financial features: ratios, high charge detection
        - Service features: service count, protection services
        - Contract features: short/long contract identification
        - Demographic features: improved processing

        Args:
            df (pd.DataFrame): Original dataset

        Returns:
            pd.DataFrame: Dataset with improved features
        """
        df_improvement = df.copy()

        # Remove Target column if it exists (to avoid data leakage in test)
        if "Target" in df_improvement.columns:
            df_improvement = df_improvement.drop("Target", axis=1)

        # Financial features
        if "MonthlyCharges" in df.columns and "TotalCharges" in df.columns:
            # Convert TotalCharges to numeric
            df_improvement["TotalCharges"] = pd.to_numeric(
                df_improvement["TotalCharges"], errors="coerce"
            )
            df_improvement["TotalCharges"] = df_improvement["TotalCharges"].fillna(
                df_improvement["TotalCharges"].median()
            )

            # New financial features
            df_improvement["Charge_Per_Month_Ratio"] = df_improvement[
                "TotalCharges"
            ] / (df_improvement["tenure"] + 1)
            df_improvement["High_Monthly_Charges"] = (
                df_improvement["MonthlyCharges"]
                > df_improvement["MonthlyCharges"].quantile(0.75)
            ).astype(int)
            df_improvement["High_Total_Charges"] = (
                df_improvement["TotalCharges"]
                > df_improvement["TotalCharges"].quantile(0.75)
            ).astype(int)

        # Service features
        service_cols = [
            "PhoneService",
            "MultipleLines",
            "InternetService",
            "OnlineSecurity",
            "OnlineBackup",
            "DeviceProtection",
            "TechSupport",
            "StreamingTV",
            "StreamingMovies",
        ]

        available_services = [col for col in service_cols if col in df.columns]
        if available_services:
            # Count active services
            df_improvement["Total_Services"] = 0
            for col in available_services:
                if col == "PhoneService":
                    df_improvement["Total_Services"] += (
                        df_improvement[col] == "Yes"
                    ).astype(int)
                elif col == "InternetService":
                    df_improvement["Total_Services"] += (
                        df_improvement[col] != "No"
                    ).astype(int)
                else:
                    df_improvement["Total_Services"] += (
                        df_improvement[col] == "Yes"
                    ).astype(int)

            # Protection services
            protection_services = [
                "OnlineSecurity",
                "OnlineBackup",
                "DeviceProtection",
                "TechSupport",
            ]
            available_protection = [
                col for col in protection_services if col in df.columns
            ]
            if available_protection:
                df_improvement["Protection_Services"] = 0
                for col in available_protection:
                    df_improvement["Protection_Services"] += (
                        df_improvement[col] == "Yes"
                    ).astype(int)

        # Demographic features
        if "SeniorCitizen" in df.columns:
            df_improvement["SeniorCitizen_Int"] = df_improvement[
                "SeniorCitizen"
            ].astype(int)

        # Contract features
        if "Contract" in df.columns:
            df_improvement["Short_Contract"] = (
                df_improvement["Contract"] == "Month-to-month"
            ).astype(int)
            df_improvement["Long_Contract"] = (
                df_improvement["Contract"] == "Two year"
            ).astype(int)

        if "PaymentMethod" in df.columns:
            df_improvement["Electronic_Payment"] = (
                df_improvement["PaymentMethod"] == "Electronic check"
            ).astype(int)

        return df_improvement

    def map_target(self, y):
        """
        Map target variable to numeric format

        Args:
            y: Target variable (can be 'Yes'/'No' or 0/1)

        Returns:
            Series/Array: Target variable mapped to 0/1
        """
        print("🗺️  Mapping target variable...")

        # Convert to pandas Series if numpy array
        if isinstance(y, np.ndarray):
            y = pd.Series(y)
            print("🔄 Converted numpy array to pandas Series")

        # Check if text or numeric
        if hasattr(y, "dtype") and y.dtype == "object":
            # If text, map to numbers
            if hasattr(y, "map"):
                y_mapped = y.map({"No": 0, "Yes": 1})
                print("✅ Target variable mapped: 'No'->0, 'Yes'->1")
            else:
                # Fallback for arrays
                y_mapped = np.where(y == "Yes", 1, 0)
                y_mapped = pd.Series(y_mapped)
                print("✅ Target variable mapped with np.where: 'No'->0, 'Yes'->1")
        else:
            # If already numeric, keep as is
            y_mapped = y
            print("✅ Target variable is already numeric")

        return y_mapped



# def hyperparameter_tuning(model, param_grid, X_train, y_train, cv=5, scoring="roc_auc", mode="auto", n_iter=20):
#     """
#     Optimizes hyperparameters with GridSearchCV or RandomizedSearchCV.
#     Shows progress with tqdm.
#     """
#     print("🧪 Starting hyperparameter optimization...")
#     print(f"   - CV: {cv} | Scoring: {scoring} | Mode: {mode}")

#     # Validate keys
#     valid_keys = set(model.get_params().keys())
#     unknown = [k for k in param_grid if k not in valid_keys]
#     if unknown:
#         raise ValueError(f"Invalid keys in grid: {unknown}")

#     # Detect mode if auto
#     if mode == "auto":
#         mode = "quick" if any(isinstance(v, rv_continuous) for v in param_grid.values()) else "accuracy"

#     use_random = mode == "quick"
#     cv_splitter = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)

#     # Build search object
#     search = RandomizedSearchCV(
#         estimator=model,
#         param_distributions=param_grid,
#         n_iter=n_iter,
#         cv=cv_splitter,
#         scoring=scoring,
#         n_jobs=-1,
#         verbose=0,
#         random_state=42,
#         error_score=np.nan,
#     ) if use_random else GridSearchCV(
#         estimator=model,
#         param_grid=param_grid,
#         cv=cv_splitter,
#         scoring=scoring,
#         n_jobs=-1,
#         verbose=0,
#         error_score=np.nan,
#     )

#     print("⏳ Executing search...")
#     start = datetime.now()
#     print(f"🕒 Start: {start.strftime('%H:%M:%S')}")

#     # Show progress bar
#     with parallel_backend("loky"):
#         tqdm_search = tqdm(desc="🔍 Optimizing..", total=n_iter if use_random else None)
#         search.fit(X_train, y_train)
#         tqdm_search.update()

#     end = datetime.now()
#     print(f"✅ Finished in {end - start} (End: {end.strftime('%H:%M:%S')})")

#     # Show results
#     print(f"🎯 Best CV score: {search.best_score_:.4f}")
#     print(f"🔍 Best parameters: {search.best_params_}")

#     return search


def hyperparameter_tuning(model, param_grid, X_train, y_train, cv=5, scoring="roc_auc", mode="auto", n_iter=20):
    """
    Robust hyperparameter optimization for any classification model.
    
    Parameters:
    - model: pipeline with estimator ('classifier')
    - param_grid: grid or hyperparameter distribution
    - X_train, y_train: training data
    - cv: number of cross-validation folds
    - scoring: metric (default 'roc_auc')
    - mode: 'quick' (random), 'accuracy' (grid), or 'auto'
    - n_iter: only used in 'quick' mode
    
    Returns:
    - Trained GridSearchCV or RandomizedSearchCV object
    """

    print("🧪 Starting hyperparameter optimization...")
    print(f"   - CV: {cv} | Scoring: {scoring} | Mode: {mode}")

    # Preprocess as the pipeline does
    X_train_proc = self._normalize_service_values(X_train.copy())
    X_train_proc = FeatureEngineer().transform(X_train_proc)

    # Validate pipeline keys
    valid_keys = set(model.get_params().keys())
    unknown = [k for k in param_grid if k not in valid_keys]
    if unknown:
        print("❌ Invalid keys detected:")
        for k in unknown:
            print(f"   - {k}")
        return None

    # Determine actual mode if in 'auto'
    if mode == "auto":
        mode = "quick" if any(isinstance(v, rv_continuous) for v in param_grid.values()) else "accuracy"

    print(f"🚀 Search type: {'RandomizedSearchCV' if mode == 'quick' else 'GridSearchCV'}")

    # Stratified CV
    cv_splitter = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)

    # Define search type
    if mode == "quick":
        search = RandomizedSearchCV(
            estimator=model,
            param_distributions=param_grid,
            n_iter=n_iter,
            cv=cv_splitter,
            scoring=scoring,
            n_jobs=-1,
            verbose=1,
            random_state=42,
            error_score=np.nan,
        )
    else:
        search = GridSearchCV(
            estimator=model,
            param_grid=param_grid,
            cv=cv_splitter,
            scoring=scoring,
            n_jobs=-1,
            verbose=1,
            error_score=np.nan,
        )

    # Execute
    print("⏳ Executing search...")
    start_time = datetime.now()
    print(f"🕒 Start: {start_time.strftime('%H:%M:%S')}")

    try:
        search.fit(X_train, y_train)
    except Exception as e:
        print(f"❌ Error during fit: {e}")
        return None

    end_time = datetime.now()
    duration = end_time - start_time
    print(f"✅ Search completed in {duration} (End: {end_time.strftime('%H:%M:%S')})")

    # Show results
    try:
        print(f"🎯 Best CV score: {search.best_score_:.4f}")
        print(f"🔍 Best parameters: {search.best_params_}")
    except Exception as e:
        print(f"⚠️ Error showing results: {e}")
        return None

    return search


# Utility function to show module information
def show_module_info():
    """
    Show module information
    """
    print("=" * 60)
    print("MODELS MODULE - STABLE VERSION")
    print("=" * 60)
    print("Version without emojis optimized for maximum compatibility")
    print("Includes:")
    print("   - TargetPredictor - Main class for modeling")
    print("   - hyperparameter_tuning - Hyperparameter optimization")
    print("   - Report and evaluation functions")
    print("=" * 60)


if __name__ == "__main__":
    show_module_info()
