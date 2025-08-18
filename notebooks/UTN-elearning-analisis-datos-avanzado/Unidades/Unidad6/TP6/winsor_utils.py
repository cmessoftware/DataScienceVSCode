import numpy as np
from copy import deepcopy
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator,TransformerMixin

class QuantileClipper(BaseEstimator, TransformerMixin):
    """
    Recorta (winsoriza) por columnas con cuantiles [lower_q, upper_q].
    Útil antes de escalar en modelos sensibles a outliers (LogReg/KNN/SVC).
    """
    def __init__(self, lower_q=0.01, upper_q=0.99):
        self.lower_q = lower_q
        self.upper_q = upper_q

    def fit(self, X, y=None):
        X_arr = X if isinstance(X, np.ndarray) else np.asarray(X)
        self.lower_ = np.nanquantile(X_arr, self.lower_q, axis=0)
        self.upper_ = np.nanquantile(X_arr, self.upper_q, axis=0)
        return self

    def transform(self, X):
        X_arr = X if isinstance(X, np.ndarray) else np.asarray(X)
        return np.clip(X_arr, self.lower_, self.upper_)
    
def add_winsor_to_numeric(preprocessor, step_name='num', insert_after='imputer',
                          lower_q=0.01, upper_q=0.99):
    """
    Inserta QuantileClipper en el pipeline numérico del ColumnTransformer.
    No modifica el original (deepcopy).
    """
    pre2 = deepcopy(preprocessor)
    new_trs = []
    for name, trans, cols in pre2.transformers:
        if name == step_name:
            # Si es Pipeline, insertamos
            if isinstance(trans, Pipeline):
                steps = trans.steps[:]
                # buscamos índice de inserción
                idx = 0
                for i, (n, _) in enumerate(steps):
                    if n == insert_after:
                        idx = i + 1
                        break
                steps.insert(idx, ('winsor', QuantileClipper(lower_q, upper_q)))
                trans = Pipeline(steps)
            else:
                # si era un SimpleImputer u otro, lo envolvemos
                trans = Pipeline([
                    ('wrapped', trans),
                    ('winsor', QuantileClipper(lower_q, upper_q)),
                ])
        new_trs.append((name, trans, cols))
    pre2.transformers = new_trs
    return pre2

