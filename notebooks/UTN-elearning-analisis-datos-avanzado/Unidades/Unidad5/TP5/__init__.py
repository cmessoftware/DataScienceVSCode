# TP5 Package
# Módulos para el proyecto de predicción de fuga de clientes

"""
TP5 - Predicción de Fuga de Clientes
===================================

Este paquete contiene los módulos necesarios para el proyecto de Machine Learning
de predicción de abandono de clientes (churn prediction).

Módulos incluidos:
- data_loader: Carga y gestión de datos
- dataset_splitter: División de datasets
- eda: Análisis Exploratorio de Datos
- models: Modelos de Machine Learning
- metrics: Cálculo y evaluación de métricas
"""

__version__ = "1.0.0"
__author__ = "Grupo M - UTN"

# Importaciones principales
from . import data_loader
from . import dataset_splitter
from . import eda
from . import models
from . import metrics

__all__ = [
    'data_loader',
    'dataset_splitter', 
    'eda',
    'models',
    'metrics'
]
