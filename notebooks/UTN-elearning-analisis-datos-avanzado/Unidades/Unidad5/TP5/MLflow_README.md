# 🎯 MLflow Experiment Tracking - Guía Completa

## 📋 Descripción General

Este proyecto integra **MLflow** para el seguimiento sistemático de experimentos de Machine Learning en la competencia de predicción de churn. MLflow nos permite:

- 📊 **Comparar modelos** de manera objetiva
- 🔄 **Reproducir experimentos** exactamente
- 📈 **Visualizar métricas** en tiempo real
- 🏆 **Identificar el mejor modelo** automáticamente

## 🚀 Inicio Rápido

### 1. Ejecutar Notebook con Tracking
```python
# Las celdas del notebook automáticamente registran experimentos en MLflow
# Solo ejecuta las celdas normalmente
```

### 2. Acceder al Dashboard Web
```bash
# Opción A: Usar script batch (Windows)
start_mlflow.bat

# Opción B: Comando manual
mlflow ui

# Luego abrir: http://localhost:5000
```

### 3. Análisis Avanzado
```bash
# Script de análisis personalizado
python mlflow_analysis.py
```

## 📊 Estructura de Experimentos

### Experimento Principal: `Churn_Prediction_TP5`

#### Tipos de Runs:
- **Modelos Base**: Logistic Regression, KNN, Naive Bayes, Random Forest
- **Modelos Optimizados**: Versiones con hiperparámetros optimizados
- **Comparaciones**: Runs especiales para comparar original vs optimizado

#### Métricas Registradas:
- `roc_auc`: Métrica principal de la competencia
- `accuracy`: Precisión general
- `precision`: Precisión por clase
- `recall`: Recall por clase  
- `f1_score`: F1-Score balanceado

#### Parámetros Registrados:
- Hiperparámetros específicos de cada modelo
- Configuración de cross-validation
- Métodos de optimización utilizados

## 🏆 Casos de Uso Principales

### 1. Comparar Modelos Base
```python
# El notebook automáticamente registra todos los modelos
# Accede al dashboard para ver comparación visual
```

### 2. Evaluar Impacto de Optimización
```python
# Los modelos optimizados se registran con tag "_OPTIMIZED"
# Dashboard muestra mejoras/deterioros automáticamente
```

### 3. Seleccionar Mejor Modelo
```python
# MLflow identifica automáticamente el modelo con mejor ROC AUC
# Disponible tanto en notebook como en dashboard web
```

### 4. Reproducir Experimentos
```python
# Cada run incluye todos los parámetros necesarios
# Modelos entrenados se guardan automáticamente
```

## 🔧 Archivos y Configuración

### Estructura de Archivos:
```
TP5/
├── mlruns/                    # Base de datos MLflow local
├── tp5_grupoM.ipynb          # Notebook principal con tracking integrado
├── mlflow_analysis.py        # Script de análisis avanzado
├── start_mlflow.bat          # Launcher del dashboard (Windows)
└── MLflow_README.md          # Esta guía
```

### Configuración Automática:
- **Tracking URI**: `file://./mlruns`
- **Experimento**: `Churn_Prediction_TP5`
- **Backend**: SQLite local
- **Storage**: Local filesystem

## 📈 Dashboard Web - Guía de Uso

### Navegación Principal:
1. **Experiments**: Lista de todos los experimentos
2. **Runs**: Detalle de cada ejecución
3. **Compare**: Comparación lado a lado
4. **Models**: Modelos registrados

### Funcionalidades Clave:

#### 🔍 Filtrar y Buscar:
```
# Ejemplos de filtros útiles:
tags.model_type = "LogisticRegression"
metrics.roc_auc > 0.85
params.C = 1.0
```

#### 📊 Visualizaciones:
- **Parallel Coordinates**: Relación parámetros-métricas
- **Scatter Plots**: Correlaciones entre métricas
- **Charts**: Evolución temporal de experimentos

#### 📋 Comparación de Modelos:
- Seleccionar múltiples runs
- Click en "Compare"
- Ver diferencias en parámetros y métricas

## 🎯 Mejores Prácticas Implementadas

### 1. Naming Convention
- Modelos base: `LogisticRegression`, `KNN`, etc.
- Modelos optimizados: `LogisticRegression_OPTIMIZED`
- Comparaciones: `COMPARISON_ModelName`

### 2. Tags Sistemáticos
- `model_type`: Tipo de algoritmo
- `problem_type`: "binary_classification"
- `dataset`: "telecom_churn"
- `optimization_method`: Método de optimización usado

### 3. Artifacts Registrados
- Modelos entrenados (.pkl)
- Gráficos de evaluación (.png)
- Reportes de métricas (.json)

### 4. Métricas Consistentes
- Todas las métricas estándar de clasificación
- Cross-validation scores cuando aplique
- Comparaciones con modelo baseline

## 🚨 Solución de Problemas

### Servidor MLflow no inicia:
```bash
# Verificar instalación
pip install mlflow mlflow[extras]

# Verificar puerto disponible
netstat -an | find "5000"

# Usar puerto alternativo
mlflow ui --port 5001
```

### No aparecen experimentos:
```bash
# Verificar directorio
ls mlruns/

# Verificar tracking URI
python -c "import mlflow; print(mlflow.get_tracking_uri())"
```

### Modelos no se registran:
- Verificar que las celdas de MLflow se ejecuten correctamente
- Revisar logs en el notebook para errores
- Confirmar que las variables necesarias estén disponibles

## 📞 Soporte y Recursos

### Documentación Oficial:
- [MLflow Documentation](https://mlflow.org/docs/latest/index.html)
- [MLflow Tracking Guide](https://mlflow.org/docs/latest/tracking.html)

### Comandos Útiles:
```bash
# Ver experimentos
mlflow experiments list

# Backup de experimentos
mlflow experiments export --experiment-id 1 --output backup.json

# Limpiar runs antiguos
mlflow gc --backend-store-uri file://./mlruns
```

---

**🎉 ¡Disfruta explorando tus experimentos de ML con MLflow!**

Este sistema te permite llevar un registro profesional de todos tus modelos y optimizaciones, facilitando la toma de decisiones basada en datos objetivos.
