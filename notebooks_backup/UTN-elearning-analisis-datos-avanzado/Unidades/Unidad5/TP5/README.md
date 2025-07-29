# Data Science VSCode - Entorno Docker

Este proyecto proporciona un entorno completo de Data Science utilizando Docker y Jupyter Lab, optimizado para el análisis de datos avanzado.

## 🚀 Inicio Rápido

### Prerrequisitos
- Docker Desktop instalado
- PowerShell (Windows)

### Opción 1: Script Automático (Recomendado)

#### Uso básico:
```powershell
.\quick-start.ps1
```

#### Opciones avanzadas:
```powershell
.\quick-start.ps1 -Help                    # Ver ayuda
.\quick-start.ps1 -SkipBuild              # Solo iniciar (imagen ya existe)
.\quick-start.ps1 -ShowLogs               # Mostrar logs al final
.\quick-start.ps1 -OpenBrowser:$false     # No abrir navegador automáticamente
```

#### Detener servicios:
```powershell
.\quick-stop.ps1
```

### Opción 2: Configuración Manual

1. **Construir la imagen Docker:**
   ```powershell
   .\docker-helper.ps1 build
   ```

2. **Iniciar Jupyter Lab:**
   ```powershell
   .\docker-helper.ps1 start
   ```

3. **Acceder a Jupyter Lab:**
   - URL: http://localhost:8888
   - Token: `datascience2024`

## 📋 Comandos Disponibles

### Gestión de Servicios
```powershell
.\docker-helper.ps1 build     # Construir imagen
.\docker-helper.ps1 start     # Iniciar servicios
.\docker-helper.ps1 stop      # Detener servicios
.\docker-helper.ps1 restart   # Reiniciar servicios
.\docker-helper.ps1 status    # Ver estado
```

### Desarrollo y Debugging
```powershell
.\docker-helper.ps1 shell     # Abrir shell en contenedor
.\docker-helper.ps1 python    # Consola Python interactiva
.\docker-helper.ps1 logs      # Ver logs
```

### Gestión de Paquetes
```powershell
.\docker-helper.ps1 install pandas    # Instalar paquete
.\docker-helper.ps1 install numpy     # Instalar otro paquete
```

### Utilidades
```powershell
.\docker-helper.ps1 backup    # Backup de notebooks
.\docker-helper.ps1 info      # Información de Jupyter
.\docker-helper.ps1 clean     # Limpiar Docker
.\docker-helper.ps1 help      # Mostrar ayuda
```

## 📁 Estructura del Proyecto

```
DataScienceVSCode/
├── UTN-elearning-analisis-datos-avanzado/  # Notebooks principales
│   ├── notebooks/
│   │   ├── clases/
│   │   │   ├── ch1/ - ch4/                 # Capítulos del curso
│   │   │   └── tbc/                        # Análisis TBC
│   │   └── custom_tools/
├── docker-helper.ps1                       # Script de gestión
├── docker-compose.yml                      # Configuración Docker
├── Dockerfile                              # Imagen personalizada
└── requirements.txt                        # Dependencias Python
```

## 🛠️ Paquetes Incluidos

### Core Data Science
- **pandas** - Manipulación de datos
- **numpy** - Computación numérica
- **matplotlib** - Visualización básica
- **seaborn** - Visualización estadística
- **plotly** - Visualización interactiva

### Machine Learning
- **scikit-learn** - Algoritmos ML
- **statsmodels** - Modelos estadísticos
- **scipy** - Computación científica

### Jupyter Ecosystem
- **jupyterlab** - Interfaz principal
- **ipywidgets** - Widgets interactivos
- **jupyter-contrib-nbextensions** - Extensiones

### Utilidades
- **tqdm** - Barras de progreso
- **pyarrow** - Formato de datos eficiente
- **openpyxl** - Lectura/escritura Excel

## 🔧 Configuración Avanzada

### Puertos
- **8888**: Jupyter Lab (principal)
- **8889**: Puerto alternativo (disponible)

### Volúmenes
- **Código fuente**: `/workspace` (todo el proyecto)
- **Notebooks**: `/workspace/notebooks` (UTN notebooks)
- **Labs**: `/workspace/ISLP_labs` (ISLP exercises)

### Variables de Entorno
- `JUPYTER_TOKEN=datascience2024`
- `JUPYTER_ROOT_DIR=/workspace`

## 🚨 Solución de Problemas

### Error de permisos
```powershell
# En PowerShell como administrador
Set-ExecutionPolicy RemoteSigned
```

### Error de permisos con scripts
```powershell
# Ejecutar solo para la sesión actual
Set-ExecutionPolicy Bypass -Scope Process -Force
.\quick-start.ps1
```

### PowerShell no reconoce el script
```powershell
# Usar ruta completa
PowerShell.exe -ExecutionPolicy Bypass -File ".\quick-start.ps1"
```

### Puerto ocupado
```powershell
# Cambiar puerto en docker-compose.yml
ports:
  - "8801:8802"  # Usar puerto alternativo
```

### Memoria insuficiente
```powershell
# Aumentar memoria en Docker Desktop
# Settings > Resources > Memory > 4GB+
```

### Ejercicios por Capítulo
- **Unidad1**: Análisis de familias
- **Unidad2**: Distribuciones de probabilidad
- **Unidad3**: Modelos binomiales e hipergeométricos
- **Unidad4**: Actividades prácticas con MPG dataset
- **Unidad5**: Predicciones y clasificaciones, introducción ML.

---

## 📒 Tutorial: Notebook de Predicción de Fuga de Clientes (Churn)

Este proyecto incluye un notebook completo para el problema de predicción de abandono de clientes (churn), ideal para competencias de Kaggle y prácticas de Machine Learning.

### 📁 Ubicación del notebook

```
notebooks/
└── UTN-elearning-analisis-datos-avanzado/
    └── Unidades/
        └── Unidad5/
            └── TP5/
                └── tp5_grupoM.ipynbP
                └── tp5_grupoM_backup.ipynb
```

## 📖 Diccionario de datos — Telco Customer Churn

Descripción de cada campo en el dataset, en lenguaje de negocio:

| Campo | Descripción |
| ----- | ----------- |
| **customerID** | Identificador único del cliente. Solo referencia; no es útil como predictor. |
| **gender** | Género del cliente (`Male` / `Female`). |
| **SeniorCitizen** | Si el cliente es adulto mayor (`1` = sí; `0` = no). |
| **Partner** | Si el cliente tiene pareja (`Yes` / `No`). |
| **Dependents** | Si tiene dependientes (hijos, familiares a cargo) (`Yes` / `No`). |
| **tenure** | Antigüedad: cantidad de meses como cliente. Predictor clave: churn tiende a ser mayor en clientes nuevos. |
| **PhoneService** | Si tiene línea telefónica (`Yes` / `No`). |
| **MultipleLines** | Si tiene más de una línea (`Yes` / `No` / `No phone service`). |
| **InternetService** | Tipo de conexión a internet (`DSL` / `Fiber optic` / `No`). |
| **OnlineSecurity** | Si tiene servicio de seguridad online contratado (`Yes` / `No` / `No internet service`). |
| **OnlineBackup** | Si tiene servicio de backup online contratado (`Yes` / `No` / `No internet service`). |
| **DeviceProtection** | Si tiene protección de dispositivos (`Yes` / `No` / `No internet service`). |
| **TechSupport** | Si tiene soporte técnico contratado (`Yes` / `No` / `No internet service`). |
| **StreamingTV** | Si tiene servicio de streaming TV contratado (`Yes` / `No` / `No internet service`). |
| **StreamingMovies** | Si tiene servicio de streaming de películas (`Yes` / `No` / `No internet service`). |
| **Contract** | Tipo de contrato (`Month-to-month` / `One year` / `Two year`). Alta relevancia: contratos más largos tienden a tener menor churn. |
| **PaperlessBilling** | Si usa facturación electrónica (`Yes` / `No`). |
| **PaymentMethod** | Método de pago (`Electronic check` / `Mailed check` / `Bank transfer (automatic)` / `Credit card (automatic)`). |
| **MonthlyCharges** | Importe mensual facturado al cliente (en USD). |
| **TotalCharges** | Total acumulado facturado durante toda la relación comercial. |
| **Churn** | 🎯 **Variable objetivo:** indica si el cliente abandonó (`Yes`) o sigue (`No`). |

---

✅ **Notas importantes:**
- Muchos campos son categóricos y requieren encoding adecuado.
- `TotalCharges` contiene algunos valores no numéricos (" ") que deben limpiarse antes de usar.
- `Churn` está desbalanceado (~20% Yes), por lo que deben usarse métricas y técnicas adecuadas.

---

Este diccionario sirve como referencia de negocio para comprender la estructura de datos y facilitar el análisis exploratorio (EDA) y modelado predictivo.


### 📝 ¿Qué contiene el notebook?
- **Introducción y contexto del problema**
- **Importación de librerías y módulos**
- **Carga y exploración de datos**
- **EDA (Análisis exploratorio de datos)**
- **Preprocesamiento y limpieza**
- **Entrenamiento de modelos (Logistic Regression, k-NN, Naive Bayes, etc.)**
- **Evaluación y selección del mejor modelo**
- **Generación de archivo de submission para Kaggle**
- **Conclusiones y recomendaciones**

### 🚦 ¿Cómo usar el notebook?

1. **Accede a Jupyter Lab**
   - URL: http://localhost:8888
   - Token: `datascience2024`

2. **Navega a la carpeta  `notebooks/UTN-elearning-analisis-datos-avanzado/Unidades/Unidad5/TP5/`**

3. **Abre el notebook `tp5_grupoM_backup.ipynb`**
   - Si el archivo principal da error, usa el backup.
   - Puedes renombrar el backup si lo deseas.

4. **Ejecuta las celdas paso a paso**
   - Sigue las instrucciones y comentarios en cada celda.
   - Modifica los parámetros y código según tu equipo y datos.

5. **Carga tus datasets**
   - Coloca los archivos `train.csv`, `test.csv` y `sample_submission.csv` en la misma carpeta que el notebook.
   - Si no tienes los datos, el notebook genera datos de ejemplo para pruebas.

6. **Entrena y evalúa modelos**
   - El notebook incluye código para entrenar varios modelos y comparar resultados.
   - Puedes agregar nuevos modelos o modificar los existentes.

7. **Genera el archivo de submission para Kaggle**
   - Sigue las instrucciones en la última sección para crear el archivo `.csv` listo para subir a la competencia.
  


## 🔧 Conceptos Clave de Machine Learning

### 📊 Preparación de Datos (Feature Engineering)

#### **¿Por qué remover `Churn` y `customerID` al crear características?**

En el notebook, verás esta línea de código:
```python
# Extraer características (X_features) - remover Churn y customerID
columns_to_drop = ['Churn']
if 'customerID' in X_train.columns:
    columns_to_drop.append('customerID')

X_features = X_train.drop(columns_to_drop, axis=1)
```

**Explicación:**

1. **Remover `Churn` (Variable Objetivo)**
   - **`Churn`** es la **variable que queremos predecir** (target/objetivo)
   - En machine learning, **NO puedes usar la variable objetivo como característica** para entrenar el modelo
   - Sería como hacer trampa: "predice si el cliente se va, usando como dato si el cliente se va"
   - **Separación obligatoria**:
     - `y = X_train['Churn']` → Lo que queremos predecir
     - `X_features` → Las características que usamos para hacer la predicción

2. **Remover `customerID` (Identificador)**
   - **`customerID`** es solo un **identificador único** (como "CUST001", "CUST002")
   - **No aporta información predictiva** sobre si un cliente se irá o no
   - Los modelos podrían memorizar estos IDs y crear **overfitting**
   - Es información administrativa, no predictiva

#### **Analogía práctica:**
Imagina que quieres predecir si va a llover:

```python
# ❌ INCORRECTO
X_features = ['temperatura', 'humedad', 'presión', 'va_a_llover', 'id_medicion']
y = 'va_a_llover'  # ¡Usas lo mismo que quieres predecir!

# ✅ CORRECTO  
X_features = ['temperatura', 'humedad', 'presión']  # Solo características útiles
y = 'va_a_llover'  # Lo que quieres predecir
```

#### **Flujo correcto en el proyecto:**
```python
# 1. Extraer variable objetivo
y = X_train['Churn']  # Target: 0 (no churn) o 1 (churn)

# 2. Extraer características (sin Churn y customerID)  
X_features = X_train.drop(['Churn', 'customerID'], axis=1)

# 3. Dividir en train/validation
X_train_split, X_val, y_train_split, y_val = train_test_split(
    X_features, y, test_size=0.2, random_state=42, stratify=y
)
```


### 💡 Consejos útiles
- Si el notebook original no abre, usa el backup (`_backup.ipynb`).
- Puedes duplicar el notebook para hacer pruebas sin perder el original.
- Si tienes errores de importación, revisa que los módulos `.py` estén en la misma carpeta.
- Usa las celdas de markdown para documentar tu trabajo y conclusiones.

### 🏆 ¿Qué aprenderás?
- Práctica completa de un workflow de Machine Learning real.
- Cómo preparar datos, entrenar y evaluar modelos.
- Cómo participar en competencias de Kaggle.
- Cómo documentar y presentar resultados en notebooks.

---

