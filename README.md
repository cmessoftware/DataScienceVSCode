# 🎓 UTN - Análisis de Datos Avanzado y Machine Learning

[![Python](https://img.shields.io/badge/Python-3.11-blue.svg)](https://python.org)
[![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-orange.svg)](https://jupyter.org)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3-green.svg)](https://scikit-learn.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0-red.svg)](https://pytorch.org)

## 📋 Descripción

Repositorio completo para el curso de **Análisis de Datos Avanzado** de la Universidad Tecnológica Nacional (UTN), que incluye implementaciones prácticas, proyectos y una colección completa de libros de referencia en Machine Learning y Estadística.

## 🚀 Inicio Rápido

### Prerrequisitos
- Docker Desktop instalado (opcional)
- Python 3.11+
- Jupyter Lab

### Opción 1: Docker (Recomendado)

#### Script automático:
```bash
./quick-start.sh      # Linux/Mac
# o
.\quick-start.ps1     # Windows
```

#### Detener servicios:
```bash
./quick-stop.sh       # Linux/Mac
# o
.\quick-stop.ps1      # Windows
```

### Opción 2: Instalación Local

1. **Instalar dependencias:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Iniciar Jupyter Lab:**
   ```bash
   jupyter lab
   ```

## 📁 Estructura del Repositorio

```
📦 DataScienceVSCode/
├── � notebooks/
│   ├── 📚 ISLP_labs/                    # Introduction to Statistical Learning with Python
│   ├── 📚 Machine-Learning-with-Pytorch-Scikit-Learn/
│   ├── 📚 mml-book.github.io/           # Mathematics for Machine Learning
│   ├── 📚 practical-statistics-for-data-scientists/
│   ├── 🏆 kaggle_competitions/
│   │   └── Titanic-Machine Learning from Disaster/
│   └── 🎓 UTN-elearning-analisis-datos-avanzado/
│       └── Unidades/
│           ├── Unidad1/ - Análisis Exploratorio
│           ├── Unidad2/ - Estadística Descriptiva
│           ├── Unidad3/ - Distribuciones de Probabilidad
│           ├── Unidad4/ - Inferencia Estadística
│           └── Unidad5/ - Machine Learning
├── 📋 Documentación/
│   ├── README.md                        # Este archivo
│   ├── RECURSOS_ML_ESTADISTICA.md       # Reporte de recursos
│   ├── INDICE_NOTEBOOKS.md             # Índice de notebooks
│   └── GUIA_CONFIGURACION.md            # Guía de configuración
└── ⚙️ Configuración/
    ├── requirements.txt                 # Dependencias Python
    ├── docker-compose.yml              # Configuración Docker
    └── Dockerfile                       # Imagen personalizada
```

## � Colección de Libros y Recursos

### 🎯 **Libros de Machine Learning**

#### 📖 **Introduction to Statistical Learning with Python (ISLP)**
- **Ubicación**: `notebooks/ISLP_labs/`
- **Descripción**: Laboratorios prácticos del famoso libro ISLR adaptado a Python
- **Nivel**: Principiante a Intermedio
- **Contenido**: 13 capítulos con ejercicios prácticos

#### 🧠 **Machine Learning with PyTorch and Scikit-Learn**
- **Ubicación**: `notebooks/Machine-Learning-with-Pytorch-Scikit-Learn/`
- **Descripción**: Implementaciones completas de algoritmos ML con PyTorch y Scikit-Learn
- **Nivel**: Intermedio a Avanzado
- **Contenido**: 11 capítulos con proyectos prácticos

#### 🔢 **Mathematics for Machine Learning**
- **Ubicación**: `notebooks/mml-book.github.io/`
- **Descripción**: Fundamentos matemáticos para ML
- **Nivel**: Intermedio
- **Contenido**: Álgebra lineal, cálculo, probabilidad y optimización

#### 📊 **Practical Statistics for Data Scientists**
- **Ubicación**: `notebooks/practical-statistics-for-data-scientists/`
- **Descripción**: Estadística aplicada para ciencia de datos
- **Nivel**: Principiante a Intermedio
- **Contenido**: 7 capítulos con casos prácticos

### 🏆 **Competencias y Proyectos**

#### 🚢 **Kaggle - Titanic Competition**
- **Ubicación**: `notebooks/kaggle_competitions/Titanic-Machine Learning from Disaster/`
- **Descripción**: Predicción de supervivencia en el Titanic
- **Técnicas**: Clasificación, feature engineering, ensemble methods

#### 🎓 **Curso UTN - Análisis de Datos Avanzado**
- **Ubicación**: `notebooks/UTN-elearning-analisis-datos-avanzado/`
- **Descripción**: Proyecto completo del curso UTN
- **Contenido**: 
  - **Unidad 5/TP5**: Predicción de churn de clientes Telco
  - **Proyecto TBC**: Análisis epidemiológico de tuberculosis
  - **Actividades por unidad**: Ejercicios progresivos

## �️ Tecnologías y Paquetes

### Core Data Science
- **pandas** - Manipulación de datos
- **numpy** - Computación numérica
- **matplotlib, seaborn, plotly** - Visualización
- **scipy, statsmodels** - Estadística y modelos

### Machine Learning
- **scikit-learn** - Algoritmos ML tradicionales
- **PyTorch** - Deep Learning
- **xgboost, lightgbm** - Gradient boosting

### Jupyter Ecosystem
- **jupyterlab** - Interfaz principal
- **ipywidgets** - Widgets interactivos

## 🔧 Configuración Docker

### Puertos
- **8888**: Jupyter Lab (principal)
- **8889**: Puerto alternativo (disponible)

### Volúmenes
- **Notebooks**: `/workspace/notebooks` (colección completa)
- **Proyectos**: `/workspace` (todo el repositorio)

### Variables de Entorno
- `JUPYTER_TOKEN=datascience2024`
- `JUPYTER_ROOT_DIR=/workspace`

## 🚨 Solución de Problemas

### Scripts de PowerShell
```powershell
# En PowerShell como administrador
Set-ExecutionPolicy RemoteSigned

# Para una sesión específica
Set-ExecutionPolicy Bypass -Scope Process -Force
.\quick-start.ps1
```

### Puerto ocupado
```bash
# Cambiar puerto en docker-compose.yml
ports:
  - "8801:8888"  # Usar puerto alternativo
```

## 📊 Casos de Uso Principales

### 📈 **Para Estudiantes de UTN**
- Material completo del curso "Análisis de Datos Avanzado"
- Proyectos paso a paso desde estadística básica hasta ML
- Datasets reales (TBC, Telco, familias)

### 🎯 **Para Aprendizaje Autodidacta**
- Progresión estructurada desde estadística hasta Deep Learning
- 4 libros completos con ejercicios prácticos
- Competencias de Kaggle para práctica

### 🏢 **Para Profesionales**
- Casos de uso empresariales (churn prediction, análisis epidemiológico)
- Código reutilizable y documentado
- Best practices en ciencia de datos

## 🤝 **Contribuciones**

Este repositorio está en constante evolución. Las contribuciones son bienvenidas:

1. **Fork** el repositorio
2. **Crea** una rama para tu feature (`git checkout -b feature/nueva-caracteristica`)
3. **Commit** tus cambios (`git commit -am 'Agregar nueva característica'`)
4. **Push** a la rama (`git push origin feature/nueva-caracteristica`)
5. **Abre** un Pull Request

## 📝 **Licencia**

Este proyecto está bajo la Licencia MIT. Ver el archivo `LICENSE` para más detalles.

## 📞 **Contacto**

Para preguntas sobre el curso UTN o este repositorio:
- **Universidad**: Universidad Tecnológica Nacional
- **Curso**: Análisis de Datos Avanzado
- **GitHub**: [cmessoftware](https://github.com/cmessoftware)

---

⭐ **¡Si este repositorio te fue útil, no olvides darle una estrella!** ⭐

