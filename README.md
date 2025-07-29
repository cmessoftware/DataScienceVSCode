# 🎓 UTN - Análisis de Datos Avanzado y Machine Learning

[![Python](https://img.shields.io/badge/Python-3.11-blue.svg)](https://python.org)
[![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-orange.svg)](https://jupyter.org)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3-green.svg)](https://scikit-learn.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0-red.svg)](https://pytorch.org)

## 📋 Descripción

Repositorio completo para el curso de **Análisis de Datos Avanzado** de la Universidad Tecnológica Nacional (UTN), que incluye implementaciones prácticas, proyectos y una colección completa de libros de referencia en Machine Learning y Estadística.

## 🗂️ Estructura del Repositorio

```
📦 UTN-elearning-analisis-datos-avanzado/
├── 📁 notebooks/
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
├── 🐳 docker-helper.ps1               # Utilidades Docker
├── ⚙️ requirements.txt               # Dependencias Python
└── 🚀 start-jupyter.sh               # Script de inicio
```

## 📚 Libros y Recursos Incluidos

### 🤖 **Machine Learning**

#### 1. **ISLP Labs - Introduction to Statistical Learning with Python**
- 📖 **Autor**: Gareth James, Daniela Witten, Trevor Hastie, Robert Tibshirani
- 🎯 **Nivel**: Principiante a Intermedio
- 💻 **Contenido**:
  - ✅ Regresión Linear y Logística
  - ✅ Árboles de Decisión y Random Forest
  - ✅ Support Vector Machines
  - ✅ K-Means y Clustering
  - ✅ Cross-validation y Bootstrap
  - ✅ Análisis Discriminante (LDA/QDA)

#### 2. **Machine Learning with PyTorch and Scikit-Learn**
- 📖 **Autor**: Sebastian Raschka, Yuxi (Hayden) Liu, Vahid Mirjalili
- 🎯 **Nivel**: Intermedio a Avanzado
- 💻 **Contenido**:
  - ✅ Deep Learning con PyTorch
  - ✅ Redes Neuronales desde cero
  - ✅ Computer Vision y NLP
  - ✅ MLOps y despliegue de modelos

#### 3. **Mathematics for Machine Learning**
- 📖 **Autor**: Marc Peter Deisenroth, A. Aldo Faisal, Cheng Soon Ong
- 🎯 **Nivel**: Intermedio (Fundamentos matemáticos)
- 💻 **Contenido**:
  - ✅ Álgebra Linear
  - ✅ Cálculo y Optimización
  - ✅ Probabilidad y Estadística
  - ✅ Análisis de Componentes Principales

### 📊 **Estadística Aplicada**

#### 4. **Practical Statistics for Data Scientists**
- 📖 **Autor**: Peter Bruce, Andrew Bruce, Peter Gedeck
- 🎯 **Nivel**: Principiante a Intermedio
- 💻 **Contenido**:
  - ✅ Estadística descriptiva y exploratoria
  - ✅ Distribuciones de probabilidad
  - ✅ Inferencia estadística
  - ✅ Diseño experimental
  - ✅ Regresión avanzada

## 🚀 Inicio Rápido

### 1. **Clonar el repositorio**
```bash
git clone https://github.com/tu-usuario/UTN-elearning-analisis-datos-avanzado.git
cd UTN-elearning-analisis-datos-avanzado
```

### 2. **Configurar el entorno**

#### Opción A: Con Docker (Recomendado)
```bash
# Iniciar Jupyter con Docker
./start-jupyter.sh
```

#### Opción B: Con Python local
```bash
# Instalar dependencias
pip install -r requirements.txt

# Iniciar Jupyter
jupyter lab notebooks/
```

### 3. **Acceder a Jupyter**
- Abrir navegador en: `http://localhost:8888`
- Token se muestra en la terminal

## 📖 Guías de Estudio Recomendadas

### 🌱 **Para Principiantes**
1. **Comenzar con ISLP Labs** (Capítulos 1-4)
   - Fundamentos de estadística
   - Regresión linear básica
   
2. **Practical Statistics** (Capítulos 1-3)
   - Análisis exploratorio de datos
   - Conceptos estadísticos fundamentales

3. **Proyecto Titanic de Kaggle**
   - Aplicación práctica inmediata

### 🚀 **Para Nivel Intermedio**
1. **ISLP Labs** (Capítulos 5-10)
   - Validación cruzada
   - Regularización
   - Métodos no lineales

2. **Mathematics for ML** (Seleccionar capítulos relevantes)
   - Álgebra linear para ML
   - Optimización

3. **PyTorch & Scikit-Learn** (Capítulos iniciales)
   - Implementaciones avanzadas

### 🏆 **Para Nivel Avanzado**
1. **Machine Learning with PyTorch** (Completo)
   - Deep Learning
   - Redes neuronales especializadas
   
2. **Mathematics for ML** (Completo)
   - Fundamentos matemáticos profundos
   
3. **Proyectos Kaggle avanzados**
   - Competencias actuales

## 🛠️ Herramientas y Tecnologías

### 📦 **Librerías Principales**
- **pandas** - Manipulación de datos
- **numpy** - Computación numérica
- **scikit-learn** - Machine Learning tradicional
- **pytorch** - Deep Learning
- **matplotlib/seaborn** - Visualización
- **jupyter** - Notebooks interactivos

### 🐳 **Entorno de Desarrollo**
- **Docker** - Contenedorización
- **Jupyter Lab** - IDE interactivo
- **Git** - Control de versiones
- **Python 3.11** - Lenguaje base

## 🏆 Proyectos Destacados

### 1. **Predicción de Churn de Clientes Telco**
- 📂 `notebooks/UTN-elearning-analisis-datos-avanzado/Unidades/Unidad5/TP5/`
- 🎯 **Objetivo**: Predecir cancelación de clientes
- 🛠️ **Técnicas**: Random Forest, Logistic Regression, Naive Bayes
- 📊 **Métricas**: ROC-AUC, Precision, Recall

### 2. **Titanic: Machine Learning from Disaster**
- 📂 `notebooks/kaggle_competitions/Titanic-Machine Learning from Disaster/`
- 🎯 **Objetivo**: Predecir supervivencia en el Titanic
- 🛠️ **Técnicas**: Feature Engineering, Ensemble Methods
- 🏅 **Estado**: Implementación completa con análisis

## 📋 Lista de Verificación para Nuevos Usuarios

- [ ] ✅ Repositorio clonado
- [ ] 🐳 Docker instalado y funcionando
- [ ] 📚 Jupyter Lab accesible
- [ ] 🧪 Ejecutar notebook de prueba
- [ ] 📖 Revisar estructura de directorios
- [ ] 🎯 Elegir ruta de aprendizaje según nivel

## 🤝 Contribuciones

### Cómo contribuir:
1. **Fork** del repositorio
2. **Crear branch** para nueva funcionalidad
3. **Commit** de cambios con mensajes descriptivos
4. **Push** al branch
5. **Pull Request** con descripción detallada

### Tipos de contribuciones bienvenidas:
- 🐛 Corrección de bugs
- 📚 Nuevos notebooks educativos
- 📖 Mejoras en documentación
- 🚀 Optimizaciones de código
- 🎯 Nuevos datasets y proyectos

## 📧 Contacto y Soporte

- **Curso**: UTN - Análisis de Datos Avanzado
- **Instructor**: [Nombre del instructor]
- **Email**: [email@utn.edu.ar]
- **Issues**: [GitHub Issues](https://github.com/tu-usuario/repo/issues)

## 📜 Licencia

Este proyecto está bajo la licencia MIT. Ver `LICENSE` para más detalles.

## 🙏 Agradecimientos

- **UTN** por el marco educativo
- **Autores de libros** por los recursos excepcionales
- **Comunidad Open Source** por las herramientas
- **Kaggle** por los datasets y competencias

---

⭐ **¡No olvides dar estrella al repositorio si te resulta útil!** ⭐

*Última actualización: Julio 2025*
