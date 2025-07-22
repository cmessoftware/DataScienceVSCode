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

## 📊 Casos de Uso

### Análisis de Datos TBC
Los notebooks en `UTN-elearning-analisis-datos-avanzado/notebooks/tbc/` contienen:
- Análisis exploratorio de datos TBC
- Visualizaciones epidemiológicas
- Modelos predictivos

### Ejercicios por Capítulo
- **Unidad1**: Análisis de familias
- **Unidad2**: Distribuciones de probabilidad
- **Unidad3**: Modelos binomiales e hipergeométricos
- **Unidad4**: Actividades prácticas con MPG dataset
- **Unidad5**: Predicciones y clasificaciones, introducción ML.

