# Practical Statistics for Data Scientists - Configuración Docker (Python)

Este proyecto está configurado para funcionar con Docker, proporcionando un entorno Python consistente y reproducible para análisis estadístico y ciencia de datos.

## 🐳 Configuración Docker

### Archivos de configuración:

- **`Dockerfile`**: Imagen con Python 3.11 y Jupyter Lab
- **`docker-compose.yml`**: Orquestación de servicios
- **`.dockerignore`**: Optimización de build
- **`docker-helper.ps1`**: Script de ayuda para Windows PowerShell
- **`requirements.txt`**: Dependencias de Python

## 🚀 Inicio Rápido

### En Windows (PowerShell):
```powershell
# Construir la imagen
.\docker-helper.ps1 build

# Iniciar Jupyter Lab
.\docker-helper.ps1 start
```

## 🌐 Acceso a Jupyter Lab

Una vez iniciado el servicio:

- **Jupyter Lab**: http://localhost:8888
  - Kernel de Python 3.11
  - Acceso a todos los notebooks del proyecto
  - Visualizaciones interactivas
  - Todas las librerías de ciencia de datos

## 📁 Estructura de volúmenes

Los siguientes directorios se montan automáticamente:
- `./data` → `/workspace/data` (datasets)
- `./python` → `/workspace/python` (código y notebooks Python)
- `.` → `/workspace` (proyecto completo)

## 🛠 Comandos útiles

```bash
# Ver estado del contenedor
docker-helper.ps1 status

# Ver logs en tiempo real
docker-helper.ps1 logs

# Abrir shell en el contenedor
docker-helper.ps1 shell

# Abrir consola Python interactiva
docker-helper.ps1 python

# Detener servicios
docker-helper.ps1 stop

# Limpiar sistema Docker
docker-helper.ps1 clean
```

## 📊 Paquetes incluidos

### Python:
- **Análisis de datos**: pandas, numpy
- **Visualización**: matplotlib, seaborn
- **Machine Learning**: scikit-learn, xgboost
- **Estadística**: scipy, statsmodels
- **Notebooks**: jupyter, jupyterlab
- **Otros**: wquantiles, pygam, dmba, pydotplus, imblearn, prince, adjustText

## 🔧 Personalización

Para agregar nuevos paquetes de Python:

1. Edita `requirements.txt`
2. Reconstruye la imagen: `docker-helper.ps1 build`

## 📝 Notas importantes

- Los cambios en código se reflejan inmediatamente (volúmenes montados)
- Para cambios en dependencias, necesitas reconstruir la imagen
- Los datos se persisten entre reinicios del contenedor
- Solo se incluyen archivos Python y datos (archivos R son ignorados)

## 🆘 Solución de problemas

Si encuentras problemas:

1. Verifica que Docker esté ejecutándose
2. Asegúrate de tener suficiente espacio en disco
3. Revisa los logs: `docker-helper.ps1 logs`
4. Reinicia los servicios: `docker-helper.ps1 restart`

¡Tu entorno de desarrollo Python para estadística está listo! 🐍📊
