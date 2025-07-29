# Dockerfile para Practical Statistics for Data Scientists
# Entorno Python con Jupyter Lab para análisis de datos

FROM python:3.11-slim

# Instalar dependencias del sistema
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    git \
    vim \
    nano \
    wget \
    graphviz \
    && rm -rf /var/lib/apt/lists/*

# Actualizar pip
RUN pip install --upgrade pip

# Instalar paquetes de Python
COPY requirements.txt /tmp/requirements.txt
RUN pip install -r /tmp/requirements.txt

# Instalar Jupyter
RUN pip install jupyter jupyterlab notebook

# Crear directorio de trabajo
WORKDIR /workspace

# Copiar todos los archivos del proyecto
COPY . /workspace/

# Exponer puertos para Jupyter
EXPOSE 8888

# Comando por defecto
CMD ["jupyter", "lab", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--allow-root", "--NotebookApp.token=''", "--NotebookApp.password=''"]
