FROM python:3.11-slim

ENV DEBIAN_FRONTEND=noninteractive

# Instalar dependencias del sistema
RUN apt-get update && apt-get install -y \
    git \
    curl \
    wget \
    build-essential \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

# Crear directorio de trabajo
WORKDIR /workspace

# Copiar requirements primero para mejor cache de Docker
COPY requirements.txt .

# Instalar dependencias de Python
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copiar el resto del proyecto
COPY . .

# Crear directorio para configuración de Jupyter
RUN mkdir -p /root/.jupyter

# Configurar Jupyter
RUN jupyter lab --generate-config

# Exponer puertos
EXPOSE 8881 8882

# Comando por defecto
CMD ["jupyter", "lab", "--ip=0.0.0.0", "--port=8881", "--no-browser", "--allow-root", "--notebook-dir=/workspace"]
