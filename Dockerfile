FROM python:3.11

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y \
    git \
    curl \
    wget \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copia TODO el proyecto al contenedor
COPY . /app

# Instala dependencias de Python
RUN pip install --no-cache-dir -r requirements.txt

EXPOSE 8888

CMD ["bash"]
