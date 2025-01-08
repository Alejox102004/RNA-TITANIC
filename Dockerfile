# Usa una imagen base de Python 3.10
FROM python:3.10-slim

# Establece el directorio de trabajo
WORKDIR /app

# Copia los archivos del proyecto
COPY . /app

# Instala las dependencias
RUN python -m venv /opt/venv \
    && . /opt/venv/bin/activate \
    && pip install --upgrade pip \
    && pip install -r requirements.txt

# Expone el puerto
EXPOSE 8000

# Comando de inicio
CMD ["gunicorn", "app:app", "--bind", "0.0.0.0:8000"]
