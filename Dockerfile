# Usa una imagen base de Python (Python 3.10 recomendado)
FROM python:3.10-slim

# Establece el directorio de trabajo dentro del contenedor
WORKDIR /app

# Copia los archivos del proyecto al contenedor
COPY . /app

# Actualiza pip e instala las dependencias
RUN pip install --upgrade pip \
    && pip install -r requirements.txt

# Expone el puerto 8000 para la aplicación
EXPOSE 8000

# Comando para ejecutar la aplicación
CMD ["gunicorn", "app:app", "--bind", "0.0.0.0:8000"]
