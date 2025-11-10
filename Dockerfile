# Usar una imagen base oficial de Python
FROM python:3.11-slim

# Establecer el directorio de trabajo en el contenedor
WORKDIR /app

# Copiar el archivo de dependencias
COPY requirements.txt .

# Instalar las dependencias de Python
RUN pip install --no-cache-dir -r requirements.txt

# Copiar la carpeta mlops_pipeline al contenedor
COPY mlops_pipeline/ ./mlops_pipeline/

# Exponer el puerto 8000 para la API
EXPOSE 8000

# Comando para ejecutar la API con uvicorn
CMD ["uvicorn", "mlops_pipeline.src._model_deploy:app", "--host", "0.0.0.0", "--port", "8000"]
