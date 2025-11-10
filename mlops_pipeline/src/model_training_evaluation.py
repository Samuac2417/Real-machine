"""
Script para entrenar el modelo de regresión y guardarlo.
"""

import numpy as np
import os
import joblib

from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error

from ft_engineering import load_data, preprocess_data, split_data, create_pipeline


if __name__ == "__main__":
    # Ruta del archivo de datos
    DATA_PATH = 'Base_de_datos.csv'

    # Rutas de guardado del modelo y pipeline
    PIPELINE_PATH = 'mlops_pipeline/models/preprocessor.joblib'
    MODEL_PATH = 'mlops_pipeline/models/model.joblib'

    # Crear la carpeta si no existe
    os.makedirs('mlops_pipeline/models', exist_ok=True)

    # Cargar los datos
    df = load_data(DATA_PATH)

    # Validar y procesar datos
    if df is not None:
        # Aplicar preprocesamiento
        df = preprocess_data(df)

        # Dividir datos en Train y Test
        X_train, X_test, y_train, y_test = split_data(df)

        # Crear el pipeline de preprocesamiento
        preprocessor = create_pipeline(X_train.columns)

        # Ajustar el pipeline con X_train
        preprocessor.fit(X_train)

        # Transformar los datos
        X_train_processed = preprocessor.transform(X_train)
        X_test_processed = preprocessor.transform(X_test)

        # Mensaje final
        print(f'Datos listos para el entrenamiento. Forma de X_train_processed: {X_train_processed.shape}')

        # --- Iniciando Entrenamiento del Modelo ---
        print('--- Iniciando Entrenamiento del Modelo ---')

        # Instanciar el modelo Ridge
        model = Ridge(alpha=1.0)

        # Ajustando el modelo
        print('Ajustando (fit) modelo Ridge con X_train_processed...')

        # Entrenar el modelo
        model.fit(X_train_processed, y_train)

        # Confirmación de entrenamiento exitoso
        print('¡Modelo entrenado exitosamente!')

        # --- Iniciando Evaluación del Modelo ---
        print('--- Iniciando Evaluación del Modelo ---')

        # Generar predicciones
        y_pred = model.predict(X_test_processed)

        # Mensaje informativo
        print('Calculando el Error Cuadrático Medio (RMSE)...')

        # Revertir transformación logarítmica
        y_test_orig = np.expm1(y_test)
        y_pred_orig = np.expm1(y_pred)

        # Calcular el Error Cuadrático Medio (MSE)
        mse = mean_squared_error(y_test_orig, y_pred_orig)

        # Calcular la Raíz del Error (RMSE)
        rmse = np.sqrt(mse)

        # Imprimir resultados finales
        print(f'¡Evaluación completada!')
        print(f'El Error (RMSE) del modelo en los datos de prueba es: ${rmse:,.2f}')

        # --- Guardando el Modelo y Pipeline ---
        print(f'Guardando el pipeline en: {PIPELINE_PATH}')
        joblib.dump(preprocessor, PIPELINE_PATH)

        print(f'Guardando el modelo en: {MODEL_PATH}')
        joblib.dump(model, MODEL_PATH)

        print('¡Modelo y pipeline guardados exitosamente!')
