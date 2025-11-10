"""
API de FastAPI para servir el modelo de predicción de precios de casas.

Esta API permite servir predicciones en tiempo real utilizando un modelo
entrenado de machine learning. Proporciona endpoints para verificar el estado
de la API y realizar predicciones sobre precios de propiedades.
"""

from fastapi import FastAPI
import uvicorn
import joblib
import pandas as pd
import numpy as np
import os
from pydantic import BaseModel, Field
from typing import List, Optional

# ==================== MODELOS PYDANTIC ====================

class HouseData(BaseModel):
    """Modelo de validación para datos de una casa individual."""
    model_config = {"populate_by_name": True}
    
    MSSubClass: int
    MSZoning: str
    LotFrontage: Optional[float] = None
    LotArea: int
    Street: str
    Alley: Optional[str] = None
    LotShape: str
    LandContour: str
    Utilities: str
    LotConfig: str
    LandSlope: str
    Neighborhood: str
    Condition1: str
    Condition2: str
    BldgType: str
    HouseStyle: str
    OverallQual: int
    OverallCond: int
    YearBuilt: int
    YearRemodAdd: int
    RoofStyle: str
    RoofMatl: str
    Exterior1st: str
    Exterior2nd: str
    MasVnrType: Optional[str] = None
    MasVnrArea: Optional[float] = None
    ExterQual: str
    ExterCond: str
    Foundation: str
    BsmtQual: Optional[str] = None
    BsmtCond: Optional[str] = None
    BsmtExposure: Optional[str] = None
    BsmtFinType1: Optional[str] = None
    BsmtFinSF1: Optional[int] = None
    BsmtFinType2: Optional[str] = None
    BsmtFinSF2: Optional[int] = None
    BsmtUnfSF: Optional[int] = None
    TotalBsmtSF: Optional[int] = None
    Heating: str
    HeatingQC: str
    CentralAir: str
    Electrical: Optional[str] = None
    FirstFlrSF: int = Field(alias='1stFlrSF')
    SecondFlrSF: int = Field(alias='2ndFlrSF')
    LowQualFinSF: int
    GrLivArea: int
    BsmtHalfBath: Optional[int] = None
    BsmtFullBath: Optional[int] = None
    FullBath: int
    HalfBath: int
    BedroomAbvGr: int
    KitchenAbvGr: int
    KitchenQual: str
    TotRmsAbvGrd: int
    Functional: str
    Fireplaces: int
    FireplaceQu: Optional[str] = None
    GarageType: Optional[str] = None
    GarageYrBlt: Optional[int] = None
    GarageFinish: Optional[str] = None
    GarageCars: Optional[int] = None
    GarageArea: Optional[int] = None
    GarageQual: Optional[str] = None
    GarageCond: Optional[str] = None
    PavedDrive: str
    WoodDeckSF: int
    OpenPorchSF: int
    EnclosedPorch: int
    ThirdSsnPorch: int = Field(alias='3SsnPorch')
    ScreenPorch: int
    PoolArea: int
    PoolQC: Optional[str] = None
    Fence: Optional[str] = None
    MiscFeature: Optional[str] = None
    MiscVal: int
    MoSold: int
    YrSold: int
    SaleType: str
    SaleCondition: str


class HouseDataBatch(BaseModel):
    """Modelo para procesar múltiples casas en una sola solicitud."""
    inputs: List[HouseData]


# ==================== INSTANCIA DE FASTAPI ====================

app = FastAPI()

# Variables globales para almacenar los modelos
PIPELINE = None
MODEL = None

# Cargar modelos
print('Cargando modelos...')

# Obtener la ruta base del proyecto
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MODELS_DIR = os.path.join(BASE_DIR, 'mlops_pipeline', 'models')

# Cargar el preprocessor
PIPELINE = joblib.load(os.path.join(MODELS_DIR, 'preprocessor.joblib'))
print('✓ Preprocessor cargado exitosamente')

# Cargar el modelo
MODEL = joblib.load(os.path.join(MODELS_DIR, 'model.joblib'))
print('✓ Modelo cargado exitosamente')


# Endpoint: Health Check
@app.get("/")
def healthcheck():
    """
    Endpoint de verificación de salud (Health Check).
    
    Retorna:
        dict: Estado de la API
    """
    return {"status": "API is running!"}


# Endpoint: Predicción de precios
@app.post("/predict")
def predict(data: HouseDataBatch):
    """
    Endpoint de predicción de precios de casas.
    
    Recibe datos de una o varias casas y retorna predicciones de precios.
    
    Args:
        data (HouseDataBatch): Lote de datos de casas a predecir
        
    Returns:
        dict: Diccionario con lista de predicciones de precios
    """
    # Convertir datos Pydantic a DataFrame
    df = pd.DataFrame([house.model_dump(by_alias=True) for house in data.inputs])
    
    # Aplicar el preprocessor (transformación)
    processed_data = PIPELINE.transform(df)
    
    # Realizar predicciones (en escala logarítmica)
    predictions_log = MODEL.predict(processed_data)
    
    # Revertir la transformación logarítmica
    predictions_orig = np.expm1(predictions_log)
    
    # Retornar predicciones
    return {"predictions": predictions_orig.tolist()}


if __name__ == "__main__":
    # Ejecutar la aplicación con uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
