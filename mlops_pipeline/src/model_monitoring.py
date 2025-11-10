"""
Dashboard de Streamlit para el monitoreo de Data Drift.

Este módulo implementa un dashboard interactivo que permite monitorear
el Data Drift entre los datos de entrenamiento (referencia) y los datos
de producción (nuevos datos). Utiliza el test de Kolmogorov-Smirnov para
detectar cambios estadísticamente significativos en las distribuciones
de las features, ayudando a identificar cuándo el modelo puede estar
degradándose debido a cambios en los datos de entrada.
"""

import streamlit as st
import pandas as pd
import numpy as np
from scipy.stats import ks_2samp
from ft_engineering import load_data
import plotly.graph_objects as go


def calculate_drift(reference, current):
    """
    Calcula el drift entre dos distribuciones usando el test de Kolmogorov-Smirnov.
    
    Args:
        reference: Serie de pandas con datos de referencia (históricos)
        current: Serie de pandas con datos actuales (producción)
    
    Returns:
        float: P-value del test K-S
    """
    statistic, p_value = ks_2samp(reference, current)
    return p_value


def plot_distribution(reference_series, current_series, column_name):
    """
    Crea un gráfico comparativo de distribuciones históricas vs actuales.
    
    Args:
        reference_series: Serie de pandas con datos de referencia
        current_series: Serie de pandas con datos actuales
        column_name: Nombre de la columna para el título
    
    Returns:
        go.Figure: Figura de Plotly con histogramas superpuestos
    """
    fig = go.Figure()
    
    # Histograma de datos de referencia (históricos)
    fig.add_trace(go.Histogram(
        x=reference_series,
        name='Datos Históricos',
        opacity=0.6,
        marker=dict(color='blue')
    ))
    
    # Histograma de datos actuales (producción)
    fig.add_trace(go.Histogram(
        x=current_series,
        name='Datos Actuales',
        opacity=0.6,
        marker=dict(color='red')
    ))
    
    # Configuración del layout
    fig.update_layout(
        barmode='overlay',
        title=f'Distribución Comparativa: {column_name}',
        xaxis_title=column_name,
        yaxis_title='Frecuencia',
        legend=dict(x=0.7, y=1)
    )
    
    return fig


# Título de la aplicación
st.title('Dashboard de Monitoreo de Data Drift')

# Carga de Datos de Referencia (Históricos)
DATA_PATH = 'Base_de_datos.csv'
reference_data = load_data(DATA_PATH)

# Simulación de Datos Actuales (Producción)
current_data = reference_data.sample(n=100, random_state=42)

# Sección de carga de datos
st.header('Carga de Datos')
st.success('Datos de referencia y actuales cargados exitosamente.')
st.text(f'Datos de Referencia (Históricos): {reference_data.shape}')
st.text(f'Datos Actuales (Simulados): {current_data.shape}')

# Definir columnas numéricas clave para monitorear
NUMERICAL_FEATURES = ['SalePrice', 'GrLivArea', 'LotArea', 'OverallQual', 'YearBuilt']

# Análisis de Drift
st.header('Análisis de Drift (Test K-S)')

# Selector de columna
selected_column = st.selectbox(
    'Selecciona la columna a analizar:',
    NUMERICAL_FEATURES
)

# Calcular drift para la columna seleccionada
p_value = calculate_drift(
    reference_data[selected_column],
    current_data[selected_column]
)

# Mostrar resultados
st.write(f'P-Value: {p_value:.4f}')

# Alerta de drift (Requisito de Rúbrica)
if p_value < 0.05:
    st.error('⚠️ Alerta: ¡Drift Detectado!')
else:
    st.success('✓ OK: No se detectó drift.')

# Gráfico comparativo de distribuciones
fig = plot_distribution(
    reference_data[selected_column],
    current_data[selected_column],
    selected_column
)
st.plotly_chart(fig, use_container_width=True)
