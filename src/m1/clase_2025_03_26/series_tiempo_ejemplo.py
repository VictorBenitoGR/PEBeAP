# ./src/m1/clase_2_2025_03_26/series_tiempo_ejemplo.py

# *** Importaciones -----------------------------------------------------------

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.subplots as sp
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller, pacf, acf
from statsmodels.tsa.arima.model import ARIMA
import yfinance as yf
from datetime import datetime, timedelta


# *** Funciones ---------------------------------------------------------------

def plot_series(data, title="", yaxis_title="Valor", xaxis_title="Fecha"):
    """
    Crea un gráfico de línea usando plotly.

    Parameters
    ----------
    data : pandas.Series
        Serie temporal a graficar
    title : str
        Título del gráfico
    yaxis_title : str
        Título del eje Y
    xaxis_title : str
        Título del eje X

    Returns
    -------
    None
        Muestra el gráfico interactivo
    """
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=data.index, y=data.values, mode='lines'))
    fig.update_layout(
        title=title,
        yaxis_title=yaxis_title,
        xaxis_title=xaxis_title
    )
    fig.show()


def plot_decomposition(decomp_result):
    """
    Visualiza la descomposición de una serie temporal.

    Parameters
    ----------
    decomp_result : DecomposeResult
        Resultado de la descomposición de la serie

    Returns
    -------
    None
        Muestra el gráfico interactivo
    """
    fig = sp.make_subplots(rows=4, cols=1, subplot_titles=('Original', 'Tendencia', 'Estacional', 'Residual'))

    fig.add_trace(go.Scatter(y=decomp_result.observed, mode='lines', name='Original'), row=1, col=1)
    fig.add_trace(go.Scatter(y=decomp_result.trend, mode='lines', name='Tendencia'), row=2, col=1)
    fig.add_trace(go.Scatter(y=decomp_result.seasonal, mode='lines', name='Estacional'), row=3, col=1)
    fig.add_trace(go.Scatter(y=decomp_result.resid, mode='lines', name='Residual'), row=4, col=1)

    fig.update_layout(height=900, title_text="Descomposición de la Serie Temporal")
    fig.show()


# Obtener datos del DAX
dax = yf.download('^GDAXI', start='1991-01-01', end='1998-12-31')['Close']

# Paso 1: Visualizar la Serie
plot_series(dax, "DAX de 1991 a 1998", "DAX")

# Descomposición de la serie
decomposition = seasonal_decompose(dax, period=252)  # 252 días trading aproximadamente
plot_decomposition(decomposition)

# Paso 2: Estacionarizar la Serie de Tiempo


def adf_test(series):
    """
    Realiza la prueba Aumentada de Dickey-Fuller.

    Parameters
    ----------
    series : pandas.Series
        Serie temporal a analizar

    Returns
    -------
    dict
        Diccionario con los resultados de la prueba
    """
    result = adfuller(series.dropna())
    return {
        'Estadístico de prueba': result[0],
        'p-valor': result[1],
        'Valores críticos': result[4]
    }


# Prueba ADF en serie original
print("\nPrueba ADF en serie original:")
print(pd.DataFrame(adf_test(dax)))

# Primera diferencia
D1 = dax.diff().dropna()
print("\nPrueba ADF en primera diferencia:")
print(pd.DataFrame(adf_test(D1)))

# Paso 3: Identificar valores de p y q


def plot_acf_pacf(series):
    """
    Grafica las funciones ACF y PACF.

    Parameters
    ----------
    series : pandas.Series
        Serie temporal a analizar

    Returns
    -------
    None
        Muestra el gráfico interactivo
    """
    acf_vals = acf(series, nlags=40)
    pacf_vals = pacf(series, nlags=40)
    lags = range(len(acf_vals))

    fig = sp.make_subplots(rows=2, cols=1, subplot_titles=('ACF', 'PACF'))

    fig.add_trace(go.Scatter(x=lags, y=acf_vals, mode='markers+lines', name='ACF'), row=1, col=1)
    fig.add_trace(go.Scatter(x=lags, y=pacf_vals, mode='markers+lines', name='PACF'), row=2, col=1)

    fig.add_hline(y=1.96 / np.sqrt(len(series)), line_dash="dash", line_color="red", row=1, col=1)
    fig.add_hline(y=-1.96 / np.sqrt(len(series)), line_dash="dash", line_color="red", row=1, col=1)
    fig.add_hline(y=1.96 / np.sqrt(len(series)), line_dash="dash", line_color="red", row=2, col=1)
    fig.add_hline(y=-1.96 / np.sqrt(len(series)), line_dash="dash", line_color="red", row=2, col=1)

    fig.update_layout(height=600, title_text="Funciones ACF y PACF")
    fig.show()


plot_acf_pacf(D1)

# Paso 4: Ajustar el modelo ARIMA
modelo = ARIMA(dax, order=(5, 1, 5))
resultado = modelo.fit()
print("\nResumen del modelo ARIMA:")
print(resultado.summary())

# Paso 5: Predicciones
n_pred = 91  # Días hasta fin de 1998
forecast = resultado.forecast(steps=n_pred, alpha=0.1)  # Intervalo del 90%
conf_int = resultado.get_forecast(steps=n_pred).conf_int(alpha=0.1)

# Graficar predicciones
fig = go.Figure()

# Datos históricos
fig.add_trace(go.Scatter(x=dax.index, y=dax.values, name='Histórico', mode='lines'))

# Predicciones
fig.add_trace(go.Scatter(x=pd.date_range(start=dax.index[-1], periods=n_pred + 1)[1:],
                         y=forecast,
                         name='Predicción',
                         mode='lines',
                         line=dict(dash='dash')))

# Intervalos de confianza
fig.add_trace(go.Scatter(x=pd.date_range(start=dax.index[-1], periods=n_pred + 1)[1:],
                         y=conf_int.iloc[:, 0],
                         fill=None,
                         mode='lines',
                         line_color='rgba(255,0,0,0.2)',
                         name='Límite inferior 90%'))

fig.add_trace(go.Scatter(x=pd.date_range(start=dax.index[-1], periods=n_pred + 1)[1:],
                         y=conf_int.iloc[:, 1],
                         fill='tonexty',
                         mode='lines',
                         line_color='rgba(255,0,0,0.2)',
                         name='Límite superior 90%'))

fig.update_layout(title='Predicciones DAX 1998-1999',
                  yaxis_title='DAX',
                  xaxis_title='Fecha',
                  showlegend=True)
fig.show()
