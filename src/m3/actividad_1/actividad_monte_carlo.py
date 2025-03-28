# ./src/m3/actividad_1/actividad_food_delivery.py

# *** Importaciones -----------------------------------------------------------
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats

# *** Funciones -----------------------------------------------------------

def estimar_tiempo_entrega(distancia, prep_tiempo, clima, trafico):
    # Velocidad base: 20 km/h = 0.33 km/min
    velocidad_base = 0.33
    
    # Factores de ajuste por clima
    factores_clima = {
        'Clear': 1.0,
        'Windy': 1.1,
        'Foggy': 1.2,
        'Rainy': 1.3,
        'Snowy': 1.5
    }
    
    # Factores de ajuste por tráfico
    factores_trafico = {
        'Low': 1.0,
        'Medium': 1.2,
        'High': 1.5
    }
    
    # Aplicar factores (o valor por defecto si no existe la clave)
    factor_clima = factores_clima.get(clima, 1.2)
    factor_trafico = factores_trafico.get(trafico, 1.2)
    
    # Calcular tiempo base por distancia
    tiempo_trayecto = (distancia / velocidad_base) * factor_clima * factor_trafico
    
    # Tiempo total = tiempo preparación + tiempo trayecto + 5 min adicionales
    tiempo_total = prep_tiempo + tiempo_trayecto + 5
    
    return tiempo_total

def simular_factores_riesgo(df, num_simulaciones=10000):
    resultados = {
        'retraso_entrega': [],
        'entrega_dañada': [],
        'cliente_ausente': [],
        'error_direccion': []
    }
    
    distancias = []
    tiempos_prep = []
    experiencias = []
    
    for _ in range(num_simulaciones):
        # Seleccionar entrega aleatoria
        entrega = df.sample(n=1).iloc[0]
        
        # Datos de la entrega
        distancia = entrega['Distance_km']
        tiempo_prep = entrega['Preparation_Time_min']
        experiencia = entrega['Courier_Experience_yrs'] if not pd.isna(entrega['Courier_Experience_yrs']) else 1.0
        clima = entrega['Weather'] if not pd.isna(entrega['Weather']) else 'Clear'
        trafico = entrega['Traffic_Level'] if not pd.isna(entrega['Traffic_Level']) else 'Medium'
        vehiculo = entrega['Vehicle_Type'] if not pd.isna(entrega['Vehicle_Type']) else 'Scooter'
        
        distancias.append(distancia)
        tiempos_prep.append(tiempo_prep)
        experiencias.append(experiencia)
        
        # 1. Riesgo de retraso en la entrega
        tiempo_estimado = estimar_tiempo_entrega(distancia, tiempo_prep, clima, trafico)
        tiempo_real = entrega['Delivery_Time_min']
        
        hay_retraso = tiempo_real > (tiempo_estimado * 1.2)  # 20% por encima de lo estimado
        
        # Añadir aleatoriedad (5% de probabilidad de evento inesperado)
        if np.random.random() < 0.05:
            hay_retraso = not hay_retraso
            
        resultados['retraso_entrega'].append(hay_retraso)
        
        # 2. Riesgo de entrega dañada
        # Mayor probabilidad con clima adverso, distancias largas y poca experiencia
        factor_clima_daño = 0.05 if clima in ['Rainy', 'Snowy'] else 0.01
        factor_distancia = min(distancia / 50, 0.1)
        factor_experiencia = max(0, (2.0 - experiencia) / 20)
        
        prob_daño = factor_clima_daño + factor_distancia + factor_experiencia
        hay_daño = np.random.random() < prob_daño
        resultados['entrega_dañada'].append(hay_daño)
        
        # 3. Riesgo de cliente ausente
        # Mayor probabilidad en la noche
        prob_ausente = 0.02  # Probabilidad base
        if entrega['Time_of_Day'] == 'Night':
            prob_ausente += 0.03
        
        cliente_ausente = np.random.random() < prob_ausente
        resultados['cliente_ausente'].append(cliente_ausente)
        
        # 4. Riesgo de error en la dirección
        # Mayor probabilidad con poca experiencia
        prob_error = 0.01 + max(0, (3.0 - experiencia) / 30)
        error_direccion = np.random.random() < prob_error
        resultados['error_direccion'].append(error_direccion)
    
    # Calcular probabilidades
    probabilidades = {}
    for riesgo, valores in resultados.items():
        probabilidades[riesgo] = sum(valores) / len(valores)
    
    # Datos adicionales
    probabilidades['distancias'] = distancias
    probabilidades['tiempos_prep'] = tiempos_prep
    probabilidades['experiencias'] = experiencias
    probabilidades['num_simulaciones'] = num_simulaciones
    
    return probabilidades

# *** Ejecución ---------------------------------------------------------------

df = pd.read_csv('src/m3/actividad_1/Food_Delivery_Times.csv')
muestra = df.sample(n=700, random_state=42)
muestra = muestra.dropna(subset=['Distance_km', 'Preparation_Time_min', 'Delivery_Time_min'])

print(f"Registros utilizados: {len(muestra)}")
print(muestra[['Distance_km', 'Preparation_Time_min', 'Courier_Experience_yrs', 'Delivery_Time_min']].describe())

resultados = simular_factores_riesgo(muestra, num_simulaciones=10000)

print("\nProbabilidades de Riesgos:")
for riesgo, prob in resultados.items():
    if riesgo not in ['distancias', 'tiempos_prep', 'experiencias', 'num_simulaciones']:
        print(f"  • {riesgo}: {prob:.2%}")

# Visualización
plt.figure(figsize=(10, 6))
riesgos = [
    'retraso_entrega', 
    'entrega_dañada', 
    'cliente_ausente', 
    'error_direccion'
]
etiquetas = [
    'Retraso en la Entrega',
    'Entrega Dañada',
    'Cliente Ausente',
    'Error en Dirección'
]

valores = [resultados[r] * 100 for r in riesgos]
colores = ['#FF5733', '#33A8FF', '#2E8B57', '#9370DB']

plt.bar(etiquetas, valores, color=colores, width=0.6)
plt.ylabel('Probabilidad (%)')
plt.xticks(rotation=0)
for i, v in enumerate(valores):
    plt.text(i, v + 0.5, f"{v:.1f}%", ha='center', fontweight='bold')
plt.text(0.5, 0.9, 
        f"Simulación Monte Carlo (n={resultados['num_simulaciones']})\n" + 
        f"Basado en {len(muestra)} registros de entrega",
        transform=plt.gca().transAxes, fontsize=9, ha='center',
        bbox=dict(facecolor='white', alpha=0.8, boxstyle='round,pad=0.5'))
plt.title('Probabilidad de riesgos en entregas de comida')
plt.ylim(0, max(valores) * 1.2)
plt.tight_layout()
plt.savefig('riesgo_entregas_comida.png', dpi=300, bbox_inches='tight')
plt.show()


plt.figure(figsize=(10, 6))
plt.scatter(muestra['Distance_km'], muestra['Delivery_Time_min'], alpha=0.6, 
            c=muestra['Courier_Experience_yrs'].fillna(0), cmap='viridis')
plt.colorbar(label='Experiencia del repartidor (años)')
plt.xlabel('Distancia (km)')
plt.ylabel('Tiempo de entrega (min)')
plt.title('Relación entre distancia y tiempo de entrega')

z = np.polyfit(muestra['Distance_km'], muestra['Delivery_Time_min'], 1)
p = np.poly1d(z)
plt.plot(muestra['Distance_km'], p(muestra['Distance_km']), "r--", alpha=0.8)

plt.tight_layout()
plt.savefig('relacion_distancia_tiempo.png', dpi=300)
plt.show() 