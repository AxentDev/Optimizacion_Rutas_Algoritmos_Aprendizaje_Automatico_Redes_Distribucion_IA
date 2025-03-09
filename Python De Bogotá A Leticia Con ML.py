# -*- coding: utf-8 -*-
"""
Created on Sun Jan 12 00:22:42 2025

@author: gran_
"""
# Importe de Bibliotecas
import heapq
import pickle
import numpy as np
from sklearn.linear_model import LinearRegression

# Definición de distancias
distancias = {
'Arauca': {'Arauca': 0,'Barranquilla': 900,'Bogotá': 735,'Bucaramanga': 460,'Cali': 1010,'Cúcuta': 370,'Ibagué': 890,'Medellín': 930,'Mocoa': 1200,'Neiva': 530,'Pasto': 1400,'Pereira': 1000,'Popayán': 1200,'Puerto Carreño': 800,'San José Del Guaviare': 700,'San Vicente Del Caguán': 950,'Tunja': 650,'Villavicencio': 500,'Leticia': 2155,},
'Barranquilla': {'Arauca': 900,'Barranquilla': 0,'Bogotá': 979,'Bucaramanga': 620,'Cali': 1200,'Cúcuta': 900,'Ibagué': 1100,'Medellín': 760,'Mocoa': 1600,'Neiva': 1200,'Pasto': 1500,'Pereira': 1050,'Popayán': 1300,'Puerto Carreño': 1200,'San José Del Guaviare': 1100,'San Vicente Del Caguán': 1200,'Tunja': 940,'Villavicencio': 1050,'Leticia': 2350,},
'Bogotá': {'Arauca': 735,'Barranquilla': 979,'Bogotá': 0,'Bucaramanga': 380,'Cali': 460,'Cúcuta': 750,'Ibagué': 200,'Medellín': 410,'Mocoa': 620,'Neiva': 410,'Pasto': 780,'Pereira': 300,'Popayán': 460,'Puerto Carreño': 705,'San José Del Guaviare': 400,'San Vicente Del Caguán': 655,'Tunja': 140,'Villavicencio': 115,'Leticia': 1857,},
'Bucaramanga': {'Arauca': 460,'Barranquilla': 620,'Bogotá': 380,'Bucaramanga': 0,'Cali': 720,'Cúcuta': 190,'Ibagué': 600,'Medellín': 400,'Mocoa': 870,'Neiva': 700,'Pasto': 1080,'Pereira': 580,'Popayán': 800,'Puerto Carreño': 750,'San José Del Guaviare': 630,'San Vicente Del Caguán': 770,'Tunja': 290,'Villavicencio': 500,'Leticia': 2015,},
'Cali': {'Arauca': 1010,'Barranquilla': 1200,'Bogotá': 460,'Bucaramanga': 720,'Cali': 0,'Cúcuta': 740,'Ibagué': 180,'Medellín': 450,'Mocoa': 1050,'Neiva': 390,'Pasto': 620,'Pereira': 225,'Popayán': 140,'Puerto Carreño': 1100,'San José Del Guaviare': 970,'San Vicente Del Caguán': 1100,'Tunja': 530,'Villavicencio': 750,'Leticia': 2100,},
'Cúcuta': {'Arauca': 370,'Barranquilla': 900,'Bogotá': 750,'Bucaramanga': 190,'Cali': 740,'Cúcuta': 0,'Ibagué': 650,'Medellín': 450,'Mocoa': 1000,'Neiva': 750,'Pasto': 1000,'Pereira': 700,'Popayán': 850,'Puerto Carreño': 450,'San José Del Guaviare': 900,'San Vicente Del Caguán': 1000,'Tunja': 630,'Villavicencio': 520,'Leticia': 2375,},
'Ibagué': {'Arauca': 890,'Barranquilla': 1100,'Bogotá': 200,'Bucaramanga': 600,'Cali': 180,'Cúcuta': 650,'Ibagué': 0,'Medellín': 230,'Mocoa': 650,'Neiva': 250,'Pasto': 570,'Pereira': 190,'Popayán': 280,'Puerto Carreño': 810,'San José Del Guaviare': 600,'San Vicente Del Caguán': 600,'Tunja': 180,'Villavicencio': 200,'Leticia': 1750,},
'Medellín': {'Arauca': 930,'Barranquilla': 760,'Bogotá': 410,'Bucaramanga': 400,'Cali': 450,'Cúcuta': 450,'Ibagué': 230,'Medellín': 0,'Mocoa': 740,'Neiva': 400,'Pasto': 820,'Pereira': 300,'Popayán': 530,'Puerto Carreño': 1200,'San José Del Guaviare': 980,'San Vicente Del Caguán': 1100,'Tunja': 320,'Villavicencio': 430,'Leticia': 2150,},
'Mocoa': {'Arauca': 1200,'Barranquilla': 1600,'Bogotá': 620,'Bucaramanga': 870,'Cali': 1050,'Cúcuta': 1000,'Ibagué': 650,'Medellín': 740,'Mocoa': 0,'Neiva': 320,'Pasto': 170,'Pereira': 700,'Popayán': 500,'Puerto Carreño': 1100,'San José Del Guaviare': 1000,'San Vicente Del Caguán': 1100,'Tunja': 840,'Villavicencio': 950,'Leticia': 1750,},
'Neiva': {'Arauca': 530,'Barranquilla': 1200,'Bogotá': 410,'Bucaramanga': 700,'Cali': 390,'Cúcuta': 750,'Ibagué': 250,'Medellín': 400,'Mocoa': 320,'Neiva': 0,'Pasto': 460,'Pereira': 540,'Popayán': 370,'Puerto Carreño': 900,'San José Del Guaviare': 590,'San Vicente Del Caguán': 250,'Tunja': 320,'Villavicencio': 270,'Leticia': 1350,},
'Pasto': {'Arauca': 1400,'Barranquilla': 1500,'Bogotá': 780,'Bucaramanga': 1080,'Cali': 620,'Cúcuta': 1000,'Ibagué': 570,'Medellín': 820,'Mocoa': 170,'Neiva': 460,'Pasto': 0,'Pereira': 750,'Popayán': 270,'Puerto Carreño': 1200,'San José Del Guaviare': 1100,'San Vicente Del Caguán': 1150,'Tunja': 830,'Villavicencio': 650,'Leticia': 1875,},
'Pereira': {'Arauca': 1000,'Barranquilla': 1050,'Bogotá': 300,'Bucaramanga': 580,'Cali': 225,'Cúcuta': 700,'Ibagué': 190,'Medellín': 300,'Mocoa': 700,'Neiva': 540,'Pasto': 750,'Pereira': 0,'Popayán': 230,'Puerto Carreño': 1050,'San José Del Guaviare': 900,'San Vicente Del Caguán': 1050,'Tunja': 370,'Villavicencio': 540,'Leticia': 1735,},
'Popayán': {'Arauca': 1200,'Barranquilla': 1300,'Bogotá': 460,'Bucaramanga': 800,'Cali': 140,'Cúcuta': 850,'Ibagué': 280,'Medellín': 530,'Mocoa': 500,'Neiva': 370,'Pasto': 270,'Pereira': 230,'Popayán': 0,'Puerto Carreño': 1150,'San José Del Guaviare': 1000,'San Vicente Del Caguán': 1150,'Tunja': 600,'Villavicencio': 850,'Leticia': 1595,},
'Puerto Carreño': {'Arauca': 800,'Barranquilla': 1200,'Bogotá': 705,'Bucaramanga': 750,'Cali': 1100,'Cúcuta': 450,'Ibagué': 810,'Medellín': 1200,'Mocoa': 1100,'Neiva': 900,'Pasto': 1200,'Pereira': 1050,'Popayán': 1150,'Puerto Carreño': 0,'San José Del Guaviare': 530,'San Vicente Del Caguán': 600,'Tunja': 1100,'Villavicencio': 570,'Leticia': 1225,},
'San José Del Guaviare': {'Arauca': 700,'Barranquilla': 1100,'Bogotá': 400,'Bucaramanga': 630,'Cali': 970,'Cúcuta': 900,'Ibagué': 600,'Medellín': 980,'Mocoa': 1000,'Neiva': 590,'Pasto': 1100,'Pereira': 900,'Popayán': 1000,'Puerto Carreño': 530,'San José Del Guaviare': 0,'San Vicente Del Caguán': 250,'Tunja': 750,'Villavicencio': 300,'Leticia': 1679,},
'San Vicente Del Caguán': {'Arauca': 950,'Barranquilla': 1200,'Bogotá': 655,'Bucaramanga': 770,'Cali': 1100,'Cúcuta': 1000,'Ibagué': 600,'Medellín': 1100,'Mocoa': 1100,'Neiva': 250,'Pasto': 1150,'Pereira': 1050,'Popayán': 1150,'Puerto Carreño': 600,'San José Del Guaviare': 250,'San Vicente Del Caguán': 0,'Tunja': 800,'Villavicencio': 650,'Leticia': 700,},
'Tunja': {'Arauca': 650,'Barranquilla': 940,'Bogotá': 140,'Bucaramanga': 290,'Cali': 530,'Cúcuta': 630,'Ibagué': 180,'Medellín': 320,'Mocoa': 840,'Neiva': 320,'Pasto': 830,'Pereira': 370,'Popayán': 600,'Puerto Carreño': 1100,'San José Del Guaviare': 750,'San Vicente Del Caguán': 800,'Tunja': 0,'Villavicencio': 270,'Leticia': 1750,},
'Villavicencio': {'Arauca': 500,'Barranquilla': 1050,'Bogotá': 115,'Bucaramanga': 500,'Cali': 750,'Cúcuta': 520,'Ibagué': 200,'Medellín': 430,'Mocoa': 950,'Neiva': 270,'Pasto': 650,'Pereira': 540,'Popayán': 850,'Puerto Carreño': 570,'San José Del Guaviare': 300,'San Vicente Del Caguán': 650,'Tunja': 270,'Villavicencio': 0,'Leticia': 2250,},
'Leticia': {'Arauca': 2155,'Barranquilla': 2350,'Bogotá': 1857,'Bucaramanga': 2015,'Cali': 2100,'Cúcuta': 2375,'Ibagué': 1750,'Medellín': 2150,'Mocoa': 1750,'Neiva': 1350,'Pasto': 1875,'Pereira': 1735,'Popayán': 1595,'Puerto Carreño': 1225,'San José Del Guaviare': 1679,'San Vicente Del Caguán': 700,'Tunja': 1750,'Villavicencio': 2250,'Leticia': 0,},
}

# Crear un mapeo de ciudades a índices
ciudades = list(distancias.keys())
ciudad_a_indice = {ciudad: idx for idx, ciudad in enumerate(ciudades)}

# Convertir las claves de distancias en índices numéricos
X = np.array([[ciudad_a_indice[key]] for key in distancias.keys()])  # Ciudades de origen
y = np.array([distancias[key]['Leticia'] for key in distancias.keys()])  # Distancias a Leticia

# Crear y entrenar el modelo
modelo = LinearRegression()
modelo.fit(X, y)

# Guardar el modelo
with open('modelo_regresion.pkl', 'wb') as f:
    pickle.dump(modelo, f)

# Función heurística
def calcular_heuristica(ciudad):
    return modelo.predict([[ciudad_a_indice[ciudad]]])[0]

# Búsqueda A*
def busqueda_a_star(inicio, objetivo):
    frontera = []
    heapq.heappush(frontera, (0, inicio, 0))  # (prioridad, ciudad, costo acumulado)
    came_from = {}
    costo_acumulado = {inicio: 0}

    while frontera:
        _, actual, costo_actual = heapq.heappop(frontera)

        if actual == objetivo:
            # Recuperar el camino
            camino = []
            while actual != inicio:
                camino.append(actual)
                actual = came_from[actual]
            camino.append(inicio)
            camino.reverse()
            return camino, costo_actual

        # Explorar vecinos
        for vecino, distancia in distancias[actual].items():
            nueva_distancia = costo_actual + distancia
            if vecino not in costo_acumulado or nueva_distancia < costo_acumulado[vecino]:
                costo_acumulado[vecino] = nueva_distancia
                prioridad = nueva_distancia + calcular_heuristica(vecino)
                heapq.heappush(frontera, (prioridad, vecino, nueva_distancia))
                came_from[vecino] = actual

    return None, None

# Ejemplo de uso
inicio = 'Cali'
objetivo = 'Leticia'

camino, distancia_total = busqueda_a_star(inicio, objetivo)
if camino:
    print(f"Camino encontrado: {camino}")
    print(f"Distancia total: {distancia_total} km")
else:
    print("No se encontró un camino")
