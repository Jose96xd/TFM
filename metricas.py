import math
import numpy as np

def distancia_euclidia(ind1:list, ind2:list) ->float:
    """
    Función auxiliar que calcula la distancia euclídea que hay entre las métricas de dos individuos.
    Parámetros:
        - ind1: list. El conjunto de métricas del individuo 1.
        - ind2: list. El conjunto de métricas del individuo 2.
    Devuelve:
            - dist: float. La distancia euclídea entre ambos individuos.
    """
    dist = 0
    for dim in range(len(ind1)):
        dist += (ind1[dim] - ind2[dim])**2
    return math.sqrt(dist)

def get_vecinos_cercanos(frente: np.array, return_ind: bool=False) -> list:
    """
    Función auxiliar que devuelve una lista con los individuos más cercanos a cada individuo de un frente. Opcionalmente, puede devolver también el índice de dicho individuo más cercano en el frente.
    Parámetros:
        - frente: np.array. Numpy Array con los individuos del frente sobre el cual queremos operar.
        - return_ind: bool=False. Booleano que se utiliza para indicar si la función debe devolver el índice del vecino más cercano (True) o no (False).
    Devuelve:
        - vecinos_cercanos: list. Una lista con el vecino más cercano de cada individuo de un frente dado.
    """
    vecinos_cercanos = []

    for ind in frente:
        distancias = []
        for i, vecino in enumerate(frente):
            if not np.array_equal(ind, vecino):
                dist = distancia_euclidia(ind, vecino)
                distancias.append((i, dist))
        distancias.sort(key=lambda x: x[1])
        if return_ind:
            vecinos_cercanos.append( (frente[distancias[0][0]], distancias[0][1]) )
        else:
            vecinos_cercanos.append(distancias[0][1])
    return vecinos_cercanos

def get_schott_metric(frente: np.array, return_media:bool=False):
    """
    Función encargada de calcular la distancia de Schott para un frente dado.
    Parámetros:
        - frente: np.array. Numpy Array con los individuos del frente sobre el cual queremos operar.
        - return_media: bool=False. Booleano que sirve para indicar si la función debe devolver la media de las distancias más cortas entre individuos, además de la distancia de Schott.
    Devuelve:
        result: float. La distancia de Schott del frente dado.
    """
    result, media_distancias = 0, 1
    if frente.shape[0] >= 2:
        distancias_menores = get_vecinos_cercanos(frente)
        media_distancias = sum(distancias_menores) / (len(frente) - 1)

        result = math.sqrt( sum([(dist - media_distancias)**2 for dist in distancias_menores]) / (len(frente) - 1) )
    if return_media:
        return result, media_distancias
    return result

def get_spacing_metric(frente: np.array) -> float:
    """
    Función encargada de calcular la distancia de espaciado.
    Parámetros:
        - frente: np.array. Numpy Array con los individuos del frente para el cual queremos calcular la distancia de espaciado.
    Devuelve:
        - spacing_metric: float. La distancia de espaciado.
    """
    schott_metric, media_distancias = get_schott_metric(frente, return_media=True)
    spacing_metric = schott_metric / media_distancias
    return spacing_metric