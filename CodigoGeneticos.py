import numpy as np
import pandas as pd
import random
import copy
from sklearn.cluster import KMeans

import matplotlib.pyplot as plt

import TestFunctions

def traducir_stands_a_indices(solucion_stands: list[str], stands: pd.DataFrame) -> list:
    """
    Función encargada de, dada una solución con la asignación de stands de cada vuelo codificado, traducir los stands a los índices que ocupan en la lista de stands.
    Parámetros:
        - solucion_stands: list[str]. Lista con los stands asignados a cada vuelo.
        - stands: pd:DataFrame. DataFrame con la información de todos los stands.
    Devuelve:
        - soluciones_indices: list. Lista con los índices de los stands asignados a cada vuelo.
    """
    solucion_indices = []
    for stand in solucion_stands:
        indice_stand = stands[stands.stands == stand].index[0]
        solucion_indices.append(indice_stand)

    solucion_indices = np.array(solucion_indices)
    return solucion_indices

def traducir_indices_a_stands(solucion_indices: list[int], stands: pd.DataFrame) -> list:
    """
    Función encargada de, dada una solución con los índices de cada stand asignado a cada vuelo, traducir los índices a los stands correspondientes.
    Parámetros:
        - solucion_indices: list[int]. Lista con los índices de los stands asignados a cada vuelo.
        - stands: pd:DataFrame. DataFrame con la información de todos los stands.
    Devuelve:
        - soluciones_stands: list. Lista con los stands asignados a cada vuelo.
    """
    solucion_stands = []
    for indice in solucion_indices:
        stand = stands.loc[indice].stands
        solucion_stands.append(stand)
    return solucion_stands

def plot_frente_pareto(frente_pareto: list[tuple], titutlo: str="Frente de Pareto", eje_X: str="Tiempo", eje_Y: str="Fuel"):
    """
    Función que plotea el frente de pareto dado.
    Parámetros:
      - frente_pareto: list[tuple]. Lista de tuplas con las puntuaciones de fitness de cada solución de la población.
      - eje_X: str="Tiempo". Nombre del eje X.
      - eje_Y: str="Fuel". Nombre del eje Y.
      - titulo: str. El título que tendrá el gráfico generado.
    """
    x_values = [fitness[0] for fitness in frente_pareto]
    y_values = [fitness[1] for fitness in frente_pareto]

    plt.scatter(x_values, y_values, label='Soluciones', color='blue', s=100)
    plt.xlabel(eje_X)
    plt.ylabel(eje_Y)
    plt.title(titutlo)
    plt.legend()
    plt.show()

def plot_historico(historico: list, plotear_medias: bool=True, metrica_a_plotear: int=None, nombres_metricas: list[str]=["Tiempo", "Fuel"]):
    """
    Función que plotea la evolución de las métricas a lo largo de la ejecución del algoritmo genético.
    Parámetros:
        - historico: list. El historico de la ejecución del algoritmo genético.
        - plotear_medias: bool=True. Booleano que se usa para indicar si se quiere plotear las medias a lo largo de las generaciones (True) o los mejores individuos de cada generación (False).
        - metrica_a_plotear: int=None. Entero que se usa para indicar qué métrica se plotea. Posibles valores: None -> Se plotean las dos métricas, 0 -> la primera, 1 -> la segunda.
        - nombres_metricas: list[str]. Lista con los nombres de las métricas a plotear.
    """
    a_plotear = historico["medias"]
    if not plotear_medias:
        a_plotear = historico["fit"]
        a_plotear = [elem[0] for elem in a_plotear]
    metrica0 = [elem[0] for elem in a_plotear]
    metrica1 = [elem[1] for elem in a_plotear]
    generaciones = [int(i) for i in range(1, len(metrica0) + 1)]

    if metrica_a_plotear is None or metrica_a_plotear == 0:
        plt.plot(generaciones, metrica0, "--ro")
        plt.ylabel(nombres_metricas[0])
        plt.xlabel("Generación")
        plt.show()
    if metrica_a_plotear is None or metrica_a_plotear == 1:
        plt.plot(generaciones, metrica1, "--bo")
        plt.ylabel(nombres_metricas[1])
        plt.xlabel("Generación")
        plt.show()

class GeneticoGenerico():
    """Clase que contiene las funcionalidades básicas de los algoritmos genéticos a implementar."""

    POSIBLES_METRICAS = {"Tiempo": "total_time_est",
                         "Stop": "duration_stop",
                         "Fuel": "total_fuel",
                         "Gases": "Gases",
                         }

    def __init__(self, fiteador:TestFunctions.TiempoAeropuerto, N: int=100,
                 pcross: float=0.8, pmut: float=0.3, psel: float=0.2, T: int=2, elitismo: int=0,
                 metricas: list[str]=["Tiempo", "Fuel"], pmut_minima: float=None, num_gen_pmut_minima: int=0):
        self.fiteador = fiteador
        self.N = N
        self.pcross = pcross
        self.pmut = pmut
        self.psel = psel
        self.T = T
        self.elitismo = elitismo
        self.metricas = metricas
        self.pmut_minima = pmut_minima if pmut_minima is not None else pmut
        self.num_gen_pmut_minima = num_gen_pmut_minima
        aux = [self.POSIBLES_METRICAS[metrica] for metrica in self.metricas]
        self.fiteador.metricas_a_devolver = aux

    @classmethod
    def get_puntos_corte(cls, num_puntos: int, len_ind: int) -> list[int]:
        """Función auxiliar utilizada en el cruce por puntos de corte.
        Parámetros: 
            - num_puntos: int. El número de puntos de corte que se desea.
            - len_ind: int. La longitud de los individuos que se quieren cruzar.
        Devuelve:
            - puntos_corte: list. Una lista con los puntos de corte.
        """
        puntos_corte = []

        while len(puntos_corte) < num_puntos:
            punto_corte = random.randint(0, len_ind)
            if punto_corte not in puntos_corte:
                puntos_corte.append(punto_corte)
        return sorted(puntos_corte)
    
    @classmethod
    def es_dominado(cls, fitness1: tuple, fitness2: tuple) -> bool:
        """Función encargada de comprobar si, dados dos fitness de dos individuos, uno de los individuos domina o no al otro.
        Parámetros: 
            - fitness1: tuple. La tupla con los fitness del individuo que queremos ver si es o no dominado.
            - fitness2: tuple. La tupla con los fitness de otro individuo, el cual queremos ver si domina o no al individuo 1.
        Devuelve:
            - Un booleano que es True si el individuo 1 es dominado por el 2 o False si no lo es.
        """
        return all(f2 <= f1 for f1, f2 in zip(fitness1, fitness2)) and any(f2 < f1 for f1, f2 in zip(fitness1, fitness2))
    
    @classmethod
    def get_rango_individuos(cls, fitness_individuos: list) -> list:
        """
        Función que devuelve el rango de los individuos en función de sus fitness y del número de individuos que los dominan.
        Parámetros: 
            - fitness_individuos: list. Una lista con las tuplas de los fitness de los individuos cuyos rangos queremos calcular.
        Devuelve:
            - rango_individuos: list. Una lista con el rango de cada individuo.
        """
        rango_individuos = []
        for fit1 in fitness_individuos:
            rango = 1

            for fit2 in fitness_individuos:
                if cls.es_dominado(fit1, fit2):
                    rango += 1
            rango_individuos.append(rango)

        return rango_individuos

    @classmethod
    def get_frentes(cls, fitness_pop: list, return_with_index: bool=True) -> list[list]:
        """
        Función que se encarga de calcular a qué frente de pareto pertenece cada individuo en función de sus fitness.
        Parámetros:
            - fitness_pop: list. Las tuplas con los fitness de cada individuo.
            - return_with_index: bool=True. Un booleno que se usa para indicar si se quiere que, además de los frentes, 
             se devuelva el índice de los individuos de cada uno de los frentes.
        Devuelve:
            - frentes: list[list]. Una lista con todos los frentes. Si return_with_index == True esas listas en vez de los individuos contienen
            tuplas de la forma (índice, individuo) siendo ese índice la posición original del individuo en la población.
        """
        frentes = []
        fitness_restantes = copy.deepcopy(fitness_pop)
        fitness_restantes = [(i, fit_restante) for i, fit_restante in enumerate (fitness_restantes)]

        while len(fitness_restantes) > 0:
            frentes.append([]) # Añadimos un nuevo frente

            for indice_fit, fit1 in fitness_restantes:
                no_dominado = True
                i = 0
                while i < len(fitness_restantes) and no_dominado:
                    _, fit2 = fitness_restantes[i]
                    if cls.es_dominado(fit1, fit2):
                        no_dominado = False
                    i += 1
                if no_dominado:
                    aux = (indice_fit, fit1)
                    frentes[-1].append(aux)
            fitness_restantes = [ind for ind in fitness_restantes if ind not in frentes[-1]]

        if return_with_index:
            return frentes
        else:
            return [[ind[1] for ind in frente] for frente in frentes]

    @classmethod
    def distancia_crowding(cls, frente: list, fitness_frente: list):
        """
        Función que, dado un frente de pareto y sus fitness, ordena el frente en función de la distancia de crowding de los individuos del mismo.
        Parámetros:
            - frente: list. El frente de pareto que queremos ordenar en función de la distancia de crowding.
            - fitness_frente: list. Los fitness de los individuos del frente.
        Devuelve:
            - individuos_crowding: list. Una lista con diccionarios que contienen, para cada individuo, el individuo, su fitness y su distancia de crowding. 
            Esta lista está ordenada en función de la distancia de crowding.
        """
        individuos_crowding = [{"ind":frente[i], "met":fitness_frente[i], "dist":0.0} for i in range(len(frente))]
        for metrica in range(len(fitness_frente[0])):
            individuos_crowding = sorted(individuos_crowding, key=lambda x: x["met"][metrica])
            max_metrica = individuos_crowding[-1]["met"][metrica]
            min_metrica = individuos_crowding[0]["met"][metrica]
            individuos_crowding[0]["dist"] = float("inf")
            individuos_crowding[-1]["dist"] = float("inf")

            for ind in range(1, len(individuos_crowding) - 1):
                termino_arriba = individuos_crowding[ind + 1]["met"][metrica] - individuos_crowding[ind - 1]["met"][metrica]
                individuos_crowding[ind]["dist"] += termino_arriba / ((max_metrica - min_metrica) + 1e-5)
        return sorted(individuos_crowding, key=lambda x: x["dist"], reverse=True)

    @classmethod
    def get_frente_pareto(cls, pop: list, fitness_pop: list) -> tuple[list]:
        """
        Función que, dada una población y su fitness, devuelve los individuos del frente de pareto y sus fitness.
        Parámetros:
            - pop: list. La población de la cual queremos extraer el frente.
            - fitness_pop: list. El fitness de la población.
        Devuelve:
            - (ind_frente, fit_frente): tuple[list]. Una tupla con los individuos del frente de pareto y sus fitness.
        """
        rangos = cls.get_rango_individuos(fitness_pop)
        ind_frente =[]
        fit_frente = []

        for i in range(len(rangos)):
            if rangos[i] == 1:
                ind_frente.append(pop[i])
                print(fitness_pop[i])
                fit_frente.append([fitness_pop[i]])
        return ind_frente, fit_frente

    def create_random_ind(self) -> list:
        """Función que genera un individuo aleatorio."""
        return self.fiteador.random_solution()
    
    def create_pop(self, true_random: bool=True) -> list[list]:
        """
        Función que genera una población de self.num_pop individuos aleatorios.
        Parámetros:
            - true_random: bool=True. Booleano que se utiliza para indicar si la población se creará de manera completamente aleatoria o corrigiendo los individuos.
        Devuelve:
            - pop: list[list]. Una lista con los individuos de la población.
        """
        pop = [list(self.create_random_ind()) for _ in range(self.N)]
        if not true_random:
            for i, ind in enumerate(pop):
                pop[i] = self.fiteador.check_bounds(ind)
        return pop

    def sort_pop(self, pop: list[list], metrica_ordenacion: list) -> tuple[list]:
        """
        Función encargada de, dada una población y sus métricas, ordenar la población en función de dichas métricas.
        Parámetros:
            - pop: list[list]. La población a ordenar.
            - metrica_ordenacion: list. Las métricas con las que ordenar la población.
        Devuelve:
            - (poblacion_ordenada, metrica_ordenacion): Una tupla con las métricas y la población ordenada en función de las métricas.
        """
        aux = list(zip(pop, metrica_ordenacion))

        poblacion_ordenada = sorted(aux, key=lambda x: x[1] )
        poblacion_ordenada = [ind[0] for ind in poblacion_ordenada]
        return (poblacion_ordenada, sorted(metrica_ordenacion))

    def mask_crossover(self, ind1: list, ind2: list) -> tuple[list]:
        """
        Función encargada de realizar el cruce por máscara de dos individuos.
        Parámetros:
            - ind1: list. Progenitor 1.
            - ind2: list. Progenitor 2.
        Devuelve:
            - hijo1: list, hijo2: list. Dos nuevos individuos fruto del cruce de los dos progenitores.
        """
        hijo1 = copy.deepcopy(ind1)
        hijo2 = copy.deepcopy(ind2)
        
        if random.random() < self.pcross:
            mascara = [random.random() >= (1 - self.pcross) for _ in range(len(ind1))]
            
            for i in range(len(hijo1)):
                hijo1[i] = ind1[i] if mascara[i] else ind2[i]
                hijo2[i] = ind2[i] if mascara[i] else ind1[i]

        return hijo1, hijo2
    
    def crossover_puntos (self, ind1: list, ind2: list) -> tuple[list]:
        """
        Función encargad del cruce por puntos de corte.
        Parámetros:
            - ind1: list. Progenitor 1.
            - ind2: list. Progenitor 2.
        Devuelve:
            - hijo1: list, hijo2: list. Dos nuevos individuos fruto del cruce de los dos progenitores.
        """
        hijo1 = copy.deepcopy(ind1)
        hijo2 = copy.deepcopy(ind2)

        if random.random() < self.pcross:
            num_puntos_corte = 1
            puntos_corte = GeneticoGenerico.get_puntos_corte(num_puntos_corte, len(ind1))
            cambio_padre = True

            for i in range(len(hijo1)):
                hijo1[i] = ind1[i] if cambio_padre else ind2[i]
                hijo2[i] = ind2[i] if cambio_padre else ind1[i]
                if i in puntos_corte:
                    cambio_padre = not cambio_padre

        return hijo1, hijo2

    def selectTorneo(self, pop: list[list]) -> list:
        """
        Función que realiza la selección por Torneo. Asume que los individuos ya están ordenados por su fitness.
        Parámetros:
            - pop: list. La población sobre la que se quiere realizar la selección.
        Devuelve:
            - Un individuo seleccionado.
        """
        elegidos = [random.choice(range(len(pop))) for _ in range(self.T)]
        seleccionado = min(elegidos)
        return copy.deepcopy(pop[seleccionado])
    
    def selectRango(self, pop: list[list]) -> list:
        """
        Función que realiza la selección por Rango. Asume que los individuos ya están ordenados por su fitness.
        Parámetros:
            - pop: list. La población sobre la que se quiere realizar la selección.
        Devuelve:
            - Un individuo seleccionado.
        """
        for ind in pop:
            if random.random() < self.psel:
                return copy.deepcopy(ind)
        return copy.deepcopy(pop[-1])

    def mutate_random(self, ind: list) -> list:
        """
        Función que realiza la mutación aleatoria sobre un individuo asignando al azar nuevos genes.
        Parámetros:
            - ind: list. Individuo a mutar.
        Devuelve:
            - ind_mutado: list. Individuo sobre el cual se han realizado las mutaciones.
        """
        ind_mutado = copy.deepcopy(ind)
        if random.random() < self.pmut:
            for indice in range(len(ind)):
                if random.random() < self.pmut:
                    ind_mutado[indice] = random.randrange(self.fiteador.bounds[0], self.fiteador.bounds[1] + 1)
        return ind_mutado
    
    def mutate_swap(self, ind: list) -> list:
        """
        Función que realiza la mutación mediante intercambio de posiciones de genes.
        Parámetros:
            - ind: list. Individuo a mutar.
        Devuelve:
            - ind_mutado: list. Individuo sobre el cual se han realizado las mutaciones.
        """
        ind_mutado = copy.deepcopy(ind)
        if random.random() < self.pmut:
            for indice in range(len(ind)):
                if random.random() < self.pmut:
                    indice_a_intercambiar = random.randrange(0, len(ind))
                    aux = ind[indice_a_intercambiar]
                    ind[indice_a_intercambiar] = ind[indice]
                    ind[indice] = aux
        return ind_mutado
     
    def fit(self, pop: list) -> list:
        """
        Función encargada de, dada una población, calcular el fitness de sus individuos y devolverlo.
        Parámetros: 
            - pop: list. La lista con los individuos de la población a evaluar.
        Devuelve:
            - fitness: list. Una lista de tuplas con los fitness de los individuos de la población.
        """
        fitness = []
        for solucion in pop: 
            metrica1, metrica2 = self.fiteador.objetive(solucion, return_total=True)
            fitness.append((metrica1, metrica2))
        if self.num_fiteos is not None:
            self.num_fiteos -= len(pop)
        return fitness
    
    def crea_siguiente_gen(self, pop: list, selTorneo: bool=True, cruceMascara: bool=True) -> tuple:
        """
        Función encargada de, dada una población, seleccionar, cruzar y mutar individuos para crear la siguiente generación.
        Parámetros:
            - pop: list: La población a partir de la cual queremos crear la siguiente generación.
            - selTorneo: bool=True. Booleano que se usa para indiciar si la selección se hace mediante Torneo (True) o por Rango (False).
            - cruceMascara: bool=True. Booleano que se usara para indicar si el cruce se hace por máscara (True) o por corte de puntos (False).
        Devuelve:
            - (pop_aux, fitness_aux): tuple. Una tupla con la nueva generación y su fitness.
        """
        pop_aux = []

        if self.elitismo > 0:
            pop_aux.extend(pop[:self.elitismo])

        while len(pop_aux) < self.N:
            if selTorneo: # Selección
                padre1 = self.selectTorneo(pop)
                padre2 = self.selectTorneo(pop)
            else:
                padre1 = self.selectRango(pop, self.psel)
                padre2 = self.selectRango(pop, self.psel)
            
            if cruceMascara: # Cruce
                hijo1, hijo2 = self.mask_crossover(padre1, padre2)
            else:
                hijo1, hijo2, self.crossover_puntos(padre1, padre2)

            # Mutación
            hijo1 = self.mutate_random(hijo1)
            hijo2 = self.mutate_random(hijo2)

            pop_aux.extend([hijo1, hijo2])

        for i, ind in enumerate(pop_aux):
            pop_aux[i] = self.fiteador.check_bounds(ind)

        fitness_aux = self.fit(pop_aux)
        return pop_aux, fitness_aux

    def evolve(self, num_gens: int=100, num_fiteos:int = 0, verbose: int=0) -> tuple:
        """
        Función abstracta que se encarga de la ejecución del propio algoritmo genético.
        Parámetros:
            - num_gens: int=100. El número de generaciones que se quiere ejecutar el algoritmo. Si se proporciona num_gens, num_fiteos debe ser 0.
            - num_fiteos: int=0. El número de evaluaciones que se le otorga al algoritmo como presupuesto. Una vez excedido dicho presupuesto el algoritmo termina la generación en la que se encuentre y finaliza.
            Si se le proporciona num_fiteos, num_gens debe ser 0. 
            - verbose: int=0. Entero que se utiliza para indicar el nivel de verbosidad del algoritmo.
        Devuelve:
            - (pop, fitness_pop): tuple. Una tupla con la poblaación final y sus fitness.
        """
        self.num_gens = num_gens
        self.num_fiteos = num_fiteos
        self.verbose = verbose
        self.historico = {"pop": [], "fit": [], "medias": [], "mejor": []}

        if self.num_gens > 0:
            self.pmut_decrecimiento = (self.pmut - self.pmut_minima) / (self.num_gen_pmut_minima if self.num_gen_pmut_minima > 0 else self.num_gens)
        if self.num_fiteos > 0:
            self.pmut_decrecimiento = ((self.pmut - self.pmut_minima) / (self.num_gen_pmut_minima if self.num_gen_pmut_minima > 0 else self.num_fiteos)) * self.N
        pass

    def actualizar_historico(self, pop: list, fitness_pop: list):
        """
        Función encargada de actualizar el histórico almacenando la progresión del algoritmo genético tras cada generación.
        Parámetros:
            - pop: list. La población de la generación a almacenar.
            - fitness_pop: list. El fitness de la población de la generación a almacenar.
        """
        medias = tuple(0 for _ in range(len(fitness_pop[0])))
        
        # Medias de las métricas
        for fit_ind in fitness_pop:
            medias = tuple(map(sum, zip(medias, fit_ind)))

        medias = tuple(elem/len(fitness_pop) for elem in medias)

        self.historico["pop"].append(pop)
        self.historico["fit"].append(fitness_pop)
        self.historico["medias"].append(medias)

    def print_info(self, fitness_pop: list, generacion: int):
        """
        Función encargada de printear información relativa al avance del algoritmo evolutivo.
        Parámetros:
            - fitness_pop: list. El fitness de la generación cuya información se quiere printear.
            - generación: int. La generación actual.
        """
        if self.verbose >= 1:
            if generacion == self.num_gens + 1 and self.num_gens > 0:
                print(f"Generación {generacion} (resultados finales).")
            if self.num_gens > 0:
                print(f"Generación {generacion} de {self.num_gens}:")
            else:
                print(f"Generación {generacion}:")
            
            tiempo = self.historico["medias"][-1][0]
            print(f"{self.metricas[0]} total medio de la generación: {tiempo}")
            otro = self.historico["medias"][-1][1]
            print(f"{self.metricas[1]} total medio de la generación: {otro}")
        if self.verbose >= 2:
            print(f"{self.metricas[0]} del mejor individuo de la generación: {fitness_pop[0][0]}")
            print(f"{self.metricas[1]} del mejor individuo de la generación: {fitness_pop[0][1]}")
        if self.verbose >= 3:
            print(f"Pmut actual: {self.pmut}")

    def plot_poblacion(self, fitness_pop: list=None, titutlo: str="Población"):
        """
        Función que plotea los fitness de una población.
        Parámetros:
            - fitness_pop: list. La población a plotear. Si se le pasa None, plotea la última generación por defecto.
            - titulo: str. El título que tendrá el gráfico generado.
        """
        if fitness_pop is None:
            fitness_pop = self.historico["fit"][-1]

        x_values = [fitness[0] for fitness in fitness_pop]
        y_values = [fitness[1] for fitness in fitness_pop]

        plt.scatter(x_values, y_values, label='Soluciones', color='blue', s=100)
        plt.xlabel(self.metricas[0])
        plt.ylabel(self.metricas[1])
        plt.title(titutlo)
        plt.legend()
        plt.show()

    def plot_frente_pareto(self, titulo: str="Frente de Pareto", verbose: int=0):
        """
        Función que plotea el frente de pareto de la última generación.
        Parámetros:
            - titulo: str. Título del plot a generar.
            - verbose: int. Entero que se usa para determinar el nivel de verborrea de la función. 
            Verbose=0: No se printea nada. Verbose=1: Se printea el número de elementos en el frente. 
        """
        frente = self.get_frentes(self.historico["fit"][-1], return_with_index=False)[0]
        if verbose > 0:
            print(len(frente))
        self.plot_poblacion(frente, titutlo=titulo)

    def plot_historico(self, plotear_medias: bool=True, metrica_a_plotear: int=None,):
        """
        Función que plotea la evolución de las métricas a lo largo de la ejecución del algoritmo genético.
        Parámetros:
            - plotear_medias: bool=True. Booleano que se usa para indicar si se quiere plotear las medias a lo largo de las generaciones (True) o los mejores individuos de cada generación (False).
            - metrica_a_plotear: int=None. Entero que se usa para indicar que métrica se plotea. Posibles valores: None -> Se plotean las dos métricas, 0 -> la primera, 1 -> la segunda.
        """
        a_plotear = self.historico["medias"]
        if not plotear_medias:
            a_plotear = self.historico["fit"]
            a_plotear = [elem[0] for elem in a_plotear]
        metrica0 = [elem[0] for elem in a_plotear]
        metrica1 = [elem[1] for elem in a_plotear]
        generaciones = [int(i) for i in range(1, len(metrica0) + 1)]

        if metrica_a_plotear is None or metrica_a_plotear == 0:
            plt.plot(generaciones, metrica0, "--ro")
            plt.ylabel(self.metricas[0])
            plt.xlabel("Generación")
            plt.show()
        if metrica_a_plotear is None or metrica_a_plotear == 1:
            plt.plot(generaciones, metrica1, "--bo")
            plt.ylabel(self.metricas[1])
            plt.xlabel("Generación")
            plt.show()

"""Fonseca, C.M., & Fleming, P.J. (1993). 
Genetic Algorithms for Multiobjective Optimization: FormulationDiscussion and Generalization. 
International Conference on Genetic Algorithms.
"""
class MOGA(GeneticoGenerico):

    def __init__(self, fiteador: TestFunctions.TiempoAeropuerto, N: int = 100,
                 pcross: float = 0.8, pmut: float = 0.3, psel: float = 0.2, T: int = 2, elitismo: int = 0,
                 metricas: list[str] = ["Tiempo", "Fuel"], pmut_minima: float = 0.1, num_gen_pmut_minima: int = 0):
        super().__init__(fiteador, N, pcross, pmut, psel, T, elitismo, metricas, pmut_minima, num_gen_pmut_minima)

    def fitness_Moga(self, pop: list) -> list:
        """
        Función encargada de calcular el fitness de acuerdo al algoritmo MOGA.
        Pasa del fitness, a los frentes, a la interpolación lineal de dichos frentes y a la media de los frentes.
        Parámetros:
            - pop: list. La lista de individuos cuyos fitness queremos calcular.
        Devuelve:
            - fitness_final: list. Los fitness de cada individuo calculados conforme al algoritmo MOGA empleando una interpolación lineal.
        """
        fitness = self.fit(pop)
        fitness = self.get_rango_individuos(fitness)
        pop, fitness = self.sort_pop(pop, fitness)

        xp = [1, self.N]
        fp = [1, max(fitness)]
        interpolacion = np.interp([i for i in range(1, self.N + 1)], xp, fp)
        
        frentes = {}
        for i in range(len(fitness)):
            if fitness[i] not in frentes.keys():
                frentes[fitness[i]] = [interpolacion[i]]
            else:
                frentes[fitness[i]].append(interpolacion[i])
        
        medias = {}
        for frente in frentes.keys():
            medias[frente] = sum(frentes[frentes]) / len(frentes[frente])
        
        fitness_final = [medias[i] for i in fitness]
        return fitness_final

    def evolve(self, num_gens: int = 100, num_fiteos:int = 0, verbose: int = 0) -> tuple:
        """
        Función que ejecuta el algoritmo MOGA.
        Parámetros:
            - num_gens: int=100. El número de generaciones que se quiere ejecutar el algoritmo. Si se proporciona num_gens, num_fiteos debe ser 0.
            - num_fiteos: int=0. El número de evaluaciones que se le otorga al algoritmo como presupuesto. Una vez excedido dicho presupuesto, el algoritmo termina la generación en la que se encuentre y termina.
            Si se le proporciona num_fiteos, num_gens debe ser 0 y viceversa. 
            - verbose: int. Entero que se utiliza para indicar el nivel de verbosidad del algoritmo.
        Devuelve:
            - (pop, fitness_pop). Una tupla con la población final y su fitness.
        """
        super().evolve(num_gens, num_fiteos, verbose)

        pop = self.create_pop()
        fitness_pop = self.fit(pop)

        generacion = 1
        # Ejecutar mientras queden generaciones o mientras queden evaluaciones.
        while (generacion <= self.num_gens) or (self.num_fiteos > 0):

            self.actualizar_historico(pop, fitness_pop)
            self.print_info(fitness_pop, generacion)

            pop, fitness_pop = self.crea_siguiente_gen(pop, fitness_pop)
            generacion += 1
            self.pmut = max(self.pmut - self.pmut_decrecimiento, self.pmut_minima)

        self.actualizar_historico(pop, fitness_pop)
        self.print_info(fitness_pop, generacion)
        
        return pop, fitness_pop

"""E. Zitzler and L. Thiele, "Multiobjective evolutionary algorithms: a comparative case study and the strength Pareto approach,"
in IEEE Transactions on Evolutionary Computation, vol. 3, no. 4, pp. 257-271, Nov. 1999, doi: 10.1109/4235.797969.
"""
class SPEA (GeneticoGenerico):
    
    def __init__(self, fiteador: TestFunctions.TiempoAeropuerto, N: int = 100,
                 pcross: float = 0.8, pmut: float = 0.3, psel: float = 0.2, T: int = 2, elitismo: int = 0, tamPiscina: int=10,
                 metricas: list[str] = ["Tiempo", "Fuel"], pmut_minima: float = None, num_gen_pmut_minima: int = 0):
        self.tamPiscina = tamPiscina
        super().__init__(fiteador, N, pcross, pmut, psel, T, elitismo, metricas, pmut_minima, num_gen_pmut_minima)

    def calcular_fitness_SPEA(self, pop:list, fitness_pop: list, piscina_elitista: list) -> tuple[list, list]:
        """
        Función encargada de calcular el fitness de acuerdo al algoritmo SPEA.
        Parámetros:
            - pop: list. La lista de individuos cuyos fitness queremos calcular.
            - fitness_pop: list. La lista con los fitness de la población.
            - piscina_elitista: list. La lista con la población elitista de SPEA.
        Devuelve:
            - (pop_aux, fitness_aux): tuple. Tupla con la población ordenada respecto al nuevo fitness y el nuevo fitness (también ordenado).
        """

        # Para cada individuo en la piscina, su nuevo fitness es el número de individuos de la población que domina 
        # entre el número total de individuos de la población + 1.
        nueva_pop = []
        for ind_piscina, fit_piscina in piscina_elitista:
            dominados = 0
            for fit_pop in fitness_pop:
                if self.es_dominado(fitness1=fit_pop, fitness2=fit_piscina):
                    dominados += 1
            nueva_pop.append((ind_piscina, dominados / self.N + 1))
        
        # Para la población normal se suma las fuerzas (lo calculado anteriormente) de todos los individuos de la piscina que dominan al individuo de la población (y se suma 1).
        for indice_ind in range(len(pop)):
            fuerzas_dominantes = 1
            for indice_piscina in range(len(piscina_elitista)):
                if self.es_dominado(fitness1=fitness_pop[indice_ind], fitness2=piscina_elitista[indice_piscina][1]):
                    fuerzas_dominantes += nueva_pop[indice_piscina][1]
            nueva_pop.append((pop[indice_ind], fuerzas_dominantes))
        
        pop_aux = [ind[0] for ind in nueva_pop]
        fitness_aux = [ind[1] for ind in nueva_pop]
        return pop_aux, fitness_aux

    def actualizar_piscina(self, pop: list, fitness_pop: list, piscina_elititsta: list) -> list[tuple]:
        """
        Función que maneja la actualización de la piscina elitista de SPEA.
        Parámetros:
            - pop: list. La población de la generación actual.
            - fitness_pop: list. Los fitness de la población actual.
            - piscina_elitista: list[tuple]. La piscina elitista a actualizar.
        Devuelve:
            - piscina_elitista: list[tuple]. La piscina elitista pero actualizada.
        """
        rangos = self.get_rango_individuos(fitness_pop)
        for i in range(len(rangos)):
            if rangos[i] == 1:
                piscina_elititsta.append((pop[i], fitness_pop[i]))
        
        rangos_piscina = self.get_rango_individuos([ind[1] for ind in piscina_elititsta])
        piscina_elititsta = [piscina_elititsta[i] for i in range(len(piscina_elititsta)) if rangos_piscina[i] == 1]

        if len(piscina_elititsta) > self.tamPiscina:
            kmeans = KMeans(n_clusters=self.tamPiscina, random_state=0, n_init="auto").fit([ind[0] for ind in piscina_elititsta])
            nueva_pop = [list(map(int, ind)) for ind in kmeans.cluster_centers_]

            for i, ind in enumerate(nueva_pop):
                nueva_pop[i] = list(self.fiteador.check_bounds(ind))

            nuevo_fit = self.fit(nueva_pop)
            piscina_elititsta = list(zip(nueva_pop, nuevo_fit))
        return piscina_elititsta

    def evolve(self, num_gens: int = 100, num_fiteos:int = 0, verbose: int = 0) -> tuple:
        """
        Función que ejecuta el algoritmo evolutivo de SPEA.
        Parámetros:
            - num_gens: int=100. El número de generaciones que se quiere ejecutar el algoritmo. Si se proporciona num_gens, num_fiteos debe ser 0.
            - num_fiteos: int=0. El número de evaluaciones que se le otorga al algoritmo como presupuesto. Una vez excedido dicho presupuesto, el algoritmo termina la generación en la que se encuentre y termina.
            Si se le proporciona num_fiteos, num_gens debe ser 0 y viceversa. 
            - verbose: int. Entero que se utiliza para indicar el nivel de verbosidad del algoritmo.
        Devuelve:
            - (pop, fitness_pop). Una tupla con la población final y su fitness.
        """
        super().evolve(num_gens, num_fiteos, verbose)

        pop = self.create_pop()
        fitness_pop = self.fit(pop)
        piscina_elitista = []

        generacion = 1

        # Ejecutar mientras queden generaciones o mientras queden evaluaciones.
        while (generacion <= self.num_gens) or (self.num_fiteos > 0):

            self.actualizar_historico(pop, fitness_pop)
            self.print_info(fitness_pop, generacion)
            
            piscina_elitista = self.actualizar_piscina(pop, fitness_pop, piscina_elitista)

            pop_aux, fitness_aux = self.calcular_fitness_SPEA(pop, fitness_pop, piscina_elitista)
            
            pop_aux, fitness_aux,  = self.sort_pop(pop_aux, fitness_aux)
            pop, fitness_pop = self.crea_siguiente_gen(pop_aux)
            generacion += 1
            self.pmut = max(self.pmut - self.pmut_decrecimiento, self.pmut_minima)

        self.actualizar_historico(pop, fitness_pop)
        self.print_info(fitness_pop, generacion)

        return pop, fitness_pop

"""K. Deb, A. Pratap, S. Agarwal and T. Meyarivan,
"A fast and elitist multiobjective genetic algorithm: NSGA-II,"
in IEEE Transactions on Evolutionary Computation, vol. 6, no. 2, pp. 182-197, April 2002, doi: 10.1109/4235.996017"""    
class NSGA2(GeneticoGenerico):

    def __init__(self, fiteador: TestFunctions.TiempoAeropuerto, N: int = 100,
                 pcross: float = 0.8, pmut: float = 0.3, psel: float = 0.2, T: int = 2, elitismo: int = 0,
                 metricas: list[str] = ["Tiempo", "Fuel"], pmut_minima: float = None, num_gen_pmut_minima: int = 0):
        super().__init__(fiteador, N, pcross, pmut, psel, T, elitismo, metricas, pmut_minima, num_gen_pmut_minima)

    def evolve(self, num_gens: int=100, num_fiteos: int=0, verbose: int=0) -> tuple:
        """
        Función encargada de realizar el proceso evolutivo del algoritmo NSGA2.
        Parámetros:
            - num_gens: int=100. El número de generaciones que se quiere ejecutar el algoritmo. Si se proporciona num_gens, num_fiteos debe ser 0.
            - num_fiteos: int=0. El número de evaluaciones que se le otorga al algoritmo como presupuesto. Una vez excedido dicho presupuesto, el algoritmo termina la generación en la que se encuentre y termina.
            Si se le proporciona num_fiteos, num_gens debe ser 0 y viceversa. 
            - verbose: int. Entero que se utiliza para indicar el nivel de verbosidad del algoritmo.
        Devuelve:
            - (pop, fitness_pop). Una tupla con la población final y su fitness.
        """
        super().evolve(num_gens, num_fiteos, verbose)
        
        pop = self.create_pop()
        fitness_pop = self.fit(pop)

        generacion = 1
        # Ejecutar mientras queden generaciones o mientras queden evaluaciones.
        while (generacion <= self.num_gens) or (self.num_fiteos > 0):

            self.actualizar_historico(pop, fitness_pop)
            self.print_info(fitness_pop, generacion)            

            rangos = self.get_rango_individuos(fitness_pop)
            pop_sorted, rangos = self.sort_pop(pop, rangos)
            pop_aux, fitness_aux = self.crea_siguiente_gen(pop_sorted)

            pop.extend(pop_aux)
            fitness_pop.extend(fitness_aux)

            frentes = self.get_frentes(fitness_pop)
            next_pop = []
            next_fit = []
            indice_frente = 0

            while (len(next_pop) < self.N) and (indice_frente < len(frentes)):
                frente_actual = frentes[indice_frente]

                # Si el frente cabe en la siguiente generación se añade entero.
                if len(frente_actual) <= (self.N - len(next_pop)):
                    next_pop.extend([pop[ind[0]] for ind in frente_actual])
                    next_fit.extend([ind[1] for ind in frente_actual])
                else: # Si no cabe, se emplea la distancia de crowding.
                    fit_frente = [ind[1] for ind in frente_actual]
                    ind_frente = [pop[ind[0]] for ind in frente_actual]

                    individuos_crowding = self.distancia_crowding(ind_frente, fit_frente)
                    seleccionados = individuos_crowding[0: (self.N - len(next_pop))]
                    next_pop.extend([seleccionado["ind"] for seleccionado in seleccionados])
                    next_fit.extend([seleccionado["met"] for seleccionado in seleccionados])

                indice_frente += 1
            
            pop = next_pop
            fitness_pop = next_fit
            generacion += 1
            self.pmut = max(self.pmut - self.pmut_decrecimiento, self.pmut_minima)

        self.actualizar_historico(pop, fitness_pop)
        self.print_info(fitness_pop, generacion)

        return pop, fitness_pop 

