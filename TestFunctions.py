import numpy as np
from AbsObjetiveFunc import AbsObjetiveFunc
from numba import jit
import pandas as pd
from collections import deque
from scipy.stats import pearson3

"""
Example of objective function.

Counts the number of ones in the array
"""


class DiophantineEq(AbsObjetiveFunc):
    def __init__(self, size, coeff, target, opt="min"):
        self.size = size
        self.coeff = coeff
        self.target = target
        super().__init__(self.size, opt)

    def objetive(self, solution):
        return abs((solution * self.coeff).sum() - self.target)

    def random_solution(self):
        return (np.random.randint(-100, 100, size=self.size)).astype(np.int32)

    def check_bounds(self, solution):
        return solution.astype(np.int32)


class MaxOnesReal(AbsObjetiveFunc):
    def __init__(self, size, opt="max"):
        self.size = size
        super().__init__(self.size, opt)

    def objetive(self, solution):
        return solution.sum()

    def random_solution(self):
        return np.random.random(self.size)

    def check_bounds(self, solution):
        return np.clip(solution.copy(), 0, 1)


# https://www.scientificbulletin.upb.ro/rev_docs_arhiva/rez0cb_759909.pdf

class Sphere(AbsObjetiveFunc):
    def __init__(self, size, opt="min"):
        self.size = size
        super().__init__(self.size, opt)

    def objetive(self, solution):
        return (solution ** 2).sum()

    def random_solution(self):
        return 200 * np.random.random(self.size) - 100

    def check_bounds(self, solution):
        return np.clip(solution, -100, 100)


class Rosenbrock(AbsObjetiveFunc):
    def __init__(self, size, opt="min"):
        self.size = size
        super().__init__(self.size, opt)

    def objetive(self, solution):
        return rosenbrock(solution)

    def random_solution(self):
        return 200 * np.random.random(self.size) - 100

    def check_bounds(self, solution):
        return np.clip(solution, -100, 100)


class Rastrigin(AbsObjetiveFunc):
    def __init__(self, size, opt="min"):
        self.size = size
        super().__init__(self.size, opt)

    def objetive(self, solution):
        return rastrigin(solution)

    def random_solution(self):
        return 10.24 * np.random.random(self.size) - 5.12

    def check_bounds(self, solution):
        return np.clip(solution, -5.12, 5.12)


class Test1(AbsObjetiveFunc):
    def __init__(self, size, opt="min"):
        self.size = size
        super().__init__(self.size, opt)

    def objetive(self, solution):
        return sum([(2 * solution[i - 1] + solution[i] ** 2 * solution[i + 1] - solution[i - 1]) ** 2 for i in
                    range(1, solution.size - 1)])

    def random_solution(self):
        return 4 * np.random.random(self.size) - 2

    def check_bounds(self, solution):
        return np.clip(solution, -2, 2)


class TiempoAeropuerto(AbsObjetiveFunc):

    MEDIA_ESTANCIA_STAND = 6057

    def __init__(self, size, stands, data, bounds, tin, tout, tstp, emissions, option="time", opt="min", no_fuel_on_stop=False, metricas_a_devolver=["total_time_est", "total_fuel"]):
        self.size = size
        self.stands = stands
        self.data = data
        self.bounds = bounds
        self.emissions = emissions
        self.Tin = tin
        self.Tout = tout
        self.Tstp = tstp
        self.option = option
        self.factor = 1
        self.no_fuel_on_stop = no_fuel_on_stop
        self.metricas_a_devolver = metricas_a_devolver
        super().__init__(self.size, opt)

    def ajustar_consistencia_vuelos(self):
        # Para garantizar la consistencia de los datos hay que asegurarse de que, una vez especificado el stand al que viene un avión, ese mismo vuelo salga de ese mismo stand y no de otro.
        # Básicamente tenemos que asegurar que el próximo vuelo que haga un avión que hayamos modificado salga del mismo stand en el que lo dejamos. 
        for index, vuelo_codificado in self.data[self.data.cod == 1].iterrows():
            vuelos_salida = self.data[(self.data.aircraft_id == vuelo_codificado.aircraft_id) & (self.data.flight_type == 1) & (self.data.index >= index)]
            if vuelos_salida.shape[0] > 0:
                indice_vuelo = vuelos_salida.index[0]
                self.data.loc[indice_vuelo, "stand"] = vuelo_codificado.stand

    def correspondencia_runways(self, runway):
        if "32" in runway:
            return "14" + runway[-1]
        elif "14" in runway:
            return "32" + runway[-1]
        elif "36" in runway:
            return "18" + runway[-1]
        elif "18" in runway:
            return "36" + runway[-1]

    def crear_vuelo_ficticio_llegada(self, vuelo_salida): # El vuelo de llegada a asociar con el vuelo de salida
        vuelo_ficticio = vuelo_salida.iloc[0].copy(deep=True)
        vuelo_ficticio.ts_taxi_out_1 = 0
        vuelo_ficticio.flight_type = 2 
        vuelo_ficticio.ts_app_2 = max(0, vuelo_salida.iloc[0].ts_taxi_out_1 - self.MEDIA_ESTANCIA_STAND)
        vuelo_ficticio.cod = -1 # Codigo de vuelo ficticio
        vuelo_ficticio.runway = self.correspondencia_runways(vuelo_ficticio.runway)
        return vuelo_ficticio
    
    def crear_vuelo_ficticio_salida(self, vuelo_llegada): # El vuelo de salida a asociar con el de llegada
        vuelo_ficticio = vuelo_llegada.iloc[0].copy(deep=True)
        vuelo_ficticio.ts_app_2 = 0
        vuelo_ficticio.flight_type = 1
        vuelo_ficticio.ts_taxi_out_1 = max(0, vuelo_llegada.iloc[0].ts_app_2 + self.MEDIA_ESTANCIA_STAND)
        vuelo_ficticio.cod = -1 # Codigo de vuelo ficticip
        vuelo_ficticio.runway = self.correspondencia_runways(vuelo_ficticio.runway)
        return vuelo_ficticio
                  
    def objetive(self, solution, return_total=True):
        solution = pd.DataFrame(solution)
        take_off_time = 120 
        take_in_time = 21 
        total_time = np.zeros(self.data.shape[0])
        # Se obtienen los stands asignados en la solucion
        self.data.loc[self.data["cod"] == 1, "stand"] = [self.stands.stands[i] for i in solution.values]
        
        self.ajustar_consistencia_vuelos() # Para que los vuelos a los que asignamos stands salgan de dichos stands
        
        self.data.loc[:, "ts_stand"] = np.zeros(self.data.shape[0])
        self.data.loc[:, "ts_rw"] = np.zeros(self.data.shape[0])
        self.data.loc[:, "duration_stop"] = np.zeros(self.data.shape[0])
        self.data.loc[:, "ts_climb"] = np.zeros(self.data.shape[0])

        self.data["inicio_uso_pista"] = np.zeros(self.data.shape[0])
        self.data["fin_uso_pista"] = np.zeros(self.data.shape[0])

        for index in range(0, self.data.shape[0]):
            stop_tipico = 0

            if self.data.loc[index, "stand"] in set(self.Tstp.stand):
                stop_tipico = abs(pearson3.rvs(skew=1)/3) * (self.Tstp.loc[self.Tstp["stand"] == self.data.loc[index, "stand"], "media"].values[0])
            
            self.data.loc[index, "duration_stop"] = stop_tipico

            # Cuando cod es 0, se busca el stand del mismo avión
            if self.data.loc[index, "cod"] == 0:
                self.data.loc[index, "stand"] = self.data.loc[
                    np.where(self.data.loc[0:index, "aircraft_id"] == self.data.loc[index, "aircraft_id"])[0][-1], "stand"]
            # Para los aterrizajes, se calcula duración como suma de tiempo medio de parada y tiempo medio de Tin
            if self.data.loc[index, "flight_type"] == 2:
                if self.data.loc[index, "runway"] == "32L":
                    column_rw = "media32L"
                elif self.data.loc[index, "runway"] == "32R":
                    column_rw = "media32R"
                elif self.data.loc[index, "runway"] == "18L":
                    column_rw = "media18L"
                else:
                    column_rw = "media18R"
                duration_to_stand = np.nan
                if self.data.loc[index, "stand"] in set(self.Tin.stand):
                    duration_to_stand = \
                        self.Tin.loc[self.Tin["stand"] == self.data.loc[index, "stand"], column_rw].values[0]

                self.data.loc[index, "duration_stop"] = stop_tipico

                media_tin = self.Tin[column_rw].mean()
                duration_tin = (duration_to_stand if not pd.isna(duration_to_stand) else media_tin) + stop_tipico

                self.data.loc[index, "ts_stand"] = self.data.loc[index, "ts_app_2"] + duration_tin
                total_time[index] = duration_tin
                self.data.loc[index, "inicio_uso_pista"] = max(0, self.data.loc[index, "ts_app_2"] - take_in_time)
                self.data.loc[index, "fin_uso_pista"] = self.data.loc[index, "ts_app_2"]


            # Para despegues, se calcula timestamp de llegada al runway como timestamp de taxi out mas tiempo medio de Tout
            elif self.data.loc[index, "flight_type"] == 1:
                if self.data.loc[index, "runway"] == "36L":
                    column_rw = "media36L"
                elif self.data.loc[index, "runway"] == "36R":
                    column_rw = "media36R"
                elif self.data.loc[index, "runway"] == "14L":
                    column_rw = "media14L"
                else:
                    column_rw = "media14R"
                duration_to_stand = np.nan
                if self.data.loc[index, "stand"] in set(self.Tout.stand):
                    duration_to_stand = \
                        self.Tout.loc[self.Tout["stand"] == self.data.loc[index, "stand"], column_rw].values[0]
                media_tout = self.Tout[column_rw].mean()
                self.data.loc[index, "ts_rw"] = self.data.loc[index, "ts_taxi_out_1"] + (duration_to_stand if not pd.isna(duration_to_stand) else media_tout) + stop_tipico
                self.data.loc[index, "inicio_uso_pista"] = self.data.loc[index, "ts_rw"] 


        # Uso de cada pista por orden de llegada (FIFO)
        data36L = self.data.loc[(self.data["runway"] == "36L") | (self.data["runway"] == "18L")].sort_values(by="inicio_uso_pista")
        previous_index = -1
        for index, row in data36L.iterrows():
            # Los aterrizajes tienen la pista reservada y asumimos que no tendrán problemas.
            if previous_index == -1:
                self.data.loc[index, "duration_stop"] = 0

            if row.flight_type == 1:
                # Si es el primer vuelo en ese runway, no tiene que parar
                if previous_index == -1:
                    self.data.loc[index, "duration_stop"] = 0
                    self.data.loc[index, "ts_climb"] = self.data.loc[index, "ts_rw"] + take_off_time
                    
                    self.data.loc[index, "fin_uso_pista"] = self.data.loc[index, "ts_climb"]
                # En caso contrario, la parada sera la diferencia entre el timestamp de climb del anterior y el timestamp de llegada al runway
                else:
                    self.data.loc[index, "duration_stop"] = max(0,self.data.loc[previous_index, "fin_uso_pista"] - self.data.loc[index, "ts_rw"])
                    self.data.loc[index, "ts_climb"] = self.data.loc[index, "ts_rw"] + self.data.loc[index, "duration_stop"] + take_off_time

                    self.data.loc[index, "fin_uso_pista"] = self.data.loc[index, "ts_climb"]
                
                # La duración del despegue es desde el taxi out hasta el climb
                total_time[index] = self.data.loc[index, "ts_climb"] - self.data.loc[index, "ts_taxi_out_1"]
            previous_index = index

        # Se extraen y ordenan por timestamp de llegada al runway los despegues con runway 36R
        data36R = self.data.loc[(self.data["runway"] == "36R") | (self.data["runway"] == "18R")].sort_values(by="inicio_uso_pista")
        previous_index = -1
        for index, row in data36R.iterrows():
            # Los aterrizajes tienen la pista reservada y asumimos que no tendrán problemas.
            if row.flight_type == 1:
                # Si es el primer vuelo en ese runway, no tiene que parar
                if previous_index == -1:
                    self.data.loc[index, "duration_stop"] = 0
                    self.data.loc[index, "ts_climb"] = self.data.loc[index, "ts_rw"] + take_off_time
                    
                    self.data.loc[index, "fin_uso_pista"] = self.data.loc[index, "ts_climb"]
                # En caso contrario, la parada sera la diferencia entre el timestamp de climb del anterior y el timestamp de llegada al runway
                else:
                    self.data.loc[index, "duration_stop"] = max(0,self.data.loc[previous_index, "fin_uso_pista"] - self.data.loc[index, "ts_rw"])
                    self.data.loc[index, "ts_climb"] = self.data.loc[index, "ts_rw"] + self.data.loc[index, "duration_stop"] + take_off_time

                    self.data.loc[index, "fin_uso_pista"] = self.data.loc[index, "ts_climb"]
                
                # La duración del despegue es desde el taxi out hasta el climb
                total_time[index] = self.data.loc[index, "ts_climb"] - self.data.loc[index, "ts_taxi_out_1"]
            previous_index = index

        # Se extraen y ordenan por timestamp de llegada al runway los despegues con runway 14L
        data14L = self.data.loc[(self.data["runway"] == "14L") | (self.data["runway"] == "32L")].sort_values(by="inicio_uso_pista")
        previous_index = -1
        for index, row in data14L.iterrows():
            # Los aterrizajes tienen la pista reservada y asumimos que no tendrán problemas.
            if row.flight_type == 1:
                # Si es el primer vuelo en ese runway, no tiene que parar
                if previous_index == -1:
                    self.data.loc[index, "duration_stop"] = 0
                    self.data.loc[index, "ts_climb"] = self.data.loc[index, "ts_rw"] + take_off_time
                    
                    self.data.loc[index, "fin_uso_pista"] = self.data.loc[index, "ts_climb"]
                # En caso contrario, la parada sera la diferencia entre el timestamp de climb del anterior y el timestamp de llegada al runway
                else:
                    self.data.loc[index, "duration_stop"] = max(0,self.data.loc[previous_index, "fin_uso_pista"] - self.data.loc[index, "ts_rw"])
                    self.data.loc[index, "ts_climb"] = self.data.loc[index, "ts_rw"] + self.data.loc[index, "duration_stop"] + take_off_time

                    self.data.loc[index, "fin_uso_pista"] = self.data.loc[index, "ts_climb"]
                
                # La duración del despegue es desde el taxi out hasta el climb
                total_time[index] = self.data.loc[index, "ts_climb"] - self.data.loc[index, "ts_taxi_out_1"]
            previous_index = index

        # Se extraen y ordenan por timestamp de llegada al runway los despegues con runway 14R
        data14R = self.data.loc[(self.data["runway"] == "14R") | (self.data["runway"] == "32R")].sort_values(by="inicio_uso_pista")
        previous_index = -1
        for index, row in data14R.iterrows():
            # Los aterrizajes tienen la pista reservada y asumimos que no tendrán problemas.
            if row.flight_type == 1:
                # Si es el primer vuelo en ese runway, no tiene que parar
                if previous_index == -1:
                    self.data.loc[index, "duration_stop"] = 0
                    self.data.loc[index, "ts_climb"] = self.data.loc[index, "ts_rw"] + take_off_time
                    
                    self.data.loc[index, "fin_uso_pista"] = self.data.loc[index, "ts_climb"]
                # En caso contrario, la parada sera la diferencia entre el timestamp de climb del anterior y el timestamp de llegada al runway
                else:
                    self.data.loc[index, "duration_stop"] = max(0,self.data.loc[previous_index, "fin_uso_pista"] - self.data.loc[index, "ts_rw"])
                    self.data.loc[index, "ts_climb"] = self.data.loc[index, "ts_rw"] + self.data.loc[index, "duration_stop"] + take_off_time

                    self.data.loc[index, "fin_uso_pista"] = self.data.loc[index, "ts_climb"]
                
                # La duración del despegue es desde el taxi out hasta el climb
                total_time[index] = self.data.loc[index, "ts_climb"] - self.data.loc[index, "ts_taxi_out_1"]
            previous_index = index

        total_fuel = np.zeros(self.data.shape[0])
        media_motores = 2.3836
        media_consumo = 0.1808

        total_HC = np.zeros(self.data.shape[0])
        total_CO = np.zeros(self.data.shape[0])
        total_NOx = np.zeros(self.data.shape[0])
        
        # Valores por defecto en caso de no tener las contaminaciones reales
        media_HC_EI_Idle = 4.228102857142857
        media_CO_EI_Idle = 24.55532857142857
        media_NOx_EI_Idle = 4.6036721428571425

        for a in range(total_time.shape[0]):

            tiempo_en_movimiento = total_time[a] # Por defecto se asume un consumo constante todo el tiempo
            if self.no_fuel_on_stop: # Si se asume que los retrasos no consumen combustible
                tiempo_en_movimiento = total_time[a] - self.data.loc[a, "duration_stop"]

            if self.data.loc[a, "equip"] in set(self.emissions["equip"]):
                if pd.isna(self.emissions.loc[self.emissions["equip"] == self.data.loc[a, "equip"],"number engines"].values[0]):
                    if pd.isna(self.emissions.loc[self.emissions["equip"] == self.data.loc[a, "equip"],"Fuel Flow Idle (kg/sec)"].values[0]):
                        total_fuel[a] = media_motores * media_consumo * tiempo_en_movimiento
                    else:
                        total_fuel[a] = media_motores * tiempo_en_movimiento * self.emissions.loc[self.emissions["equip"] == self.data.loc[a, "equip"], "Fuel Flow Idle (kg/sec)"]
                else:
                    if pd.isna(self.emissions.loc[self.emissions["equip"] == self.data.loc[a, "equip"],"Fuel Flow Idle (kg/sec)"].values[0]):
                        total_fuel[a] = self.emissions.loc[self.emissions["equip"] == self.data.loc[a, "equip"], "number engines"] * \
                                        tiempo_en_movimiento * media_consumo
                    else:
                        total_fuel[a] = self.emissions.loc[self.emissions["equip"] == self.data.loc[a, "equip"], "number engines"] * tiempo_en_movimiento * self.emissions.loc[self.emissions["equip"] == self.data.loc[a, "equip"], "Fuel Flow Idle (kg/sec)"]            
            else:  # si no tenemos los datos de consumo del avión multiplicamos el tiempo por un valor costante
                total_fuel[a] = media_motores * media_consumo * tiempo_en_movimiento

            # Cálculo de los gases. Utilizamos el if para agilizar el código cuando no es necesario calcular los gases.
            if "Gases" in self.metricas_a_devolver:
                total_HC[a] = total_fuel[a] * media_HC_EI_Idle 
                total_CO[a] = total_fuel[a] * media_CO_EI_Idle
                total_NOx[a] = total_fuel[a] * media_NOx_EI_Idle

                if self.data.loc[a, "equip"] in set(self.emissions["equip"]):
                    if not pd.isna(self.emissions.loc[self.emissions["equip"] == self.data.loc[a, "equip"],"HC EI Idle (g/kg)"].values[0]):
                        total_HC[a] = total_fuel[a] * self.emissions.loc[self.emissions["equip"] == self.data.loc[a, "equip"], "HC EI Idle (g/kg)"]
                    if not pd.isna(self.emissions.loc[self.emissions["equip"] == self.data.loc[a, "equip"],"CO EI Idle (g/kg)"].values[0]):
                        total_CO[a] = total_fuel[a] * self.emissions.loc[self.emissions["equip"] == self.data.loc[a, "equip"], "CO EI Idle (g/kg)"]
                    if not pd.isna(self.emissions.loc[self.emissions["equip"] == self.data.loc[a, "equip"],"NOx EI Idle (g/kg)"].values[0]):
                        total_NOx[a] = total_fuel[a] * self.emissions.loc[self.emissions["equip"] == self.data.loc[a, "equip"], "NOx EI Idle (g/kg)"]

        self.data["total_fuel"] = total_fuel
        self.data["total_time_est"] = total_time
        if "Gases" in self.metricas_a_devolver:
            self.data["total_HC"] = total_HC
            self.data["total_CO"] = total_CO
            self.data["total_NOx"] = total_NOx

        if return_total:
            metricas = []
            for metrica in self.metricas_a_devolver:
                if metrica == "Gases":
                    metricas.append(total_HC.sum() + total_CO.sum() + total_NOx.sum())
                else:
                    metricas.append(self.data[metrica].sum())
            return tuple(metricas)
        else:
            return (total_time, total_fuel)

    def random_solution(self):
        return np.round((self.bounds[1] - self.bounds[0]) * np.random.random(self.size) + self.bounds[0]).astype(int)

    def check_bounds(self, solution):
        listapr = ['TERMINAL AVIACIÓN GENERAL', 'T-123 REMOTO']
        stands_util = deque()
        solution_mod = pd.DataFrame(np.array(np.clip(solution, self.bounds[0], self.bounds[1])).reshape(-1, 1))
        self.data.loc[self.data["cod"] == 1, "stand"] = solution_mod.values
        solution_mod = pd.DataFrame(np.array(solution_mod).reshape(1, -1))
        ocupado = pd.DataFrame(np.zeros(np.shape(self.stands)[0]))
        ind = 0
        for index in range(self.data.shape[0]):
            if self.data.loc[index, "cod"] == 1 and self.data.loc[index, "flight_type"] == 2 and (ocupado.iloc[solution_mod[ind][0]] == 0)[0]:
                if self.data.loc[index, "equip"] not in self.emissions["equip"].values:
                    if self.stands.loc[self.data.loc[index, "stand"], "terminal"] != "TERMINAL AVIACIÓN GENERAL":
                        ocupado.iloc[solution_mod[ind][0]] = 1
                        if stands_util.__contains__(solution_mod[ind][0]):
                            stands_util.remove(solution_mod[ind][0])
                        stands_util.append(solution_mod[ind][0])
                        ind += 1
                elif self.emissions.loc[self.emissions["equip"] == self.data.loc[index, "equip"], "manufacturer"].values[0] != "Privado" \
                      and self.stands.loc[self.data.loc[index, "stand"], "terminal"] != "TERMINAL AVIACIÓN GENERAL":
                    ocupado.iloc[solution_mod[ind][0]] = 1
                    if stands_util.__contains__(solution_mod[ind][0]):
                        stands_util.remove(solution_mod[ind][0])
                    stands_util.append(solution_mod[ind][0])
                    ind += 1
                elif self.emissions.loc[self.emissions["equip"] == self.data.loc[index, "equip"], "manufacturer"].values[0] == "Privado" \
                      and listapr.__contains__(str(self.stands.loc[self.data.loc[index, "stand"], "terminal"])):
                    ocupado.iloc[solution_mod[ind][0]] = 1
                    if stands_util.__contains__(solution_mod[ind][0]):
                        stands_util.remove(solution_mod[ind][0])
                    stands_util.append(solution_mod[ind][0])
                    ind += 1
            elif self.data.loc[index, "cod"] == 1 and self.data.loc[index, "flight_type"] == 1 and \
                    (ocupado.iloc[solution_mod[ind][0]] == 0)[0] and \
                    (((self.data.loc[index, "equip"] not in self.emissions["equip"].values or self.emissions.loc[
                        self.emissions["equip"] == self.data.loc[index, "equip"], "manufacturer"].values[0] != "Privado") and self.stands.loc[self.data.loc[index, "stand"], "terminal"] != "TERMINAL AVIACIÓN GENERAL") or (self.emissions.loc[self.emissions["equip"] == self.data.loc[index, "equip"], "manufacturer"].values[
                         0] == "Privado" and listapr.__contains__(str(self.stands.loc[self.data.loc[index, "stand"], "terminal"])))):
                ocupado.iloc[solution_mod[ind][0]] = 1
                if stands_util.__contains__(solution_mod[ind][0]):
                    stands_util.remove(solution_mod[ind][0])
                stands_util.append(solution_mod[ind][0])
                ind += 1
            elif self.data.loc[index, "cod"] == 0 and index > 0:
                ocupado.loc[self.data.loc[
                    np.where(self.data.loc[0:index, "aircraft_id"] == self.data.loc[index, "aircraft_id"])[0][
                        -1], "stand"]] = 0
                if stands_util.__contains__(solution_mod[ind][0]):
                    stands_util.remove(solution_mod[ind][0])
                stands_util.append(solution_mod[ind][0])
            else:  # encoded failed
                if index > 0:
                    if ocupado.sum()[0] < self.stands.__len__():
                        # si no se han llenado todos los stands
                        distance = abs(pd.DataFrame(range(0, self.stands.__len__()))[ocupado == 0] - solution_mod[ind][0])
                        # si es comercial se le asigna el que más cerca esté que no sea de la terminal T.A.G.
                        if self.data.loc[index, "equip"] not in self.emissions["equip"].values or self.emissions.loc[self.emissions["equip"] == self.data.loc[index, "equip"], "manufacturer"].values[0] != "Privado":
                            distance[self.stands["terminal"] == "TERMINAL AVIACIÓN GENERAL"] = 1e6
                        else:
                            distance[self.stands["terminal"] == "T-123"] = 1e6
                            distance[self.stands["terminal"] == "T-4"] = 1e6
                            distance[self.stands["terminal"] == "T-4 REMOTO"] = 1e6
                            distance[self.stands["terminal"] == "T-4S"] = 1e6
                            distance[self.stands["terminal"] == "T-4S REMOTO"] = 1e6
                        solution_mod[ind][0] = np.where(distance == distance.min())[0][0]
                        ocupado.iloc[solution_mod[ind][0]] = 1
                        if stands_util.__contains__(solution_mod[ind][0]):
                            stands_util.remove(solution_mod[ind][0])
                        stands_util.append(solution_mod[ind][0])
                        ind += 1
                    else:  # si llega el caso le asigno el último que se usó
                        pila = deque()
                        corregido = False
                        # si es comercial se le asigna el stand que no sea de la terminal TAG y que se utilizó hace más tiempo
                        if self.data.loc[index, "equip"] not in self.emissions["equip"].values or self.emissions.loc[self.emissions["equip"] == self.data.loc[index, "equip"], "manufacturer"].values[0] != "Privado":
                            for a in range(stands_util.__len__()):
                                checkstand = stands_util.popleft()
                                if self.stands.loc[checkstand, "terminal"] != "TERMINAL AVIACIÓN GENERAL":
                                    solution_mod[ind][0] = checkstand
                                    corregido = True
                                else:
                                    pila.append(checkstand)
                                if corregido:
                                    break
                        else:
                            for a in range(stands_util.__len__()):
                                checkstand = stands_util.popleft()
                                if listapr.__contains__(str(self.stands.loc[checkstand, "terminal"])):
                                    solution_mod[ind][0] = checkstand
                                    corregido = True
                                else:
                                    pila.append(checkstand)
                                if corregido:
                                    break
                        for a in range(pila.__len__()):
                            stands_util.appendleft(pila.pop())
                        stands_util.append(solution_mod[ind][0])
                        ind += 1
        return np.array(solution_mod).reshape(-1).reshape(-1)


@jit(nopython=True)
def rosenbrock(solution):
    term1 = solution[1:] - solution[:-1] ** 2
    term2 = 1 - solution[:-1]
    result = 100 * term1 ** 2 + term2 ** 2
    return result.sum()

@jit(nopython=True)
def rastrigin(solution, A=10):
    return (A * len(solution) + (solution ** 2 - A * np.cos(2 * np.pi * solution)).sum())