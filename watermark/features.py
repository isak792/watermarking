from pathlib import Path

import typer
from loguru import logger
from tqdm import tqdm

from watermark.config import PROCESSED_DATA_DIR

import os
import pandas as pd

import math
import pywt
import numpy as np


class DataFrameAnalyzer:
    def __init__(self, df):
        self.df = df  
    
    def print_info(self):
        print("DataFrame Information:")
        self.df.info()  
    
    def print_class_distribution(self, class_column):
        if class_column in self.df.columns:
            print("\nClass distribution:")
            print(self.df[class_column].value_counts())  
        else:
            print(f"La columna '{class_column}' no existe en el DataFrame.")

class DataStatistics:
    def __init__(self, df):
        self.df = df
    
    def print_summary_statistics(self):
        print("\nSummary statistics:")
        print(self.df.describe())

class Utilities:
    def __init__(self):
        pass

    def convertir_timestamp_a_binario(self,timestamp: str) -> str:
       """
         Convierte un timestamp dado a una cadena binaria.
    
         Cada carácter del timestamp se convierte en su representación en 8 bits.
    
        :param timestamp: Cadena de texto con el timestamp, por ejemplo "2025-01-31 15:30:00"
        :return: Cadena que representa el timestamp en formato binario
       """
       binary_string = ''.join(format(ord(char), '08b') for char in timestamp)
       return binary_string
    
    # Función para obtener un vector a partir de los bits
    def get_vector(self,bits):
       vector = [int(bit) for bit in bits]
       return vector
    
    def get_signal(self,vector, max_x):
       # Número de ciclos de la señal
       num_ciclos = max_x//len(vector) + 1
       # Lista para almacenar las coordenadas (x, y)
       coordenadas = []
       # Generar coordenadas para la señal de secuencia
       for ciclo in range(num_ciclos):
        for i, valor in enumerate(vector):
            if valor == 1:
                coordenadas.append((i + len(vector) * ciclo, 1))
            else:
                coordenadas.append((i + len(vector) * ciclo, -1))

       # Añadir las coordenadas finales para llegar al valor máximo en el eje x
       coordenadas.append((max_x, coordenadas[-1][1]))
       # Extraer coordenadas x y y
       x, y = zip(*coordenadas)
       return x, y
    
    def get_dwt_coeffs(self,x_series, vector, porcentaje):
       wavedecs_list = [] #Lista de df con f1 y f2
       for serie in x_series:
        coeffs_f2 = pywt.wavedec(serie, wavelet='db4', level=3)
        freq_2_mod = 2  # 1 las freq mas altas, 2 las medias
        og_coeffs = coeffs_f2[freq_2_mod] # Coeficientes a modificar
        mod_coeffs = np.zeros_like(og_coeffs)
        max_x = len(og_coeffs)
        _, y = self.get_signal(vector, max_x)

        for i, c in enumerate(og_coeffs):
            if y[i] == 1:
                mod_coeffs[i] = math.floor(c/porcentaje)*porcentaje + (3*porcentaje)/4
            else:
                mod_coeffs[i] = math.floor(c/porcentaje)*porcentaje + porcentaje/4

        df = pd.DataFrame({'original_c':og_coeffs, 'wm_c':mod_coeffs})
        wavedecs_list.append(df)
       return wavedecs_list
    
    


