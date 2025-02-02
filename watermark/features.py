from pathlib import Path

import typer
from loguru import logger
from tqdm import tqdm

from watermark.config import PROCESSED_DATA_DIR

import os
import pandas as pd


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
    


