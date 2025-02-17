import hashlib
import json
import math
import os
import random
from pathlib import Path

import numpy as np
import pandas as pd
import pywt
import typer
from loguru import logger
from PIL import Image
from scipy.stats import kurtosis, skew
from tqdm import tqdm

from watermark.config import PROCESSED_DATA_DIR


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

    def string2bit(self, text: str):
        """ Genera un array de bits de una cadena de caracteres """
        binary_string = ''.join(format(ord(char), '08b') for char in text)
        output = [int(char) for char in binary_string]
        return output

    def bit2string(self, arr):
        """ Toma un arr de bits y lo tranforma a un string """
        # Binary data
        bin_data = ""
        for bit in arr:
            bin_data += str(bit)
        
        # Split the binary string into chunks of 8 bits (1 byte)
        char = [bin_data[i:i+8] for i in range(0, len(bin_data), 8)]

        # Convert binary to string
        string = ''.join(chr(int(i, 2)) for i in char)

        return string


    def array_xor(self, arr1, arr2):
        """ Implementa un XOR elemento a elemento para 2 arrays del mismo tamaño """
        # Revisar que ambos arreglos sean del mismo tamanio
        if len(arr1) != len(arr2):
            return None
        
        output = []
        for i in range(len(arr1)):
            # Tenemos que castear los valores a booleano, para hacer XOR
            # posterior a esto el resultado lo castemos a int para mayor sencilles.
            output.append(int(bool(arr1[i]) ^ bool(arr2[i])))

        return output

    def convertir_timestamp_a_binario(self,timestamp: str) -> str:
        """
        ESTATUS - Deprecrated
        Convierte un timestamp dado a una cadena binaria.
        
        Cada carácter del timestamp se convierte en su representación en 8 bits.
        
            :param timestamp: Cadena de texto con el timestamp, por ejemplo "2025-01-31 15:30:00"
            :return: Cadena que representa el timestamp en formato binario
        """
        binary_string = ''.join(format(ord(char), '08b') for char in timestamp)
        return binary_string
    
    def get_vector(self,bits):
        """ 
        ESTATUS - Not Used.
        Genera un vector a partir de una secuencia de bits
        """
        vector = [int(bit) for bit in bits]
        return vector
    

    def get_signal(self, vector, max_x):
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
    

    def sign2wavelet(self, firma, wavelet='haar', nivel=2):
        """
        Aplica la transformada wavelet discreta a una firma y extrae estadísticas de los coeficientes.

        INPUTS
        ---
            :param firma: lista de señales de las cuales vamos a obtener las estadisticas.
            :param wavelet: El algoritmo para obtener las estadisticas de la señal
            :param nivel: Parametro de funcion auxiliar el nivel de la señal a revisar HH
            :return: Regresa las caracteristicas de la señal de entrada en forma de una lista.
        """
        coeffs = pywt.wavedec(firma, wavelet, level=nivel) # Se requiere mas info del funcionamiento de wavedec
        estadisticas = []

        # Definir por que usamos estas estadisticas.
        for coef in coeffs:
            estadisticas.extend([np.mean(coef), np.std(coef), 
                                np.min(coef), np.max(coef),
                                skew(coef), kurtosis(coef)])

        return np.array(estadisticas)
    

    def dates_generator(self, n):
        """
        Genera un número de N de fechas aleatorias.

        INPUTS
        ---
            :param n: Número de Fechas a generar
            :return: una lista de con las fechas generadas.
        """
        dates = []
        for i in range(n):
            # Diccionario de los dias por mes para evitar errores en la generacion
            days_per_month = {1: 31, 2: 28, 3: 31, 4: 30, 5: 31, 6: 30, 7: 31, 
                    8: 31, 9: 30, 10: 31, 11: 30, 12: 31}
            año = random.randint(1995, 2025)
            mes = random.randint(1, 12)
            dia = random.randint(1, days_per_month[mes])
            hora = random.randint(1,23)
            minuto = random.randint(0,59)
            seg = random.randint(0,59)
            dates.append(f"{año}/{mes:02d}/{dia:02d} {hora:02d}:{minuto:02d}:{seg:02d}")
        return dates
    
    def hashear_fecha(self,fecha):
        """
        Convierte la fecha en un hash SHA-256 y lo transforma en un vector numérico
        
        INPUTS
        ---
            :param fecha: Fecha a la cual se le generará el mapa hash
            :return: un vector de valores del mapa hash.
        """
        hash_obj = hashlib.sha256(fecha.encode())
        hash_hex = hash_obj.hexdigest()
        hash_vector = np.array([int(hash_hex[i : i+2], 16) for i in range(0, len(hash_hex), 2)])
        return hash_vector
    
    #Aplicar Logistic Chaotic Map para cifrar la imagen
    def logistic_map_encrypt(self,image_matrix, r=3.99, x0=0.5):
        """Cifra una imagen con Logistic Chaotic Map"""
        h, w = image_matrix.shape
        chaos_seq = np.zeros(h * w)
        x = x0

        for i in range(h * w):
           x = r * x * (1 - x)  # Logistic map equation
           chaos_seq[i] = x

        # Convertir la secuencia caótica en índices de permutación
        chaos_idx = np.argsort(chaos_seq)

        # Aplanar la imagen y aplicar la permutación
        img_flat = image_matrix.flatten()
        encrypted_flat = img_flat[chaos_idx]

        # Restaurar la forma de la imagen
        encrypted_image = encrypted_flat.reshape(h, w)
        return np.uint8(encrypted_image), chaos_idx  # Retornamos también el índice de permutación
    
    
    def text_to_binary_v2(self,text):#
        """Convierte un texto en su representación binaria de 8 bits por carácter"""
        bin_text = ''.join(format(ord(c), '08b') for c in text)
        return bin_text

    def generate_image_from_text_v2(self,text):
        """Convierte un texto en una imagen binaria de 32x32"""
        bin_text = self.text_to_binary_v2(text)
        faltantes = 1024 - len(bin_text)
        if faltantes > 0:
           bin_text += ''.join(str(b) for b in np.random.randint(0, 2, faltantes))
        bit_array = np.array([int(b) for b in bin_text[:1024]]).reshape(32, 32) * 255
        return Image.fromarray(np.uint8(bit_array), mode="L"), bit_array
    
class Encription_images:
    def __init__(self):
        pass

    def generate_encrypted_date_images(self,n, save_path="generated_data.json"):
        """Genera N imágenes cifradas de fechas aleatorias y guarda los datos en un archivo JSON."""
        utils = Utilities()
        dates = utils.dates_generator(n)
        images = []
        stored_chaos_indices = {}
        stored_dates = {}

        for idx, date in enumerate(dates):
           img, bit_array = utils.generate_image_from_text_v2(date)
           encrypted_img, chaos_idx = utils.logistic_map_encrypt(bit_array)

           stored_chaos_indices[idx] = chaos_idx.tolist()  # Guardar índices de permutación
           stored_dates[idx] = date  # Guardar fecha original

           images.append((idx, date, img, encrypted_img))  # Guardar imagen cifrada
    
        # Guardar los datos en un archivo JSON
        data_to_save = {
          "dates": stored_dates,
          "chaos_indices": stored_chaos_indices
        }
        with open(save_path, "w") as f:
             json.dump(data_to_save, f, indent=4)

        return images, stored_chaos_indices
    
    def logistic_map_decrypt(self,encrypted_matrix, chaos_idx):
        """Descifra una imagen usando Logistic Chaotic Map (permuta inversa)"""
        h, w = encrypted_matrix.shape
        decrypted_flat = np.zeros(h * w, dtype=np.uint8)

        # Asegúrate de que chaos_idx sea un array de enteros y tenga la longitud correcta
        chaos_idx = np.array(chaos_idx, dtype=int)

        # Asegúrate de que los índices estén dentro del rango
        if chaos_idx.max() >= h * w:
            raise ValueError("Los índices de permutación están fuera del rango permitido")

        # Aplicar la permutación inversa
        decrypted_flat[chaos_idx] = encrypted_matrix.flatten()

        # Restaurar la forma original de la imagen
        decrypted_image = decrypted_flat.reshape(h, w)
        return decrypted_image