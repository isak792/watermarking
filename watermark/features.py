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


