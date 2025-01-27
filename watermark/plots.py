from pathlib import Path

import typer
from loguru import logger
from tqdm import tqdm

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

from watermark.config import FIGURES_DIR, PROCESSED_DATA_DIR

class PlotHandler:
    """
    Clase principal para crear y guardar visualizaciones básicas de datos.

    Esta clase proporciona métodos para generar diferentes tipos de gráficos
    estadísticos y guardarlos automáticamente en el directorio de figuras.

    Attributes:
        data (pd.DataFrame): DataFrame con los datos a visualizar.

    Example:
        >>> plotter = PlotHandler(df)
        >>> plotter.plot_hist('column_name', save=True)
    """

    def __init__(self, data: pd.DataFrame):
        """
        Inicializa el manejador de gráficos.

        Args:
            data (pd.DataFrame): DataFrame con los datos a visualizar.
        """
        self.data = data

    def save_plot(self, filename: str, show: bool = False) -> None:
        """
        Guarda el gráfico actual en un archivo.

        Args:
            filename (str): Nombre del archivo para guardar el gráfico.
            show (bool, opcional): Si se debe mostrar el gráfico. Defaults to False.

        Note:
            - Automáticamente agrega la extensión .png si no está presente
            - Guarda el gráfico en el directorio FIGURES_DIR
        """
        if not filename.endswith('.png'):
            filename += '.png'
        output_path = FIGURES_DIR / filename
        
        plt.savefig(output_path, bbox_inches='tight')
        logger.success(f"Plot saved successfully to {output_path}")
        if show:
            plt.show()
        plt.close()

    def plot_hist(self, column: str, bins: int = 30, save: bool = False, 
                 filename: str = "histogram.png") -> None:
        """
        Genera un histograma con estimación de densidad de kernel.

        Args:
            column (str): Nombre de la columna a visualizar.
            bins (int, opcional): Número de bins del histograma. Defaults to 30.
            save (bool, opcional): Si se debe guardar el gráfico. Defaults to False.
            filename (str, opcional): Nombre del archivo. Defaults to "histogram.png".

        Raises:
            KeyError: Si la columna no existe en el DataFrame.
        """
        plt.figure(figsize=(10, 6))
        sns.histplot(self.data[column], bins=bins, kde=True)
        plt.title(f'Histogram of {column}')
        plt.xlabel(column)
        plt.ylabel('Frequency')
        
        if save:
            self.save_plot(filename)

    def plot_corr_matrix(self, save: bool = False, 
                        filename: str = "correlation_matrix.png") -> None:
        """
        Genera una matriz de correlación usando un mapa de calor.

        Args:
            save (bool, opcional): Si se debe guardar el gráfico. Defaults to False.
            filename (str, opcional): Nombre del archivo. Defaults to "correlation_matrix.png".

        Note:
            Solo incluye columnas numéricas en el análisis de correlación.
        """
        plt.figure(figsize=(12, 8))
        numeric_data = self.data.select_dtypes(include='number')
        correlation_matrix = numeric_data.corr()
        sns.heatmap(correlation_matrix, annot=True, fmt='.2f', 
                   cmap='coolwarm', square=True)
        plt.title('Correlation Matrix')
        
        if save:
            self.save_plot(filename)

    def plot_boxplot(self, column: str, save: bool = False, 
                    filename: str = "boxplot.png") -> None:
        """
        Genera un boxplot para una columna específica.

        Args:
            column (str): Nombre de la columna a visualizar.
            save (bool, opcional): Si se debe guardar el gráfico. Defaults to False.
            filename (str, opcional): Nombre del archivo. Defaults to "boxplot.png".

        Raises:
            KeyError: Si la columna no existe en el DataFrame.
        """
        plt.figure(figsize=(10, 6))
        sns.boxplot(y=self.data[column])
        plt.title(f'Boxplot of {column}')
        
        if save:
            self.save_plot(filename)

class ClassPlotter:
    """
    Clase para visualizar la distribución de clases en un conjunto de datos.

    Esta clase proporciona funcionalidades para contar y visualizar la
    distribución de valores en una columna categórica.

    Attributes:
        df (pd.DataFrame): DataFrame con los datos.
        column (str): Nombre de la columna a analizar.
        class_counts (pd.Series): Conteo de clases (inicialmente None).
    """
    def __init__(self, df: pd.DataFrame, column: str):
        """
        Inicializa el visualizador de clases.

        Args:
            df (pd.DataFrame): DataFrame con los datos.
            column (str): Nombre de la columna a analizar.
        """
        self.df = df
        self.column = column
        self.class_counts = None

    def count_classes(self) -> None:
        """Calcula la frecuencia de cada clase en la columna especificada."""
        self.class_counts = self.df[self.column].value_counts()

    def plot_classes(self) -> None:
        """
        Genera un gráfico de barras con la distribución de clases.

        Raises:
            ValueError: Si no se ha llamado a count_classes() primero.
        """
        if self.class_counts is None:
            raise ValueError("You must count the classes before plotting.")
        
        plt.bar(self.class_counts.index, self.class_counts.values)
        plt.title(f'Count of Each {self.column}')
        plt.xlabel(self.column)
        plt.ylabel('Count')
        plt.show()

class CorrelationHeatmap:
    """
    Clase para generar mapas de calor de correlación.

    Esta clase proporciona métodos para calcular y visualizar matrices
    de correlación entre variables numéricas.

    Attributes:
        data (pd.DataFrame): DataFrame con los datos numéricos.
        correlation_matrix (pd.DataFrame): Matriz de correlación calculada.
    """

    def __init__(self, data: pd.DataFrame):
        """
        Inicializa el generador de mapas de calor.

        Args:
            data (pd.DataFrame): DataFrame con los datos numéricos.
        """
        self.data = data
        self.correlation_matrix = None

    def calculate_correlation(self) -> None:
        """Calcula la matriz de correlación para los datos proporcionados."""
        self.correlation_matrix = self.data.corr()
    
    def plot_heatmap(self, figsize: tuple = (16, 12), cmap: str = 'coolwarm', 
                    annot: bool = False) -> None:
        """
        Genera un mapa de calor de la matriz de correlación.

        Args:
            figsize (tuple, opcional): Tamaño de la figura. Defaults to (16, 12).
            cmap (str, opcional): Paleta de colores. Defaults to 'coolwarm'.
            annot (bool, opcional): Si se muestran valores. Defaults to False.

        Raises:
            ValueError: Si no se ha calculado la matriz de correlación.
        """
        if self.correlation_matrix is None:
            raise ValueError("You must calculate the correlation matrix before plotting.")
        
        plt.figure(figsize=figsize)
        sns.heatmap(self.correlation_matrix, cmap=cmap, annot=annot, 
                   cbar=False, xticklabels=True, yticklabels=True)
        plt.title('Correlation Heatmap')
        plt.show()

class BoxPlotGenerator:
    """
    Clase para generar múltiples boxplots en una cuadrícula.

    Esta clase crea una visualización de boxplots para múltiples features
    agrupados por una variable objetivo.

    Attributes:
        df (pd.DataFrame): DataFrame con los datos.
        target_column (str): Nombre de la columna objetivo.
        features (Index): Nombres de las columnas de features.
        n_cols (int): Número de columnas en la cuadrícula.
        n_features (int): Número total de features.
        n_rows (int): Número de filas necesarias en la cuadrícula.
    """

    def __init__(self, df: pd.DataFrame, target_column: str, n_cols: int = 4):
        """
        Inicializa el generador de boxplots.

        Args:
            df (pd.DataFrame): DataFrame con los datos.
            target_column (str): Nombre de la columna objetivo.
            n_cols (int, opcional): Número de columnas en la cuadrícula. 
                                  Defaults to 4.
        """
        self.df = df
        self.target_column = target_column
        self.features = df.columns.drop(target_column)
        self.n_cols = n_cols
        self.n_features = len(self.features)
        self.n_rows = (self.n_features - 1) // self.n_cols + 1

    def create_plots(self) -> None:
        """
        Genera una cuadrícula de boxplots para todos los features.

        Note:
            - Cada boxplot muestra la distribución del feature por clase objetivo
            - Las etiquetas del eje x se rotan 45 grados para mejor legibilidad
            - Los subplots no utilizados se ocultan automáticamente
        """
        fig, axes = plt.subplots(self.n_rows, self.n_cols, 
                               figsize=(20, 5 * self.n_rows))
        axes = axes.flatten()
        
        for i, feature in enumerate(self.features):
            sns.boxplot(x=self.target_column, y=feature, data=self.df, ax=axes[i])
            axes[i].set_title(feature)
            axes[i].tick_params(axis='x', rotation=45)
        
        # Ocultar subplots vacíos
        for j in range(i + 1, len(axes)):
            axes[j].set_visible(False)
            
        plt.tight_layout()
        plt.show()
