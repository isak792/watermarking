from pathlib import Path

import numpy as np
import pandas as pd
import typer
import yaml
from loguru import logger
from tqdm import tqdm

from watermark.config import INTERIM_DATA_DIR, PROCESSED_DATA_DIR, RAW_DATA_DIR


class DataHandler:
    """
    Clase para manejar operaciones de carga, procesamiento y guardado de datos.

    Esta clase proporciona funcionalidades para el manejo completo del ciclo de vida
    de datos, incluyendo carga desde Pkl, procesamiento, escalado y guardado.

    Attributes:
        data (pd.DataFrame): DataFrame que contiene los datos cargados.
        le (LabelEncoder): Instancia de LabelEncoder para codificación de etiquetas.
    """

    def __init__(self):
        """
        Inicializa una nueva instancia de DataHandler.

        El DataFrame data se inicializa como None y se creará al cargar los datos.
        """
        self.data = None
        

    def load_data(self, input_path: Path, input_filename: str) -> pd.DataFrame:
        """
        Carga datos desde un archivo PKL.

        Args:
            input_path (Path): Ruta del directorio que contiene el archivo.
            input_filename (str): Nombre del archivo pkl a cargar.

        Returns:
            pd.DataFrame: DataFrame con los datos cargados.

        Examples:
            >>> handler = DataHandler()
            >>> data = handler.load_data(Path("data/raw"), "dataset.csv")
        """
        # Construir la ruta completa al archivo
        full_input_path = input_path / input_filename
        logger.info(f"Loading data from {full_input_path}")

        try: 
            
            self.data = pd.read_pickle(full_input_path)
            logger.success(f"Data loaded successfully from {full_input_path}")
            return self.data
        except FileNotFoundError as e:
            logger.error(f"File not found: {full_input_path}")
            raise e
        except Exception as e:
            logger.error(f"An error occurred while loading data: {e}")      
            raise e

    def save_data(self, output_dir: Path, filename: str) -> None:
        """
        Guarda el DataFrame actual en un archivo CSV.

        Args:
            output_dir (Path): Directorio donde se guardará el archivo.
            filename (str): Nombre del archivo de salida.

        Returns:
            None

        Raises:
            Warning: Si no hay datos cargados en el DataFrame.

        Examples:
            >>> handler = DataHandler()
            >>> handler.save_data(Path("data/processed"), "processed_data.csv")
        """
        if self.data is None:
            logger.warning("No data to save. Load the data first.")
            return
        
        output_path = output_dir / filename
        logger.info(f"Saving data to {output_path}")
        
        self.data.to_csv(output_path, index=False)
        logger.success(f"Data saved successfully to {output_path}")

    def process_data(self) -> pd.DataFrame:
        """
        Procesa los datos eliminando valores faltantes y realiza análisis de missing values.

        Returns:
            pd.DataFrame: DataFrame procesado sin valores faltantes.

        Raises:
            Warning: Si no hay datos cargados para procesar.

        Examples:
            >>> handler = DataHandler()
            >>> processed_data = handler.process_data()
        """
        if self.data is None:
            logger.warning("No data loaded. Load data before processing.")
            return
        
        analyzer = MissingValueAnalyzer(self.data)
        missing_columns = analyzer.get_missing_columns()
        analyzer.display_missing_columns()

        # Manejo de valores extremos
        self.data.replace([np.inf, -np.inf], np.nan, inplace=True)  # Reemplazar infinitos por NaN
        self.data.dropna(inplace=True)
        logger.success("Data processed successfully.")
        return self.data
    
    
    def run(self, input_path: Path, input_filename: str, output_filename: str) -> 'DataHandler':
        """
        Ejecuta el pipeline completo de procesamiento de datos.

        Este método ejecuta secuencialmente la carga, procesamiento y
        guardado de datos.

        Args:
            input_path (Path): Ruta del directorio de entrada.
            input_filename (str): Nombre del archivo de entrada.
            output_filename (str): Nombre del archivo de salida.

        Returns:
            DataHandler: Instancia actual del DataHandler.

        Examples:
            >>> handler = DataHandler()
            >>> handler.run(Path("data/raw"), "input.csv", "output.csv")
        """
        self.load_data(input_path, input_filename)
        self.process_data()

        self.save_data(PROCESSED_DATA_DIR, output_filename)
        return self


class MissingValueAnalyzer:
    """
    Clase para analizar valores faltantes en un DataFrame.

    Esta clase proporciona métodos para identificar y mostrar información sobre
    columnas que contienen valores faltantes.

    Attributes:
        data (pd.DataFrame): DataFrame a analizar.
    """

    def __init__(self, data: pd.DataFrame):
        """
        Inicializa el analizador de valores faltantes.

        Args:
            data (pd.DataFrame): DataFrame a analizar.
        """
        self.data = data
    
    def get_missing_columns(self) -> pd.Series:
        """
        Identifica las columnas que contienen valores faltantes.

        Returns:
            pd.Series: Series con el conteo de valores faltantes por columna.

        Examples:
            >>> analyzer = MissingValueAnalyzer(df)
            >>> missing_cols = analyzer.get_missing_columns()
        """
        missing_flag = False
        missing_values = self.data.isnull().sum()
        porcentage_list = missing_values/len(self.data)*100
        for columna, valor in porcentage_list.items():
            print(f"{columna}: {valor:.3f} %")
            if valor != 0: missing_flag = True
        return missing_flag

    
    def display_missing_columns(self) -> None:
        """
        Muestra en consola las columnas con valores faltantes.

        Si no hay valores faltantes, muestra un mensaje indicándolo.

        Examples:
            >>> analyzer = MissingValueAnalyzer(df)
            >>> analyzer.display_missing_columns()
        """
        missing_flag = self.get_missing_columns()
        if missing_flag is False:
            print("No hay columnas con valores faltantes.")
        else:
            print("Columnas con valores faltantes:")
            # print(columns_with_missing)

    def statistics_resume(self):
        print("Estadisticas")
        return self.data.describe()


if __name__ == "__main__":
    with open('params.yaml', 'r') as f:
        params = yaml.safe_load(f)

    handler = DataHandler()
    handler.run(RAW_DATA_DIR, params['data']['input_filename'], params['data']['output_filename'])
