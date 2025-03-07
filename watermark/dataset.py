from pathlib import Path

import numpy as np
import pandas as pd
import typer
import cv2
import yaml
from loguru import logger
from tqdm import tqdm

from torchvision import transforms

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
    def get_extended_data(self,list_data, strides, limit_len):
        sequences_list = []
        for j, task in enumerate(list_data): # Recorrer la lista que contiene los datos de las tareas
            sequences = [] # Lista para guardar las secuencias extendidas de cada tarea
            for data in task: # Recorrer las señales de cada tarea
                if len(data) <= limit_len: # Si la señal es menor o igual a la longitud límite no se hace ningún ajuste
                 sequences.append(data)
                 continue
                else: # Si la señal es mayor a la longitud límite se obtienen las secuencias extendidas
                 i=0
                 while limit_len + i*strides[j] <= len(data): # Mientras la longitud de la secuencia más el desplazamiento no sea mayor a la longitud de la señal
                    sequences.append(data[i*strides[j]:limit_len+i*strides[j]]) # Se obtiene la secuencia extendida
                    i+=1 # Se incrementa el contador hasta que el límite de la señal sea alcanzado
            sequences_list.append(sequences) # Se guarda la lista de secuencias extendidas de la tarea
        return sequences_list
    
    # Función para convertir un trazo en imagen
    def trazo_a_imagen_v2(self,x_trazo, y_trazo, image_size=224):
       # Crear una imagen en blanco
       img = np.zeros((image_size, image_size), dtype=np.uint8)

       # Normalizar las coordenadas para que encajen dentro de la imagen
       x_min, x_max = np.min(x_trazo), np.max(x_trazo)
       y_min, y_max = np.min(y_trazo), np.max(y_trazo)

       # Redimensionar coordenadas a un rango de 0 a 223 (para 224x224 píxeles)
       x_scaled = np.clip(((x_trazo - x_min) / (x_max - x_min)) * (image_size - 1), 0, image_size - 1)
       y_scaled = np.clip(((y_trazo - y_min) / (y_max - y_min)) * (image_size - 1), 0, image_size - 1)

       # Dibujar el trazo en la imagen
       for i in range(1, len(x_scaled)):
            cv2.line(img, (int(x_scaled[i - 1]), int(y_scaled[i - 1])), 
                  (int(x_scaled[i]), int(y_scaled[i])), 255, 1)

       return img
    
    # Función para preprocesar una imagen y convertirla a tensor
    def preprocesar_imagen_tensor(self, img):
        # Si la imagen es un objeto PIL, convertirla a numpy array
        if not isinstance(img, np.ndarray):
            img = np.array(img)
        
        # Verificar la dimensionalidad de la imagen
        if img.ndim == 2:
            # La imagen es en escala de grises (2D), replicar el canal para obtener RGB
            img_rgb = np.stack([img] * 3, axis=-1)
        elif img.ndim == 3:
            if img.shape[2] == 1:
                # Imagen con 1 canal, replicarlo para RGB
                img_rgb = np.concatenate([img, img, img], axis=-1)
            elif img.shape[2] == 3:
                # La imagen ya es RGB, no se modifica
                img_rgb = img
            else:
                raise ValueError(f"La imagen tiene {img.shape[2]} canales, se esperaba 1 o 3.")
        else:
            raise ValueError("Formato de imagen inesperado. Se esperaba una imagen 2D o 3D.")

        # Transformaciones necesarias para ResNet101
        transform = transforms.Compose([
            transforms.ToPILImage(),       # Convertir la imagen de NumPy a PIL
            transforms.Resize((224, 224)),   # Redimensionar a 224x224
            transforms.ToTensor(),           # Convertir a tensor
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225])  # Normalización
        ])

        # Aplicar las transformaciones
        img_tensor = transform(img_rgb)
        img_tensor = img_tensor.unsqueeze(0)  # Añadir la dimensión del batch: [1, 3, 224, 224]

        return img_tensor
        

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
        missing_values = self.data.isnull().sum()
        porcentage_list = missing_values/len(self.data)*100
        for columna, valor in porcentage_list.items():
            print(f"{columna}: {valor:.3f} %")

    
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
