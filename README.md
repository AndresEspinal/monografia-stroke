# **Predicción de Ocurrencia de Accidentes Cerebrovasculares**

En kaggle se encuentra la base de datos llamada Stroke Prediction Dataset que busca predecir según un número de variables si el paciente puede sufrir o no de un accidente cerebrovascular. Aquí se utilizaron herramientas de machine learning con la intención de predecir en un alto porcentaje y pueda ser de ayuda para futuros casos médicos que cumplan con esta condición.
En este proyecto se abordarán varias técnicas referentes a modelos supervisados como la regresión logística, knn, máquinas de soporte de vectores, random forest, árboles de decisiones, entre otros. Se usaron metodologías de validación como la división de datos de entrenamiento y validación y la validación cruzada y se usó el balanceo de clases de la variable respuesta debido a una considerable diferencia de registros positivos y negativos.
El proyecto está actualmente en curso y corresponde al trabajo de monografía de la Especialización en Analítica y Ciencia de Datos de la Universidad de Antioquia.
Para correr el proyecto se dejará instrucción con múltiples maneras para hacerlo.

## **Desde Kaggle:**

Para usar el código desde kaggle se debe utilizar un API token generado por su perfil de Kaggle para descargar la base de datos directamente desde el sitio web. Deben almacenar el token dentro de su espacio en Google Drive.
En la dirección web donde se encuentra el proyecto [Stroke Prediction Dataset](https://www.kaggle.com/datasets/fedesoriano/stroke-prediction-dataset) deben ir a su perfil que es el logo o foto que usas en el costado derecho en la parte de arriba de la pantalla y allí debes ir a settings y luego buscar el botón Create New Token y como se dijo anteriormente deben almacenarlo en su drive.
Corren este primer código para conectar a colab con su drive:

```python
from google.colab import drive
import os
drive.mount('/content/drive/')
```

Luego deben ejecutar este otro código:

```python
os.environ['KAGGLE_CONFIG_DIR'] = '/content/drive/MyDrive/' + input('Input the directory with your Kaggle json file: ') # Dejar input vacío en caso de que el archivo se encuentre en la raíz de Drive
!kaggle datasets download -d fedesoriano/stroke-prediction-dataset # Descarga del archivo comprimido
!unzip \*.zip && rm *.zip # Descomprensión y eliminación de cualquier archivo .zip
```

No es necesario poner nada, solo darle enter cuando salga el espacio para escribir.

## **Desde GitHub:**

Solo es necesario correr este código de abajo y luego en la parte de cargue de datos hacerlo en el que dice **Cargue de datos desde GitHub**.

```python
!git clone https://github.com/AndresEspinal/monografia-stroke.git
```

## **Desde Jupyter:**

Para abrirlo desde Jupyter solo hace falta descargar o ubicar el script en el pc y buscar en la carpeta que fue creado, allí solo debemos poner la base de datos que está localizada en el [Github](https://github.com/AndresEspinal/monografia-stroke) en la carpeta BD y luego descargarla. Cuando esté descargada es necesario ubicar la base de datos en la misma carpeta del script. Sigue los pasos normales y en la parte de cargue de datos debes hacerlo desde **Cargue de datos desde el Drive/Kaggle/Jupyter**.

## **Librerías:**

Las librerías que se van a usar o tener en cuenta son las siguientes:

```python
import pandas as pd
import io
import requests
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import matplotlib.patches as patches

from sklearn.impute import KNNImputer 
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder
from sklearn.neighbors import LocalOutlierFactor
from sklearn import svm, datasets
from sklearn.model_selection import GridSearchCV
from sklearn.random_projection import johnson_lindenstrauss_min_dim, GaussianRandomProjection, SparseRandomProjection
from sklearn.decomposition import PCA # Análisis de Componentes Principales
from sklearn.decomposition import KernelPCA # Kernel PCA
from sklearn.datasets import make_classification
from sklearn import metrics
from sklearn.metrics import precision_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
from sklearn.metrics import recall_score
from sklearn import svm
from sklearn import neighbors
from sklearn.naive_bayes import CategoricalNB
from sklearn.naive_bayes import ComplementNB
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_validate
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.metrics import calinski_harabasz_score
from sklearn.metrics import davies_bouldin_score
from sklearn.metrics import pairwise_distances
from sklearn.tree import plot_tree

from imblearn.over_sampling import RandomOverSampler

from scipy.special import entr
from scipy import special
from scipy.stats import spearmanr
from scipy.spatial.distance import euclidean

from collections import defaultdict

from yellowbrick.cluster import KElbowVisualizer
```
## **Bibliotecas:**

En caso de no tener todas las bibliotecas instaladas en su entorno, en la siguiente lista se mostrará las que fueron usadas o próximas a usar:
```python
!pip install pandas
!pip install seaborn
!pip install numpy
!pip install matplotlib
!pip install scikit-learn
!pip install imblearn
!pip install scipy
!pip install yellowbrick
```
