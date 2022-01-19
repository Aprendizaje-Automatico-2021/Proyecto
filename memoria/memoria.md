# BodyPerfomance Classification
| Autores|email|
|-|-|
|Clara Daniela Sima|csima@ucm.es|
|Stiven Arias Giraldo|starias@ucm.es|

Resultados del proyecto final de la asignatura de **``Aprendizaje Automático y Minería de Datos``** del grado de **``Desarrollo de Videojuegos``** de la **``Universidad Complutense de Madrid``**

# Descripción
La base de datos que hemos escogido está compuesta por múltiples atributos que describen características físicas de diversas personas. Por ser más concretos, contamos con ``13393 ejemplos de entrenamiento``, es decir, 13393 personas, de las cuales tenemos ``11 atributos`` y un resultado para cada una. 

Esto quiere decir que nuestro dataset está compuesto por una ``matriz de (13393, 12) dimensiones``, donde la última columna nos indica la calificación de un _``entrenamiento físico``_ de cada persona en función del resto de atributos, pudiendo ser **``A, B, C, D``** dicha calificación, siendo **A** el mejor resultado posible.

![image](https://user-images.githubusercontent.com/47497948/149372136-b6356d03-dd10-4deb-a039-9ddb2182d937.png)

![image](https://user-images.githubusercontent.com/47497948/149371625-e32eefcb-1069-4653-bc3d-71cfeb01bf75.png)

Así pues, la idea principal de este proyecto es aprovechar los diferentes algoritmos de aprendizaje automático realizados durante el curso, procesando los diferentes datos para poder determinar qué sistema de aprendizaje resulta más óptimo para clasificar correctamente el _dataset_.
Cada uno de los sistemas tendrán que predecir cuál ha sido el grado de eficiencia del usuario (A, B, C o D) para finalmente comparar dicha predicción con los datos reales.

Los sistemas son los siguientes: *SVM, Regresión logística y Redes Neuronales*

# Inicialización y selección de los datos de entrenamiento

Dentro del **main.py** tenemos el lanzador del programa, donde importamos los diferentes métodos de los sistemas de clasificación.

Para agilizar el proceso de selección del sistema tenemos una variable **_system_** para seleccionar el sistema que se desea probar:

```py
from svmPerformance import svmClassification as svm
from logisticRegresion import bestPairClassification as pairLog, logisticRegresionClassification as log
from redesNeuronales import neuralNetworkClassification as red_neu
from initData import *


def main(system=0):
    # Carga de los datos en un diccionario dataset
    allX, allY, dataset = loadData()
    # Fragmentación del dataset
    X, y, Xval, yval, Xtest, ytest = selectingData(allX, allY)

    if system == 0:
        # Clasificacion de los datos mediante SVM
        svm(X, y, Xval, yval, Xtest, ytest)
    elif system == 1:
        # Clasificacion de los datos mediante Regresión logistica
        log(X, y, Xval, yval, Xtest, ytest)
        # Clasificacion de los datos mediante Regresión logistica
        # escogiendo el mejor par de atributos
        pairLog(X, y, Xval, yval, Xtest, ytest, dataset)
    elif system == 2:
        # Clasificacion de los datos mediante Redes Neuronales
        red_neu(X, y, Xval, yval, Xtest, ytest)
    return 0
    
system = 1
main(system)
```
En primer lugar, se cargan los datos con **``loadData``**, donde se lee el **.csv** y se hacen las correspondientes conversiones de datos para obtener matrices de **floats**. Además, tenemos el método **``plot_coor``**, el cual nos sirve para dibujar una gráfica que muestra la matriz de correlaciones entre cada par de datos (exceptuando ``gender`` y ``Class``, ya que es lo que devuelve **``df.corr()``**, además que nuestro sistema va a clasificar los valores en función de Class). Esta matriz de correlaciones nos parece interesante porque con ella se pueden tener otro tipo de observaciones, sin embargo, no es del todo relevante para nuestro sistema de clasificación. 

```py
from tarfile import DIRTYPE
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def plot_corr(df,size=10):
    """
    Dibuja una matriz de correlaciones por filas y columnas
    para cada par de datos del dataset
    """

    corr = df.corr()
    fig, ax = plt.subplots(figsize=(size, size))
    im = ax.matshow(corr, cmap="RdPu")
    plt.colorbar(im, label="Correlación entre los atributos")
    plt.xticks(range(len(corr.columns)), corr.columns, rotation=20)
    plt.yticks(range(len(corr.columns)), corr.columns)
        
    values = corr.values
    for (i, j), z in np.ndenumerate(values):
        ax.text(j, i, '{:0.1f}'.format(z), fontsize=14, ha='center', va='center')

    plt.show()

def loadData(disp_corr=False):
    """
    Lectura del dataset que contiene los datos de enetrenamiento del sistema
    """
    # Carga de los datos del csv y conversión a diccionario sencillo de atributos
    dataset = pd.read_csv("./src/assets/bodyPerformance.csv")
    dataset.drop_duplicates(inplace=True)
    dataset.columns = dataset.columns.str.strip().str.replace(' ', '_')
    dataset.rename(columns={'class':'Class', 'body_fat_%':'body_fat', 
    'sit-ups_counts':'sit_ups_counts', 'sit_and_bend_forward_cm':'sit_bend_forw_cm'}, inplace=True)
    
    # El número de elementos del data set es de 14K * 12, por tanto, 
    # para reducir el tiempo de Debug del programa se va a elegir un grupo reducido
    rows = int(dataset.shape[0])
    cols = int(dataset.shape[1] - 1)

    # Muestra el gráfico de las correlaciones
    plot_corr(dataset)

    # Carga de atributos en matrices de numpy
    features = np.array(dataset.values[:rows, :cols])
    # Se consideran hombres = 1, mujeres = 0
    features[:, 1] = (dataset['gender'][:rows] == 'M') * 1
    features = features.astype(np.double)

    # Carga de los resultados de cada ejemplo de entrenamiento
    results = np.zeros(rows)  
    test = dataset['Class'].values
    for i in range(rows):
        results[i] = ord(test[i]) - 64
        
    return features, results, dataset
```

Hemos separado los datos en 3 grupos: ``entrenamiento``, ``validación`` y ``testing`` donde la proporción de cada grupo es ``60%``, ``20%`` y ``20%`` del número total de datos respectivamente.

```py
def selectingData(allX, allY):
    """
    Seleccion de los datos de entrenamiento, 
    de validación y de pruebas
    """
    # Se cogerá un 60% de los datos para entrenar
    X = normalizeMatrix(allX[:int(0.6 * np.shape(allX)[0])])
    y = allY[:int(0.6 * np.shape(allY)[0])]

    # Después se coge un 20% para evaluar
    Xval = normalizeMatrix(allX[int(0.6 * np.shape(allX)[0]) : int(0.8 * np.shape(allX)[0])])
    yval = allY[int(0.6 * np.shape(allY)[0]) : int(0.8 * np.shape(allX)[0])]

    # Por último, el 20% resrtante para testing
    Xtest = normalizeMatrix(allX[int(0.8 * np.shape(allX)[0]):])
    ytest = allY[int(0.8 * np.shape(allX)[0]):]

    return X, y, Xval, yval, Xtest, ytest

def normalizeMatrix(X):
    """
    Normaliza el array X
    xi = (xi - ui) / si
    """
    matriz_normal = np.empty_like(X)

    # La media y la varianza de cada columna
    u = np.mean(X, axis=0)
    s = np.std(X, axis=0)

    matriz_normal = (X - u) / s
    return matriz_normal
```

En los valores **_X_** guardamos los atributos de cada ejemplo de entrenamiento y en los valores **_Y_**, los resultados.

La idea de esta distinción de valores es utilizar los **``datos de entrenamiento``** para entrenar el sistema utilizado. A partir de los valores generados por el sistema, se utilizarán los **``datos de validación``** para verificar que los valores obtenidos son realmente óptimos y, finalmente, con los **``datos de testing``** se pone a prueba el sistema al completo.

# Sistema SVM
En primer lugar, realizamos la clasificación de los datos mediante **SVM** (Support Vector Machine).

Para realizar el entrenamiento con SVM hay que tener en cuenta 2 parámetros: **``initialValue``**, **``iters``**.

- **initialValue**: sirve para inicializar tanto el parámetro de regularización ``C`` y el parámetro ``sigma``. Ambos valores serán usados en la función **SVC**, la cual está basada en un kernel gaussiano ``rbf`` y se irán modificando a través del bucle de entrenamiento.
  
- **iters**: es el número de iteraciones que habrá para realizar el entrenamiento de los valores ``C`` y ``sigma``, es decir, en total habrá **``iters * iters``** iteraciones para el entrenamiento.

Durante este proceso, vamos guardando los mejores valores del entrenamiento en función de la puntuación obtenida con los datos de validación ``Xval, yval``.

```py
def svmClassification(X, y, Xval, yval, Xtest, ytest):
    """
    Clasificación de losd atos mediantes SVM
    """
    print("Entrenando sistema de clasificacion de bodyPerfomance")
    initialValue = 0.0001
    iters = 20
    XX = np.insert(X, 0, 1, axis = 1)
    XXval = np.insert(Xval, 0, 1, axis = 1)
    XXtest = np.insert(Xtest, 0, 1, axis = 1)

    svm, params, bestIndex, bestScore, eTraining, eValidation = selectParameters(XX, y, XXval, yval, initialValue, iters)

    reg = params[0, bestIndex[0]]
    sigma = params[1, bestIndex[1]]

    # Precision del svm
    print(f"Error: {1 - bestScore}")
    print(f"Mejor reg: {reg}")
    print(f"Mejor sigma: {sigma}")
    testScore = np.zeros(3)
    testScore[0] = svm.score(XX, y)
    print(f"Precisión sobres los datos de entrenamiento: {testScore[0] * 100}%")
    testScore[1] = svm.score(XXval, yval)
    print(f"Precisión sobres los datos de evaluación: {testScore[1] * 100}%")
    testScore[2] = svm.score(XXtest, ytest)
    print(f"Precisión sobres los datos de testing: {testScore[2] * 100}%")
    print(f"Precisión media: {testScore.mean() * 100}%")
    print("Success")

    drawGraphics(XX, y, XXval, yval, XXtest, ytest, svm, params, eTraining, eValidation, bestIndex)
```

Finalmente, no solo utilizamos ``Xtest, ytest`` para realizar la comprobación de resultados, aunque sus datos serían los más relevantes, sino que comprobamos la puntuación de cada grupo de datos, imprimiendo los cálculos en consola y mostrándolos mediante gráficas.

## SVM Resultados