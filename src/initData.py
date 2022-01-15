import pandas as pd
import numpy as np

def loadData():
    """
    Lectura del dataset que contiene los datos de enetrenamiento del sistema
    """
    # Carga de los datos del csv y conversión a diccionario sencillo de atributos
    dataset = pd.read_csv("./src/assets/bodyPerformance.csv")
    dataset.drop_duplicates(inplace=True)
    dataset.columns = dataset.columns.str.strip().str.replace(' ', '_')
    dataset.rename(columns={'class':'Class', 'body_fat_%':'body_fat', 
    'sit-ups_counts':'sit_ups_counts'}, inplace=True)
    
    # El número de elementos del data set es de 14K * 12, por tanto, 
    # para reducir el tiempo de Debug del programa se va a elegir un grupo reducido
    rows = int(dataset.shape[0] * 0.05)
    cols = int(dataset.shape[1] - 1)

    # Carga de atributos en matrices de numpy
    features = np.array(dataset.values[:rows, :cols])
    features[:, 1] = dataset['gender'][:rows] == 'M' # Se consideran hombres = 1, mujeres = 0
    features = features.astype(float)

    # Carga de los resultados de cada ejemplo de entrenamiento
    results = np.zeros(rows)
    test = dataset['Class'].values
    for i in range(rows):
        results[i] = ord(test[i]) - 64
        
    return features, results

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