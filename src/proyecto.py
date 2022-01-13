from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
#--------------------LOADING-DATA--------------------#
def loadData():
    """
    Lectura del dataset que contiene los datos de enetrenamiento del sistema
    """

    # Carga de los datos del csv y conversión a diccionario sencillo de atributos
    valores = pd.read_csv("./src/assets/bodyPerformance.csv")
    valores.drop_duplicates(inplace=True)
    valores.columns = valores.columns.str.strip().str.replace(' ', '_')
    valores.rename(columns={'class':'Class', 'body_fat_%':'body_fat', 
    'sit-ups_counts':'sit_ups_counts'}, inplace=True)
    
    # Carga de atributos en matrices de numpy
    features = np.zeros([valores.shape[0], valores.shape[1] - 1])
    features[:, 0] = valores['age']

    # Se consideran hombres = 1, mujeres = 0
    features[:, 1] = valores['gender'] == 'M'
    features[:, 2] = valores['height_cm']
    features[:, 3] = valores['weight_kg']
    features[:, 4] = valores['body_fat']
    features[:, 5] = valores['diastolic']
    features[:, 6] = valores['systolic']
    features[:, 7] = valores['gripForce']
    features[:, 8] = valores['sit_and_bend_forward_cm']
    features[:, 9] = valores['sit_ups_counts']
    features[:, 10] = valores['broad_jump_cm']

    # Carga de los resultados de cada ejemplo de entrenamiento
    results = np.zeros(valores.shape[0])
    test = valores['Class'].values
    max = results.shape[0]
    for i in range(max):
        results[i] = ord(test[i])
        
    return features, results

def selectingData(allX, allY):
    """
    Seleccion de los datos de entrenamiento, 
    de validación y de pruebas
    """
    # Se cogerá un 60% de los datos para entrenar
    X = allX[:int(0.6 * np.shape(allX)[0])]
    y = allY[:int(0.6 * np.shape(allY)[0])]

    # Después se coge un 20% para evaluar
    Xval = allX[int(0.6 * np.shape(allX)[0]) : int(0.8 * np.shape(allX)[0])]
    yval = allY[int(0.6 * np.shape(allY)[0]) : int(0.8 * np.shape(allX)[0])]

    # Por último, el 20% resrtante para testing
    Xtest = allX[int(0.8 * np.shape(allX)[0]):]
    ytest = allY[int(0.8 * np.shape(allX)[0]):]

    return X, y, Xval, yval, Xtest, ytest

#--------------OTROS---------------#
def selectParameters(X, y, Xval, yval, initialValue, iter):
    reg = initialValue
    sigma = initialValue

    bestScore = 0
    bestSvm = 0
    bestReg = reg
    bestSigma = sigma
    
    for i in range(iter):
        reg = initialValue * 3**i
        for j in range(iter):
            sigma = initialValue * 3**j
            svm = SVC(kernel='rbf', C=reg, gamma= 1 / (2 * sigma**2))
            svm.fit(X, y.ravel())
            accuracy = accuracy_score(yval, svm.predict(Xval))
            if(accuracy > bestScore):
                bestSvm = svm
                bestScore = accuracy
                bestReg = reg
                bestSigma = sigma

    return bestSvm, bestReg, bestSigma, bestScore

#--------------DRAW---------------#

#--------------AA---------------#
def svmClassification(X, y, Xval, yval, Xtest, ytest):
    print("Entrenando sistema de clasificacion de bodyPerfomance")
    initialValue = 0.01
    svm, reg, sigma, bestScore = selectParameters(X, y, Xval, yval, initialValue, 8)

    # Matrices de prediccion de cada conjunto de datos - Aún no se usa, pero puede ser interesante para algo
    yp = svm.predict(X)
    yvalp = svm.predict(Xval)
    ytestp = svm.predict(Xtest)
    
    performance = [ord("A"), ord("B"), ord("C"), ord("D")]

    # Precision del svm
    print(f"Error: {1 - bestScore}")
    print(f"Mejor C: {reg}")
    print(f"Mejor sigma: {sigma}")
    testScore = svm.score(X, y)
    print(f"Precisión sobres los datos de entrenamiento: {testScore * 100}%")
    testScore = svm.score(Xval, yval)
    print(f"Precisión sobres los datos de evaluación: {testScore * 100}%")
    testScore = svm.score(Xtest, ytest)
    print(f"Precisión sobres los datos de testing: {testScore * 100}%")
    print("Success")

def main():
    # Carga de los datos en un diccionario dataset
    allX, allY = loadData()
    # Fragmentación del dataset
    X, y, Xval, yval, Xtest, ytest = selectingData(allX, allY)

    # Clasificacion de los datos mediante SVM
    svmClassification(X, y, Xval, yval, Xtest, ytest)
    # Clasificacion de los datos mediante Redes Neuronales
    # Clasificacion de los datos mediante Regresión logistica
    # Clasificacion de los datos mediante Regresión Lineal
    
    return 0
    
main()