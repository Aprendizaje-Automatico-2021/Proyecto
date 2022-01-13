from svmPerformance import *
from initData import *
import matplotlib.pyplot as plt

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