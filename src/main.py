from tkinter import TRUE
from svmPerformance import svmClassification as svm
from logisticRegresion import bestPairClassification as pairLog, logisticRegresionClassification as log
from neuralNetwork import neuralNetworkClassification as red_neu
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