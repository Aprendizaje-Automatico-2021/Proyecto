from svmPerformance import svmClassification as svm
from logisticRegresion import logisticRegresionClassification as log
from redesNeuronales import neuralNetworkClassification as red_neu
from initData import *


def main(system=0):
    # Carga de los datos en un diccionario dataset
    allX, allY = loadData()
    # Fragmentación del dataset
    X, y, Xval, yval, Xtest, ytest = selectingData(allX, allY)

    if system == 0:
        # Clasificacion de los datos mediante SVM
        svm(X, y, Xval, yval, Xtest, ytest)
    elif system == 1:
        # Clasificacion de los datos mediante Regresión logistica - Stiven
        log(X, y, Xval, yval, Xtest, ytest)
    elif system == 2:
        # Clasificacion de los datos mediante Redes Neuronales - Clara
        red_neu(X, y, Xval, yval, Xtest, ytest)

    return 0
    
system = 2
main(system)