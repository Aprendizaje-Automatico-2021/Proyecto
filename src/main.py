from svmPerformance import svmClassification as svm
from logisticRegresion import logisticRegresionClassification as log
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
        print("Work in progress...")

    return 0
    
system = 1
main(system)