from svmPerformance import *
from initData import *

def main():
    # Carga de los datos en un diccionario dataset
    allX, allY = loadData()
    # Fragmentación del dataset
    X, y, Xval, yval, Xtest, ytest = selectingData(allX, allY)

    # Clasificacion de los datos mediante SVM
    svmClassification(X, y, Xval, yval, Xtest, ytest)
    # Clasificacion de los datos mediante Redes Neuronales - Clara
    # Clasificacion de los datos mediante Regresión logistica - Stiven
    
    return 0
    
main()