from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
import matplotlib.pyplot as plt
import numpy as np

def selectParameters(X, y, Xval, yval, initialValue, iter):
    """
    Entrena el SVM eligiendo los mejores resultados en función del
    término de regularización "reg" y el valor sigma necesarios en la 
    ecuación gaussiana. Finalmente devuelve el los mejores resultados de:
    svm, reg, sigma y score
    """
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

def svmClassification(X, y, Xval, yval, Xtest, ytest):
    """
    Clasificación de losd atos mediantes SVM
    """
    print("Entrenando sistema de clasificacion de bodyPerfomance")
    initialValue = 0.01
    iters = 8
    svm, reg, sigma, bestScore = selectParameters(X, y, Xval, yval, initialValue, iters)

    # Precision del svm
    print(f"Error: {1 - bestScore}")
    print(f"Mejor reg: {reg}")
    print(f"Mejor sigma: {sigma}")
    testScore = svm.score(X, y)
    print(f"Precisión sobres los datos de entrenamiento: {testScore * 100}%")
    testScore = svm.score(Xval, yval)
    print(f"Precisión sobres los datos de evaluación: {testScore * 100}%")
    testScore = svm.score(Xtest, ytest)
    print(f"Precisión sobres los datos de testing: {testScore * 100}%")
    print("Success")

    drawGraphics(X, y, Xval, yval, Xtest, ytest, svm)
    

#---------------GRAPHICS---------------#
def calculateUsers(y):
    initLabel = 1
    
    users = np.zeros(4)
    users[0] = np.sum(y == initLabel)
    users[1] = np.sum(y == initLabel + 1)
    users[2] = np.sum(y == initLabel + 2)
    users[3] = np.sum(y == initLabel + 3)

    return users

def drawGraphics(X, y, Xval, yval, Xtest, ytest, svm):
    #-------------------------------------------------------------------#
    # Barras de comparación de los datos de entrenamiento
    yp = svm.predict(X)
    realUsers = calculateUsers(y)
    predictUsers = calculateUsers(yp)
    drawBarsComparision(realUsers, predictUsers)
    # Barras de comparación de los datos de validación
    yvalp = svm.predict(Xval)
    realUsers = calculateUsers(yval)
    predictUsers = calculateUsers(yvalp)
    drawBarsComparision(realUsers, predictUsers)
    # Barras de comparación de los datos de testing
    ytestp = svm.predict(Xtest)
    realUsers = calculateUsers(ytest)
    predictUsers = calculateUsers(ytestp)
    drawBarsComparision(realUsers, predictUsers)
    #-------------------------------------------------------------------#

def drawBarsComparision(realUsers, predictUsers):
    """
    Dibuja las gráficas comparativas en forma de barras entre los datos
    reales y la predicción
    """
    plt.figure
    performance = ["A", "B", "C", "D"]

    # Ejemplo de barras individuales
    #plt.bar(performance, realUsers)
    #plt.title('Datos reales')
    #plt.xlabel('Tipos de grados de entrenamiento')
    #plt.ylabel('Número de personas por grado')
    #plt.show()
    
    X = np.arange(len(performance))
    Y1 = realUsers
    Y2 = predictUsers
    plt.bar(X, +Y1, facecolor='#9999ff', edgecolor='white')
    plt.bar(X, -Y2, facecolor='#ff9999', edgecolor='white')
    
    i = 0
    for x, y in zip(X, Y1):
        plt.text(x, y - 5.0, '%.0f' % y, ha='center', va='top')
        plt.text(x, 5.0, performance[i], ha='center', va='bottom')
        i += 1

    for x, y in zip(X, Y2):
        plt.text(x, -y + 5.0, '%.0f' % y, ha='center', va='bottom')

    plt.xticks([])
    plt.yticks([])
    plt.xlabel('Tipos de grados de entrenamiento')
    plt.ylabel('Número de personas por grado')

    plt.show()
