import os
import numpy as np
import scipy.optimize as opt
import matplotlib.pyplot as plt

def gradiente(theta, X, Y, lamb):
    """
        Calculate the new value of Theta with matrix
    """
    m = X.shape[0]  # m = 5K
    S = sigmoide_fun(np.matmul(X, theta))   # (5K,)
    diff = S - Y # Y.shape = (5K,)
    newTheta = (1 / m) * np.matmul(X.T, diff) + (lamb/m) * theta
    newTheta[0] -= (lamb/m) * theta[0]

    return newTheta

def sigmoide_fun(Z):
    return 1 / (1 + (np.exp(-Z)))

def coste(Theta, X, Y, lamb):
    """
        Calculates the J function of the cost
        of the Logistic Regresion    
    """
    m = X.shape[0]  # m = 5K
    S = sigmoide_fun(np.matmul(X, Theta)) # (5K,)
    Sum1 = np.dot(Y, np.log(S)) # 

    # This add is to dodge the log(0)
    Diff = (1 - S) + 0.00001
    Sum2 = np.dot((1 - Y), np.log(Diff))
    # First part
    Sum = Sum1 + Sum2
    Sum = (-1 / m) * Sum
    # Lambda part
    Sum3 = np.sum(np.power(Theta, 2))
    Sum += (lamb / (2 * m)) * Sum3

    return Sum 

def getEtiqueta(Y, etiqueta):
    """
    Devuelve el vector de booleanos para determinar
    que se trata de la etiqueta correcta
    """
    y_etiqueta = (np.ravel(Y) == etiqueta) * 1 # Vector de booleanos
    return y_etiqueta   # (numElems,)

def evalua(i, theta, X, Y, threshold=0.8): 
    S = sigmoide_fun(np.matmul(X, theta))
    pos = np.where(S >= threshold)
    neg = np.where(S < threshold)
    posExample = np.where(Y == 1)
    negExample = np.where(Y == 0)

    # intersect1d: sirve para coger añadir elementos al vector
    # cuando éstos sean iguales
    totalPos = np.intersect1d(pos, posExample).shape[0] / S.shape[0]
    totalNeg = np.intersect1d(neg, negExample).shape[0] / S.shape[0]
    # El porcentaje total sale de la cantidad de ejemplos identificados
    # como la etiqueta y de la cantidad que ha identifiado que no son la etiqueta
    return totalPos + totalNeg

def getLabelMatrixY(y, num_labels):
    """
    Genera una matriz Y selecciona con 0s y 1s
    el grado de performance de cada usuario
    """
    # Matriz para identificar qué performance posee cada usuario
    perfmY = np.zeros((y.shape[0], num_labels)) # (numElems, numLabels)

    for i in range(num_labels):
        perfmY[:, i] = getEtiqueta(y, i)

    # Las etiquetas de los 0's
    perfmY[:, 0] = getEtiqueta(y, 4)

    return perfmY

def oneVsAll(X, y, Xval, yval, num_labels=4, iters=8, initReg=0.01):
    """
    Entrenamiento de varios clasificadores por regresión logística
    """
    numFeatures = X.shape[1]
    # Matriz de parámetros theta
    theta = np.zeros((num_labels, numFeatures))
    perfmY = getLabelMatrixY(y, num_labels)
    # Matriz de etiquetas yval
    ylv = getLabelMatrixY(yval, num_labels)

    # Entrenamiento
    validation = np.zeros(num_labels)
    bestScore = np.zeros(num_labels)
    bestReg = np.zeros(num_labels)
    bestTheta = np.zeros((num_labels, numFeatures))

    for i in range(num_labels):
        for j in range(iters):
            reg = initReg * 3**j
            # Se entrena con las X
            result = opt.fmin_tnc(func = coste, x0 = theta[i, :], fprime = gradiente,
                    args=(X, perfmY[:, i], reg))
            theta[i, :] = result[0]

            # Se evalua con las Xval
            # Matriz de etiquetas yval
            validation[i] = evalua(i, theta[i, :], Xval, ylv[:, i])
            if(validation[i] > bestScore[i]):
               bestScore[i] = validation[i] 
               bestReg[i] = reg
               bestTheta[i, :] = theta[i, :]

    return bestScore, bestReg, bestTheta

def logisticRegresionClassification(X, y, Xval, yval, Xtest, ytest, dataset):
    """
    Clasificación de los datos mediante Regresión Logística
    """
    print("Entrenando sistema de clasificación de bodyPerfomance")
    bestScore, bestReg, bestTheta = oneVsAll(X, y, Xval, yval)
    
    #----------------TEST-DATA-----------------#
    numTest = Xtest.shape[0]
    num_labels = 4
    yp = np.zeros((numTest, num_labels))
    for i in range(num_labels):
        Z = np.dot(bestTheta[i, :], Xtest.T)
        yp[:, i] = sigmoide_fun(Z)
     

    #----------------PRINT-DATA----------------#
    os.system('cls')
  
    print("\n------------------------------------------")
    print("\nMejores resultados del entrenamiento")
    for i in range(num_labels):
        str1 = f"Mejor reg {chr(i + 65)}: {bestReg[i]}"
        str2 = f"Evaluación {chr(i + 65)}: {bestScore[i] * 100}%" 
        print(str1 + " - " + str2)
    
    print(f"Error: {1 - bestScore.mean()}")
    print("Evaluación media: ", bestScore.mean() * 100)
    print("\n------------------------------------------")

    print("\nComprobación de parámetros con ytest, Xtest")
    testResults = np.zeros(num_labels)
    # Matriz ytest de etiquetas
    ylt = getLabelMatrixY(ytest, num_labels)

    for i in range(num_labels):
        testResults[i] = evalua(i, bestTheta[i, :], Xtest, ylt[:, i])
        print(f"Evaluación {chr(i + 65)}: {testResults[i] * 100}%")
    print("Evaluación media test: ", testResults.mean() * 100)

    print("\n------------------------------------------")
    print("Success")
   
    #----------------GRAPHICS----------------#


    return 0

#---------------GRAPHICS---------------#