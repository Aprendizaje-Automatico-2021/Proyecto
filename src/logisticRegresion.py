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
    Sum1 = np.dot(Y, np.log(S + 0.000001)) # 

    # This add is to dodge the log(0)
    Diff = (1 - S) + 0.00001
    Sum2 = np.dot((1 - Y), np.log(Diff + 0.000001))
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

def oneVsAll(X, y, Xval, yval, initReg=0.01, iters=8, num_labels=4):
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
                    args=(X, perfmY[:, i], reg), disp=0)
            theta[i, :] = result[0]

            # Se evalua con las Xval
            # Matriz de etiquetas yval
            validation[i] = evalua(i, theta[i, :], Xval, ylv[:, i])
            if(validation[i] > bestScore[i]):
               bestScore[i] = validation[i] 
               bestReg[i] = reg
               bestTheta[i, :] = theta[i, :]

    return bestScore, bestReg, bestTheta

def logisticRegresionClassification(X, y, Xval, yval, Xtest, ytest):
    """
    Clasificación de los datos mediante Regresión Logística
    """
    print("Entrenando sistema de clasificación de bodyPerfomance")
    bestScore, bestReg, bestTheta = oneVsAll(X, y, Xval, yval)
    
    #----------------PRINT-DATA----------------#
    num_labels = 4

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

def calculatePair(X, Xval, i, j):
    """
    Devuelve el par correspondiente de datos
    """
    numFeatures = 2
    pair = np.zeros((X.shape[0], numFeatures))
    pair[:, 0] = X[:, i]
    pair[:, 1] = X[:, j]

    pair_val = np.zeros((Xval.shape[0], numFeatures))
    pair_val[:, 0] = Xval[:, i]
    pair_val[:, 1] = Xval[:, j]

    return pair, pair_val

def bestPairClassification(X, y, Xval, yval, Xtest, ytest, dataset):
    """
    Clasificación de los datos mediantes Regresión Logística
    para cada par de datos, escogiendo los 2 mejores atributos
    que clasifiquen el resultado
    """
    print("Entrenando sistema de clasificación de bodyPerfomance")
    bestScore = np.zeros(1)
    bestPair = 0
    bestReg = 0
    bestTheta = 0
    f1, f2 = 0, 0
    
    initReg = 0.0003
    n = 1
    for i in range(X.shape[1]):
        for j in range(i + 1, X.shape[1]):
            pair, pair_val = calculatePair(X, Xval, i, j)
            score, reg, theta = oneVsAll(pair, y, pair_val, yval, initReg, 10)
            if(score.mean() > bestScore.mean()):
                bestScore = score
                bestPair = [pair, pair_val]
                bestReg = reg
                bestTheta = theta
                f1, f2 = i, j

    #----------------PRINT-DATA----------------#
    num_labels = 4
    features = dataset.columns[f1] + ", " + dataset.columns[f2]
    
    print("\n------------------------------------------")
    
    print("\nMejores resultados del entrenamiento")
    print(f"Mejor par de atributos: {features}")
    print(f"Mejor reg: {bestReg}")
    for i in range(num_labels):
        str1 = f"Mejor reg {chr(i + 65)}: {bestReg[i]}"
        str2 = f"Evaluación {chr(i + 65)}: {bestScore[i] * 100}%" 
        print(str1 + " - " + str2)
    
    print("Evaluación media: ", bestScore.mean() * 100)
    print(f"Error: {1 - bestScore.mean()}")

    print("\n------------------------------------------")

    print("\nComprobación de parámetros con ytest, Xtest")
    testResults = np.zeros(num_labels)
    # Matriz ytest de etiquetas
    numFeatures = 2
    ylt = getLabelMatrixY(ytest, num_labels)
    pair_test = np.zeros((Xtest.shape[0], numFeatures))
    pair_test[:, 0] = Xtest[:, f1]
    pair_test[:, 1] = Xtest[:, f2]

    for i in range(num_labels):
        testResults[i] = evalua(i, bestTheta[i, :], pair_test, ylt[:, i])
        print(f"Evaluación {chr(i + 65)}: {testResults[i] * 100}%")
    print("Evaluación media test: ", testResults.mean() * 100)

    print("\n------------------------------------------")

    return 0
#---------------GRAPHICS---------------#
def drawPerformanceGraph(X, Y):
    """
    Dibuja los datos en función del grado de Perfomance 
    que se quiera predecir
    """
    plt.figure()
    # Results of admitted
    pos = np.where (Y == 1)
    # Result of not admitted
    neg = np.where (Y == 0)

    plt.scatter(X[pos, 0], X[pos, 1], marker='+', c='k', label="y = 1")
    plt.scatter(X[neg, 0], X[neg, 1], c='#c6ce00', label="y = 0")
    plt.legend(loc='lower left')
    plt.show()

def lineal_fun_graph(X, Theta, poly, limit=0.8):

    x1_min, x1_max = X[:, 0].min(), X[:, 0].max()
    x2_min, x2_max = X[:, 1].min(), X[:, 1].max()

    xx1, xx2 = np.meshgrid(np.linspace(x1_min, x1_max),
                            np.linspace(x2_min, x2_max))

    poly = poly.fit_transform(np.c_[xx1.ravel(),
                        xx2.ravel()])
    h = sigmoide_fun(np.dot(poly, Theta))

    h = h.reshape(xx1.shape)

    plt.contour(xx1, xx2, h, [limit], linewidths=1, colors='b')

    return 0