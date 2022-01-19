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
    params = np.zeros((2, iter))
    # La primera fila es para reg
    params[0, 0] = initialValue
    # La segunda fila es para sigma
    params[1, 0] = initialValue
    # Evolución de las puntuaciones para cada par de reg, sigma
    eTraining = np.zeros((iter, iter))
    eValidation = np.zeros((iter, iter))

    # Mejores resultados
    bestScore = 0
    bestSvm = 0
    bestIndex = np.zeros(2, dtype=int)

    for i in range(iter):
        params[0, i] = initialValue * 3**i
        for j in range(iter):
            params[1, j] = initialValue * 3**j
            svm = SVC(kernel='rbf', C= params[0, i], gamma= 1 / (2 * params[1, j]**2))
            svm.fit(X, y.ravel())
            eTraining[i, j] = svm.score(X, y)
            eValidation[i, j] = svm.score(Xval, yval)
            if(eValidation[i, j] > bestScore):
                bestSvm = svm
                bestScore = eValidation[i, j]
                bestIndex[0] = i
                bestIndex[1] = j

    return bestSvm, params, bestIndex, bestScore, eTraining, eValidation

def svmClassification(X, y, Xval, yval, Xtest, ytest):
    """
    Clasificación de losd atos mediantes SVM
    """
    print("Entrenando sistema de clasificacion de bodyPerfomance")
    initialValue = 0.0001
    iters = 20
    XX = np.insert(X, 0, 1, axis = 1)
    XXval = np.insert(Xval, 0, 1, axis = 1)
    XXtest = np.insert(Xtest, 0, 1, axis = 1)

    svm, params, bestIndex, bestScore, eTraining, eValidation = selectParameters(XX, y, XXval, yval, initialValue, iters)

    reg = params[0, bestIndex[0]]
    sigma = params[1, bestIndex[1]]

    # Precision del svm
    print(f"Error: {1 - bestScore}")
    print(f"Mejor reg: {reg}")
    print(f"Mejor sigma: {sigma}")
    testScore = np.zeros(3)
    testScore[0] = svm.score(XX, y)
    print(f"Precisión sobres los datos de entrenamiento: {testScore[0] * 100}%")
    testScore[1] = svm.score(XXval, yval)
    print(f"Precisión sobres los datos de evaluación: {testScore[1] * 100}%")
    testScore[2] = svm.score(XXtest, ytest)
    print(f"Precisión sobres los datos de testing: {testScore[2] * 100}%")
    print(f"Precisión media: {testScore.mean() * 100}%")
    print("Success")

    drawGraphics(XX, y, XXval, yval, XXtest, ytest, svm, params, eTraining, eValidation, bestIndex)
    

#---------------GRAPHICS---------------#

def drawParamSelection(params, eTraining, eValidation, bestIndex):
    """
    Compara la evolución de los datos de entrenamiento y de
    validación con las diferentes selecciones de los parámetros sigma y reg 
    """
    iter = bestIndex[0] + 1
    for i in range(iter):
        # Muestra la evolucion de sigma con un determinado valor de reg
        reg = params[0, i]
        sigma = params[1, : bestIndex[1] + 1]
        plt.title(f"C = {reg}")        
        train = plt.plot(sigma, eTraining[i, : bestIndex[1] + 1], label="Entrenamiento", color='green')
        val = plt.plot(sigma, eValidation[i, : bestIndex[1] + 1], label="Validacion", color='yellow')
        plt.xlabel(fr'Evolución de $\sigma$')
        plt.ylabel('Puntuación media')
        leg = plt.legend()
        plt.show()

    iter = bestIndex[1] + 1
    for i in range(iter):
        # Muestra la evolucion de reg con un determinado valor de sigma
        sigma = params[1, i]
        reg = params[0, : bestIndex[0] + 1]
        plt.title(fr'$\sigma$ = {sigma}')        
        plt.plot(reg, eTraining[: bestIndex[0] + 1, i], label="Entrenamiento", color='blue')
        plt.plot(reg, eValidation[: bestIndex[0] + 1, i], label="Validacion", color='red')
        plt.xlabel('Evolución de C')
        plt.ylabel('Puntuación media')
        plt.legend()
        plt.show()

    
def drawBarsComparision(realUsers, predictUsers):
    """
    Dibuja las gráficas comparativas en forma de barras entre los datos
    reales y la predicción
    """
    performance = ["A", "B", "C", "D"]

    X = np.arange(len(performance))
    Y1 = realUsers
    Y2 = predictUsers
    plt.bar(X, +Y1, facecolor='#9999ff', edgecolor='white', label='Reales')
    plt.bar(X, -Y2, facecolor='#ff9999', edgecolor='white', label='Predicciones')
    i = 0

    
    for x, y in zip(X, Y1):
        plt.text(x, y - 5.0, '%.0f' % y, ha='center', va='top')
        plt.text(x, 5.0, performance[i], ha='center', va='bottom')
        i += 1

    diff = 1 - np.abs(realUsers - predictUsers) / realUsers
    i = 0
    for x, y in zip(X, Y2):
        plt.text(x, -y + 5.0, '%.0f' % y, ha='center', va='bottom')
        plt.text(x, -y + 5.0, '%.0f' % y, ha='center', va='bottom')
        plt.text(x, -5.0, f'{round(diff[i] * 100, 2)} %', ha='center', va='top')

        i += 1

    plt.xticks([])
    plt.yticks([])
    plt.xlabel('Tipos de grados de entrenamiento')
    plt.ylabel('Número de personas por grado')
    plt.legend(bbox_to_anchor=(0.5, 1.05, 0.5, 0.), loc='right',
            ncol=2, mode="expand", borderaxespad=0.)

    plt.show()

def calculateUsers(y):
    initLabel = 1
    
    users = np.zeros(4)
    users[0] = np.sum(y == initLabel)
    users[1] = np.sum(y == initLabel + 1)
    users[2] = np.sum(y == initLabel + 2)
    users[3] = np.sum(y == initLabel + 3)

    return users

def drawGraphics(X, y, Xval, yval, Xtest, ytest, svm, params, eTraining, eValidation, bestIndex):
    # Gráfica de selección de parámetros
    drawParamSelection(params, eTraining, eValidation, bestIndex)
    #-------------------------------------------------------------------#
    # Barras de comparación de los datos de entrenamiento
    yp = svm.predict(X)
    realUsers = calculateUsers(y)
    predictUsers = calculateUsers(yp)
    plt.title("Predicción con X, y", loc='left')
    drawBarsComparision(realUsers, predictUsers)
    # Barras de comparación de los datos de validación
    yvalp = svm.predict(Xval)
    realUsers = calculateUsers(yval)
    predictUsers = calculateUsers(yvalp)
    plt.title("Predicción con Xval, yval", loc='left')
    drawBarsComparision(realUsers, predictUsers)
    # Barras de comparación de los datos de testing
    ytestp = svm.predict(Xtest)
    realUsers = calculateUsers(ytest)
    predictUsers = calculateUsers(ytestp)
    plt.title("Predicción con Xtest, ytest", loc='left')
    drawBarsComparision(realUsers, predictUsers)
    #-------------------------------------------------------------------#