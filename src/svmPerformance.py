from sklearn.metrics import accuracy_score
from sklearn.svm import SVC

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
    print("Entrenando sistema de clasificacion de bodyPerfomance")
    initialValue = 0.01
    iters = 8
    svm, reg, sigma, bestScore = selectParameters(X, y, Xval, yval, initialValue, iters)

    # Matrices de prediccion de cada conjunto de datos - Aún no se usa, pero puede ser interesante para algo
    yp = svm.predict(X)
    yvalp = svm.predict(Xval)
    ytestp = svm.predict(Xtest)
    
    performance = [ord("A"), ord("B"), ord("C"), ord("D")]

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