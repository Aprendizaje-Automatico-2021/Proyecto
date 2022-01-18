import numpy as np
import scipy.optimize as opt


def sigmoide_fun(Z):
    return 1 / (1 + (np.exp(-Z)))

def getEtiqueta(Y, etiqueta):
    """
    Devuelve el vector de booleanos para determinar
    que se trata de la etiqueta correcta
    """
    y_etiqueta = (np.ravel(Y) == etiqueta) * 1 # Vector de booleanos
    return y_etiqueta   # (numElems,)

# def evalua(i, theta, X, Y): 
#     S = sigmoide_fun(np.matmul(X, theta))   # (5K,)
#     pos = np.where(S >= 0.5)   #(5K,)
#     neg = np.where(S < 0.5) #(5K,)
#     posExample = np.where(Y == 1)
#     negExample = np.where(Y == 0)

#     # intersect1d: sirve para coger añadir elementos al vector
#     # cuando éstos sean iguales
#     totalPos = np.intersect1d(pos, posExample).shape[0] / S.shape[0]
#     totalNeg = np.intersect1d(neg, negExample).shape[0] / S.shape[0]
#     # El porcentaje total sale de la cantidad de ejemplos identificados
#     # como la etiqueta y de la cantidad que ha identifiado que no son la etiqueta
#     return totalPos + totalNeg

def getLabelMatrixY(y, numPerfm):
    """
    Genera una matriz Y selecciona con 0s y 1s
    el grado de performance de cada usuario
    """
    # Matriz para identificar qué performance posee cada usuario
    perfmY = np.zeros((y.shape[0], numPerfm)) # (numElems, numLabels)

    for i in range(numPerfm):
        perfmY[:, i] = getEtiqueta(y, i)

    # Las etiquetas de los 0's
    perfmY[:, 0] = getEtiqueta(y, 4)

    return perfmY



# def oneVsAll(X, y, Xval, yval, numPerfm=4, iters=8, initReg=0.01):
#     """
#     Entrenamiento de varios clasificadores por regresión logística
#     """
#     numFeatures = X.shape[1]
#     # Matriz de parámetros theta
#     theta = np.zeros((numPerfm, numFeatures))
#     perfmY = getLabelMatrixY(y, numPerfm)
#     perfmYval = getLabelMatrixY(yval, numPerfm)

#     # Entrenamiento
#     evaluacion = np.zeros(numPerfm)
#     bestScore = np.zeros(numPerfm)
#     bestReg = np.zeros(numPerfm)

#     for i in range(numPerfm):
#         for j in range(iters):
#             reg = initReg * 3**j
#             # Se entrena con las X
#             result = opt.fmin_tnc(func = coste, x0 = theta[i, :], fprime = gradiente,
#                     args=(X, perfmY[:, i], reg))
#             theta[i, :] = result[0]

#             # Se evalua con las Xval
#             evaluacion[i] = evalua(i, theta[i, :], Xval, perfmYval[:, i])
#             if(evaluacion[i] > bestScore[i]):
#                bestScore[i] = evaluacion[i] 
#                bestReg[i] = reg

#     return bestScore, bestReg 

#--------New Code----------------------------------------------------------------------------------#
    
def J(theta1, theta2, X, y, k = 10):
    """
        Calculates the J function of the cost
        of the Logistic Regresion    
    """
    m = X.shape[0]
    a1, a2, h = matrix_forward_propagate(X, theta1, theta2) # (10, )
    sum1 = y * np.log(h + 1e-9)
    sum2 = (1 - y) * np.log(1 - h + 1e-9)
    total = np.sum(sum1 + sum2)

    return (-1 / m) * total

def regularization(thetas1, thetas2, m, lamb):
    """
    Calcula el termino regularizado de la función de coste
    en función de lambda
    """
    total = 0
    # t1(25, 400), t2(10, 25)
    sum1 = np.sum(np.power(thetas1[1:], 2))
    sum2 = np.sum(np.power(thetas2[1:], 2))
    total = sum1 + sum2
    return (lamb / (2 * m)) * total

def gradient_regularitation(delta, m, reg, theta):
	index0 = delta[0]
	delta = delta + (reg / m) * theta
	delta[0] = index0
	return delta

def backprop (params_rn, num_entradas, num_ocultas, num_etiquetas, X, y , reg):
    """
    Back-Propagation
    """
    Theta1 = np.reshape(params_rn[:num_ocultas * ( num_entradas + 1)], (num_ocultas, (num_entradas + 1)))
    Theta2 = np.reshape(params_rn[num_ocultas * ( num_entradas + 1):], (num_etiquetas, (num_ocultas + 1)))

    delta1 = np.zeros_like(Theta1)
    delta2 = np.zeros_like(Theta2)

    A1, A2, H = matrix_forward_propagate(X, Theta1, Theta2)

    m = X.shape[0]
    cost = J(Theta1, Theta2, X, y) + regularization(Theta1, Theta2, m, reg)

    for t in range(m):
        a1t = A1[t, :]
        a2t = A2[t, :]
        ht = H[t, :]
        yt = y[t]

        d3t = ht - yt   # Este es el error comparando el valor obtenido con el que deberia obtenerse
        d2t = np.dot(Theta2.T, d3t) * (a2t * (1 - a2t)) 
        # ¡OJO!: En d2t había un fallo tremendo porque se estaba haciendo np.dot(a2t * (1 - a2t)) en lugar
        # de la multiplicación actual, lo cual provocaba cambios en el valor delta1 bastante catastróficos

        delta1 = delta1 + np.dot(d2t[1:, np.newaxis], a1t[np.newaxis, :])
        delta2 = delta2 + np.dot(d3t[:, np.newaxis], a2t[np.newaxis, :])

    delta1 /= m
    delta2 /= m

    # Delta's gradients
    delta1 = gradient_regularitation(delta1, m, reg, Theta1)
    delta2 = gradient_regularitation(delta2, m, reg, Theta2) 

    gradient = np.concatenate((delta1.ravel(), delta2.ravel()))

    return cost, gradient

def matrix_forward_propagate(X, thetas1, thetas2):
    m = X.shape[0]
    # Input
    A1 = np.hstack([np.ones([m, 1]), X])

    # Hidden
    # (5K, 401) * (401, 25) = (5K, 25)
    Z2 = np.dot(A1, thetas1.T)
    A2 = np.hstack([np.ones([m, 1]), sigmoide_fun(Z2)])

    # Output
    # (5k, 26) * (26, 10) = (5K, 10)
    Z3 = np.dot(A2, thetas2.T)
    A3 = sigmoide_fun(Z3)

    return A1, A2, A3   

def neuralNetworkClassification(X, y, Xval, yval, Xtest, ytest):
    """
    Clasificación de los datos mediante Redes Neuronales
    """
    num_ex = X.shape[0] #m - number training examples
    num_features = X.shape[1] #n - number features 
    num_labels = 4

    input_layer = num_features
    hidden_layer = 100
    output_layer = num_labels

    epsilon = 0.12
    iterations = 250
    #lamb = 1
    #theta1 -> hidden x (num_features + 1) = hidden_layer * (input_layer + 1) 
    #theta2 -> num_labels * hidden = output_layer * (hidden_layer+1)
    weights_size = hidden_layer * (input_layer + 1) + output_layer * (hidden_layer+1)
    lambdas = [0, 0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1, 3, 10]
    for j in range(len(lambdas)):
        lamb = lambdas[j]
        weights = np.random.uniform(-epsilon, epsilon, weights_size)
        y_labels = getLabelMatrixY(y, num_labels) # y matrice cu n linii cu 10 coloane 1 unde e eticheta
        result = opt.minimize(fun = backprop, x0 = weights,
                        args = (input_layer, hidden_layer, output_layer, X, y_labels, lamb), 
                        method='TNC', jac=True, options={'maxiter': iterations})
        
        optT1 = np.reshape(result.x[:hidden_layer * ( input_layer + 1)], (hidden_layer, (input_layer + 1)))
        optT2 = np.reshape(result.x[hidden_layer * ( input_layer + 1):], (num_labels, (hidden_layer + 1)))

        correct = 0
        h = matrix_forward_propagate(X, optT1, optT2)[2]

        # Indices maximos
        max = np.argmax(h, axis = 1)

        correct = np.sum(max == y.ravel())
        print(f"Lamda: {lamb}   Porcentaje de acierto: {correct * 100 /np.shape(h)[0]}%")
    return 0

    