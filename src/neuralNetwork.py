import numpy as np
import scipy.optimize as opt
from checkNNGradients import checkNNGradients
import matplotlib.pyplot as plt

def sigmoide_fun(Z):
    return 1 / (1 + (np.exp(-Z)))

def getEtiqueta(Y, etiqueta):
    """
    Devuelve el vector de booleanos para determinar
    que se trata de la etiqueta correcta
    """
    y_etiqueta = (np.ravel(Y) == etiqueta) * 1 # Vector de booleanos
    return y_etiqueta   # (numElems,)

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

def J(theta1, theta2, X, y, k = 4):
    """
        Calculates the J function which returns the cost
        of the Logistic Regresion    
    """
    m = X.shape[0]
    a1, a2, h = matrix_forward_propagate(X, theta1, theta2) 
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
    m = X.shape[0] # number of training examples
    # Input
    A1 = np.hstack([np.ones([m, 1]), X])

    # Hidden
    Z2 = np.dot(A1, thetas1.T)
    A2 = np.hstack([np.ones([m, 1]), sigmoide_fun(Z2)])

    # Output
    Z3 = np.dot(A2, thetas2.T)
    A3 = sigmoide_fun(Z3)

    return A1, A2, A3 

def show_total_percent(setType, X, y, optT1, optT2):
    correct = 0
    h = matrix_forward_propagate(X, optT1, optT2)[2]
    # Indices maximos
    max = np.argmax(h, axis = 1)
    max[max == 0] = 4
    correct = np.sum(max == y.ravel())
    print(f"{setType}   Porcentaje de acierto: {correct * 100 /np.shape(h)[0]}%")
    percent = correct * 100 /np.shape(h)[0]
    return percent


def neuralNetworkClassification(X, y, Xval, yval, Xtest, ytest):
    """
    Clasificación de los datos mediante Redes Neuronales
    """
    num_features = X.shape[1]
    num_labels = 4

    input_layer = num_features
    #hidden_layer = 70
    output_layer = num_labels

    #epsilon = 0.12
    #iteration = 250
    epsilons = list(np.arange(0.1, 0.3, 0.02))
    iterations = list(np.arange(250, 251, 50))
    hidden_layers = list(np.arange(70, 71, 10))
    #lambdas = [0, 0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1, 3, 10]
    lambdas = [ 0.01, 0.03, 0.1, 0.3, 1, 3, 10]
    y_labels = getLabelMatrixY(y, num_labels)
    
    # Check if the gradient is good
    # for lamb in lambdas:
    #     checking = checkNNGradients(backprop, lamb)
    #     print("Comprobación de checkeo...", lamb, '  ', checking.sum() < 10e-9)

    max_percent = -1
    best_optT1 = []
    best_optT2 = []
    for iteration in iterations:
        for hidden_layer in hidden_layers:
            for epsilon in epsilons:
                Jval = np.ones(len(lambdas))
                Jtraining = np.ones(len(lambdas))
                for i in range(len(lambdas)):
                    lamb = lambdas[i]
                    # Se entrena con las X
                    weights_size = hidden_layer * (input_layer + 1) + output_layer * (hidden_layer + 1)
                    weights = np.random.uniform(-epsilon, epsilon, weights_size)
                    result = opt.minimize(fun = backprop, x0 = weights,
                                    args = (input_layer, hidden_layer, output_layer, X, y_labels, lamb), 
                                    method='TNC', jac=True, options={'maxiter': iteration})
                    
                    optT1 = np.reshape(result.x[:hidden_layer * (input_layer + 1)], (hidden_layer, (input_layer + 1)))
                    optT2 = np.reshape(result.x[hidden_layer * (input_layer + 1):], (num_labels, (hidden_layer + 1)))
                    print(f"Lambda {lamb}   Epsilon: {epsilon} Iterations: {iteration} Hidden layers: {hidden_layer}")
                    Jtraining[i] = 10 - show_total_percent('training', X, y, optT1, optT2)/10
                    current_percent = show_total_percent('validation', Xval, yval, optT1, optT2)
                    Jval[i] = 10 - current_percent/10
                    if current_percent > max_percent:
                        max_percent = current_percent
                        best_optT1 = optT1
                        best_optT2 = optT2
                    show_total_percent('test', Xtest, ytest, optT1, optT2)
                    show_total_percent('curent best test', Xtest, ytest, best_optT1, best_optT2)
                    
                    print("                       ")
                name = 'lambdas_it' + str(iteration) + '_hidd' + str(hidden_layer) + '_eps' + str(round(epsilon,2)) + 'sizex ' + str(X.shape[0])+'.png'
                plt.figure()
                plt.plot(np.linspace(0,11,len(lambdas),dtype=int), Jtraining, label='Train')
                plt.plot(np.linspace(0,11,len(lambdas),dtype=int), Jval, label='Cross Validation')
                plt.legend()
                text = 'Hidden layer = ' + str(hidden_layer) + ' ' + 'Iterations = ' + str(iteration) + ' ' + '$\epsilon$ = ' + str(epsilon)
                plt.title(text)
                plt.suptitle(r'Learning curve for neural network')
                plt.tight_layout(rect=[1,1,1,1])
                plt.xlabel(r'$\lambda$')
                plt.ylabel('Error') 
                plt.savefig(name)   
                plt.show()
    show_total_percent('Best final test', Xtest, ytest, best_optT1, best_optT2)

    