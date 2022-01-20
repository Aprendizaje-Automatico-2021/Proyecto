import numpy as np
import scipy.optimize as opt
from sklearn import neural_network
from checkNNGradients import checkNNGradients
import matplotlib.pyplot as plt
import os

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

def get_total_percent(setType, X, y, optT1, optT2):
    correct = 0
    h = matrix_forward_propagate(X, optT1, optT2)[2]
    # Indices maximos
    max = np.argmax(h, axis = 1)
    max[max == 0] = 4
    correct = np.sum(max == y.ravel())
    print(f"{setType}   Porcentaje de acierto: {correct * 100 /np.shape(h)[0]}%")
    percent = correct * 100 /np.shape(h)[0]
    return percent

def create_learning_curve_graphic(path, parameters, lambdas, Jtraining, Jval, label):
    plt.plot(lambdas, Jtraining, label='Traning Data')
    plt.plot(lambdas, Jval, label='Validation Data')
    plt.legend()
    plt.title(parameters)
    plt.suptitle(r'Learning curve for neural network')
    plt.tight_layout(rect=[1,1,1,1])
    plt.xlabel(label)
    plt.ylabel('Cost') 
    plt.savefig(path)   
    #plt.show()
    plt.close()

def make_neural_network(input_layer, hidden_layer, output_layer, X, y_labels, lamb, iteration, epsilon):
    # Initialize the thetas random values between -eps and eps
    weights_size = hidden_layer * (input_layer + 1) + output_layer * (hidden_layer + 1)
    weights = np.random.uniform(-epsilon, epsilon, weights_size)

    # Calculate the best thetas
    result = opt.minimize(fun = backprop, x0 = weights,
                    args = (input_layer, hidden_layer, output_layer, X, y_labels, lamb), 
                    method='TNC', jac=True, options={'maxiter': iteration})
    # As the result is an array we remake the thetas matrixes with the corespondent sizes
    optT1 = np.reshape(result.x[:hidden_layer * (input_layer + 1)], (hidden_layer, (input_layer + 1)))
    optT2 = np.reshape(result.x[hidden_layer * (input_layer + 1):], (output_layer, (hidden_layer + 1)))
    return optT1, optT2

def neuralNetworkClassification(X, y, Xval, yval, Xtest, ytest):
    """
    Clasificación de los datos mediante Redes Neuronales
    """
    # We remember the path for the graphics
    script_dir = os.path.dirname(__file__)
    result_dir = script_dir[:-4] + '\\memoria\\assets\\neu_net\\'

    # Initialize variables
    num_features = X.shape[1]
    num_labels = 4
    y_labels = getLabelMatrixY(y, num_labels)
    yval_labels = getLabelMatrixY(yval, num_labels)
    # Epsilon
    init_epsilon = 0.1
    epsilons = 6
    best_eps = 0.1

    # NeuralNet
    neural_net_iters = 200 # 100
    hid = 75    # 25
    input_layer = num_features
    output_layer = num_labels

    # Lambdas
    initLambda = 0.01
    lambdas = np.zeros(7)
    best_lambda = 0.01
    
    # Misc
    min_cost = np.inf
    best_percent = -1
    best_optT1 = []
    best_optT2 = []
    min_diff = np.inf
    min_special_cost = np.inf
    # Notas: No hace falta repetir tanto el código, solo con esta cadena de for funciona
    for j in range(epsilons):
        epsilon = init_epsilon + 0.02 * j
        Jval = np.ones(len(lambdas))
        Jtraining = np.ones(len(lambdas))
        for i in range(len(lambdas)):
            # Training with X
            lambdas[i] = initLambda * 3**i
            optT1, optT2 = make_neural_network(input_layer, hid, output_layer, X,
                                        y_labels, lambdas[i], neural_net_iters, epsilon)

            print(f"Lambda {lambdas[i]}   Epsilon: {epsilon} Iterations:"
                + f"{neural_net_iters} Hidden layers: {hid}")

            # We calculate the percentages of succes and using them we calculate the error for the training data set and cross validation data set
            Jtraining[i] = 10 - get_total_percent('training', X, y, optT1, optT2)/10
            current_percent = get_total_percent('validation', Xval, yval, optT1, optT2)
            Jval[i] = 10 - current_percent/10
            # Jtraining[i] = J(optT1, optT2, X, y_labels)
            # Jval[i] = J(optT1, optT2, Xval, yval_labels)

            diff = np.abs(Jtraining[i] - Jval[i])
            # We remember the thetas that have the smallest error(biggest percent) on the cross validation data set but also the minimum distance 
            special_cost = Jval[i]/10 + diff/4
            if  Jval[i] < min_cost:
                min_cost = Jval[i]
                min_cost_lamb = lambdas[i]
                min_cost_eps = epsilon
            if diff < min_diff:
                min_diff = diff
                diff_lamb = lambdas[i]
                diff_eps = epsilon
            if  special_cost < min_special_cost: #Combined distance and cost
                min_special_cost = special_cost
                best_percent = current_percent
                best_lambda = lambdas[i]
                best_eps = epsilon
                best_optT1 = optT1
                best_optT2 = optT2
            print("\n")

        # We prepare the parameters for the graphics
        name = 'Lambda' + 'LearningCurve' + '_it_' + str(neural_net_iters) + '_hidd_' + str(hid) + '_eps_' + str(round(epsilon,2)) + '_sizex_' + str(X.shape[0])+'.png'
        path = result_dir + name
        parameters = 'Hidden layer = ' + str(hid) + ' ' + 'Iterations = ' + str(neural_net_iters) + ' ' + '$\epsilon$ = ' + str(round(epsilon,2))
        create_learning_curve_graphic(path, parameters, lambdas, Jtraining, Jval, label = "$\lambda$")
    print()
    print(fr"Min Diff: {round(min_diff,2)} lamb: {diff_lamb} epsilon: {round(diff_eps,2)}")
    print(fr"Min Cost {round(min_cost,2)} Max Percent: {10 - round(min_cost,2)} lamb: {min_cost_lamb} epsilon: {min_cost_eps}")
    print(fr"Mejor lambda: {best_lambda}")
    print(fr"Mejor epsilon: {round(best_eps,2)}")
    print(f"Precisión mejor validación: {round(best_percent,2)} best cost: {100 - round(best_percent,2)}")
    get_total_percent('Current best test', Xtest, ytest, best_optT1, best_optT2)

    return 0
    