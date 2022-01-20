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

def create_learning_curve_graphic(path, parameters,size, Jtraining, Jval, begin, end, label):
    plt.plot(np.linspace(begin,end,size,dtype=float), Jtraining, label='Train')
    plt.plot(np.linspace(begin,end,size,dtype=float), Jval, label='Cross Validation')
    plt.legend()
    plt.title(parameters)
    plt.suptitle(r'Learning curve for neural network')
    plt.tight_layout(rect=[1,1,1,1])
    plt.xlabel(label)
    plt.ylabel('Error') 
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

    input_layer = num_features
    output_layer = num_labels

    #hidden_layer = 70
    #epsilon = 0.12
    #iteration = 250
    epsilons = [ 0.1, 0.12, 0.14, 0.16,0.18, 0.2] #6
    #epsilons = [0.12]
    iterations = list(np.arange(250, 251, 25)) #6
    hidden_layers = list(np.arange(70, 71, 20))#6
    #lambdas = [0, 0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1, 3, 10] #10
    lambdas = [ 0.01, 0.03, 0.1, 0.3, 1, 3, 10]
    y_labels = getLabelMatrixY(y, num_labels)
    
    # Check if the gradient is good
    # for lamb in lambdas:
    #     checking = checkNNGradients(backprop, lamb)
    #     print("Comprobación de checkeo...", lamb, '  ', checking.sum() < 10e-9)

    max_percent = -1
    best_optT1 = []
    best_optT2 = []

    #Lambda
    for iteration in iterations:
        for hidden_layer in hidden_layers:
            for epsilon in epsilons:
                Jval = np.ones(len(lambdas))
                Jtraining = np.ones(len(lambdas))
                for i in range(len(lambdas)):
                    # Training with X
                    lamb = lambdas[i]
                    optT1, optT2 = make_neural_network(input_layer, hidden_layer, output_layer, X, y_labels, lamb, iteration, epsilon)
                    print(f"Lambda {lamb}   Epsilon: {epsilon} Iterations: {iteration} Hidden layers: {hidden_layer}")

                    # We calculate the percentages of succes and using them we calculate the error for the training data set and cross validation data set
                    Jtraining[i] = 10 - get_total_percent('training', X, y, optT1, optT2)/10
                    current_percent = get_total_percent('validation', Xval, yval, optT1, optT2)
                    Jval[i] = 10 - current_percent/10
                    #get_total_percent('test', Xtest, ytest, optT1, optT2)
                    #get_total_percent('Current best test', Xtest, ytest, best_optT1, best_optT2)

                    # We remember the thetas that have the smallest error(biggest percent) on the cross validation data set
                    if current_percent > max_percent:
                        max_percent = current_percent
                        best_optT1 = optT1
                        best_optT2 = optT2
                    print("                       ")
                # We prepare the parameters for the graphics
                name = 'Lambda' + 'LearningCurve' + '_it_' + str(iteration) + '_hidd_' + str(hidden_layer) + '_eps_' + str(round(epsilon,2)) + '_sizex_' + str(X.shape[0])+'.png'
                path = result_dir + 'lambda\\' + name
                parameters = 'Hidden layer = ' + str(hidden_layer) + ' ' + 'Iterations = ' + str(iteration) + ' ' + '$\epsilon$ = ' + str(round(epsilon,2))
                size = len(lambdas)
                create_learning_curve_graphic(path, parameters, size, Jtraining, Jval, min(lambdas), max(lambdas), label = "$\lambda$")
    get_total_percent('Current best test', Xtest, ytest, best_optT1, best_optT2)
    #Iterations
    # for hidden_layer in hidden_layers:
    #     for epsilon in epsilons:
    #         for lamb in lambdas:
    #             lg = len(iterations)
    #             Jval = np.ones(lg)
    #             Jtraining = np.ones(lg)
    #             for i in range(lg):
    #                 # Training with X
    #                 iteration = iterations[i]
    #                 optT1, optT2 = make_neural_network(input_layer, hidden_layer, output_layer, X, y_labels, lamb, iteration, epsilon)
    #                 print(f"Lambda {lamb}   Epsilon: {epsilon} Iterations: {iteration} Hidden layers: {hidden_layer}")

    #                 # We calculate the percentages of succes and using them we calculate the error for the training data set and cross validation data set
    #                 Jtraining[i] = 10 - get_total_percent('training', X, y, optT1, optT2)/10
    #                 current_percent = get_total_percent('validation', Xval, yval, optT1, optT2)
    #                 Jval[i] = 10 - current_percent/10
    #                 #get_total_percent('test', Xtest, ytest, optT1, optT2)
    #                 #get_total_percent('Current best test', Xtest, ytest, best_optT1, best_optT2)

    #                 # We remember the thetas that have the smallest error(biggest percent) on the cross validation data set
    #                 if current_percent > max_percent:
    #                     max_percent = current_percent
    #                     best_optT1 = optT1
    #                     best_optT2 = optT2
    #                 print("                       ")
    #             # We prepare the parameters for the graphics
    #             name = 'Iteration' + 'LearningCurve' +  '_hidd_' + str(hidden_layer) + '_eps_' + str(round(epsilon,2)) + '_sizex_' + str(X.shape[0])+ '_lam_' + str(lamb) + '.png'
    #             path = result_dir + 'hidden\\' + name
    #             parameters = 'Hidden layer = ' + str(hidden_layer) + ' ' + '$\lambda$ = ' + str(lamb)  + ' ' + '$\epsilon$ = ' + str(round(epsilon,2))
    #             create_learning_curve_graphic(path, parameters, lg, Jtraining, Jval, min(iterations), max(iterations), label = "Iterations")       
    # get_total_percent('Current best test', Xtest, ytest, best_optT1, best_optT2)

    #For epsilon     
    for iteration in iterations:
        for hidden_layer in hidden_layers:
            for lamb in lambdas:
                lg = len(epsilons)
                Jval = np.ones(lg)
                Jtraining = np.ones(lg)
                for i in range(lg):
                    # Training with X
                    epsilon = epsilons[i]
                    optT1, optT2 = make_neural_network(input_layer, hidden_layer, output_layer, X, y_labels, lamb, iteration, epsilon)
                    print(f"Lambda {lamb}   Epsilon: {epsilon} Iterations: {iteration} Hidden layers: {hidden_layer}")

                    # We calculate the percentages of succes and using them we calculate the error for the training data set and cross validation data set
                    Jtraining[i] = 10 - get_total_percent('training', X, y, optT1, optT2)/10
                    current_percent = get_total_percent('validation', Xval, yval, optT1, optT2)
                    Jval[i] = 10 - current_percent/10
                    #get_total_percent('test', Xtest, ytest, optT1, optT2)
                    #get_total_percent('Current best test', Xtest, ytest, best_optT1, best_optT2)

                    # We remember the thetas that have the smallest error(biggest percent) on the cross validation data set
                    if current_percent > max_percent:
                        max_percent = current_percent
                        best_optT1 = optT1
                        best_optT2 = optT2
                    print("                       ")
                # We prepare the parameters for the graphics
                name = 'Epsilon' + 'LearningCurve' + '_it_' + str(iteration)  + '_hidd_' + str(hidden_layer) + '_sizex_' + str(X.shape[0]) + '_lam_' + str(lamb)+ '.png'
                path = result_dir + 'epsilon\\' + name
                parameters = 'Iterations = ' + str(iteration) + ' '+ 'Hidden layer = ' + str(hidden_layer)+ ' ' + '$\lambda$ = ' + str(lamb)  
                create_learning_curve_graphic(path, parameters, lg, Jtraining, Jval, min(epsilons), max(epsilons), label = "Epsilons")    
    # get_total_percent('Current best test', Xtest, ytest, best_optT1, best_optT2)
    # #Hidden layers
    # for iteration in iterations:
    #     for epsilon in epsilons:
    #         for lamb in lambdas:
    #             lg = len(hidden_layers)
    #             Jval = np.ones(lg)
    #             Jtraining = np.ones(lg)
    #             for i in range(lg):
    #                 # Training with X
    #                 hidden_layer = hidden_layers[i]
    #                 optT1, optT2 = make_neural_network(input_layer, hidden_layer, output_layer, X, y_labels, lamb, iteration, epsilon)
    #                 print(f"Lambda {lamb}   Epsilon: {epsilon} Iterations: {iteration} Hidden layers: {hidden_layer}")

    #                 # We calculate the percentages of succes and using them we calculate the error for the training data set and cross validation data set
    #                 Jtraining[i] = 10 - get_total_percent('training', X, y, optT1, optT2)/10
    #                 current_percent = get_total_percent('validation', Xval, yval, optT1, optT2)
    #                 Jval[i] = 10 - current_percent/10
    #                 #get_total_percent('test', Xtest, ytest, optT1, optT2)
    #                 #get_total_percent('Current best test', Xtest, ytest, best_optT1, best_optT2)

    #                 # We remember the thetas that have the smallest error(biggest percent) on the cross validation data set
    #                 if current_percent > max_percent:
    #                     max_percent = current_percent
    #                     best_optT1 = optT1
    #                     best_optT2 = optT2
    #                 print("                       ")
    #             # We prepare the parameters for the graphics
    #             name = 'Hidden' + 'LearningCurve' + '_it_' + str(iteration)  + '_eps_' + str(round(epsilon,2)) + '_sizex_' + str(X.shape[0]) + '_lam_' + str(lamb)+ '.png'
    #             path = result_dir + 'hidden\\' + name
    #             parameters = 'Iterations = ' + str(iteration) + ' ' + '$\lambda$ = ' + str(lamb)  + ' ' + '$\epsilon$ = ' + str(round(epsilon,2))
    #             create_learning_curve_graphic(path, parameters, lg, Jtraining, Jval, min(hidden_layers), max(hidden_layers), label = "Hidden layers")

    get_total_percent('Best final test', Xtest, ytest, best_optT1, best_optT2)

    