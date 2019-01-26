import numpy as np
import matplotlib.pyplot as plt
from datasetLoader import loadDataset


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))

# dim - size of a vector w
def zeroInit(dim):
    w = np.zeros((dim, 1))
    b = 0
    
    return w, b

def computeBinaryCrossEntropy(A, Y, m):    
    return -1.0 / m * (np.sum(Y * np.log(A) + (1. - Y) * np.log(1. - A)))

def propagate(w, b, X, Y):
    """
    Arguments:
    w -- weights, a numpy array of size (w * h * 3, 1)
    b -- bias, a scalar
    X -- data of size (w * h * 3, number of examples)
    Y -- true "label" vector of size (1, number of examples)

    Return:
    cost -- negative log-likelihood cost for logistic regression
    dw -- gradient of the loss with respect to w, thus same shape as w
    db -- gradient of the loss with respect to b, thus same shape as b
    """
    
    # number of examples 
    m = X.shape[1]
    
    # compute z^(i) for every example x^(i)
    # size = (1, m)
    Z = np.dot(w.T, X) + b
    
    # compute activation for every z^(i)
    # size = (1, m)
    A = sigmoid(Z)
    
    # compute cost J as binary crossentropy 
    cost = computeBinaryCrossEntropy(A, Y, m)
    
    # compute dJ / dw = [ dJ / dw1, ... , dJ / dw_m ]
    # size is the same as w (number of features)
    # db is the same size as b, i.e scalar 
    dw = 1. / m * np.dot(X, (A - Y).T) 
    db = 1. / m * np.sum(A - Y)

    assert(dw.shape == w.shape)
    assert(db.dtype == float)
    cost = np.squeeze(cost)
    assert(cost.shape == ())
    
    grads = {"dw": dw,
             "db": db}
    
    return grads, cost 

def optimizeGD(w, b, X, Y, iterations, lr):
    """
    This function optimizes w and b by running a gradient descent algorithm
    
    Arguments:
    w -- weights, a numpy array of size (w * h * 3, 1)
    b -- bias, a scalar
    X -- data of shape (w * h * 3, number of examples)
    Y -- true "label" vector (1, number of examples)
    numIterations -- number of epochs (number of GD steps)
    lr -- learning rate of the gradient descent update rule
    
    Returns:
    params -- dictionary containing the weights w and bias b
    grads -- dictionary containing the gradients of the weights and 
             bias with respect to the cost function
    costs -- list of all the costs computed during the optimization, 
             this will be used to plot the learning curve.
    """
    
    costs = []    
    for i in range(iterations):
        grads, cost = propagate(w, b, X, Y)
        
        # dw size is the same as w - (number of features, 1)
        # db is scalar 
        dw = grads["dw"]
        db = grads["db"]
        
        # GD update 
        w = w - lr * dw
        b = b - lr * db         
        
        costs.append(cost)        
        print ("Cost after iteration %i: %f" %(i, cost))
    
    params = {"w": w,
              "b": b}
    
    grads = {"dw": dw,
             "db": db}
    
    return params, grads, costs

def predict(w, b, X):
    '''
    Predict whether the label is 0 or 1 using learned logistic regression parameters (w, b)
    
    Arguments:
    w -- weights, a numpy array of size (w * h * 3, 1)
    b -- bias, a scalar
    X -- data of size (w * h * 3, number of examples)
    
    Returns:
    yPrediction -- a numpy array (vector) containing all predictions (0/1) for the examples in X
    '''
    # m - number of samples 
    m = X.shape[1]
    yPrediction = np.zeros((1, m))
    
    # size is (number of features, 1) 
    w = w.reshape(X.shape[0], 1)
    
    # Compute vector "A" predicting the probabilities for every samples 
    # size is (1, m)
    A = sigmoid(np.dot(w.T, X) + b)
    
    # predictions 
    # size (1, m)
    yPrediction = (A > 0.5).astype(float)    
    
    return yPrediction

def model(xTrain, yTrain, xTest, yTest, iterations = 2000, lr = 0.5):
    """
    Builds the logistic regression model
    
    Arguments:
    xTrain -- training set of shape (w * h * 3, m_train)
    yTrain -- training labels of shape (1, mTrain), where mTrain - number of samples
    xTest -- test set (w * h* 3, mTest)
    yTest -- test labels of shape (1, mTest)
    iterations -- hyperparameter representing the number of iterations to optimize the weights
    lr -- learning rate
    
    Returns:
    d -- dictionary containing information about the model.
    """
    
    # initialize parameters with zeros
    # size of w - (number of features, 1)
    # size of b = 1, scalar 
    w, b = zeroInit(xTrain.shape[0])

    # Gradient descent
    parameters, grads, costs = optimizeGD(w, b, xTrain, yTrain, iterations, lr)
    
    # Retrieve parameters w and b from dictionary "parameters"
    w = parameters["w"]
    b = parameters["b"]
    
    # Predict test/train set examples (≈ 2 lines of code)
    yPredTest = predict(w, b, xTest)
    yPredTrain = predict(w, b, xTrain)

    # Print train/test Errors
    print("train accuracy: {} %".format(100 - np.mean(np.abs(yPredTrain - yTrain)) * 100))
    print("test accuracy: {} %".format(100 - np.mean(np.abs(yPredTest - yTest)) * 100))

    
    d = {"costs": costs,
         "yPredTest": yPredTest, 
         "yPredTrain" : yPredTrain, 
         "w" : w, 
         "b" : b,
         "lr" : lr,
         "iterations": iterations}
    
    return d

if __name__ == "__main__":
    trainX, trainY, testX, testY, classes = loadDataset()
    
    print ("Number of training examples: {0}".format(len(trainX)))
    print ("Number of testing examples: {0}".format(len(testX)))
    print ("Shape of each image: {0}".format(trainX[0].shape))
    
    # transpose is implemented because of convenienece 
    trainXpreproc = trainX.reshape(trainX.shape[0], -1).T / 255.
    testXpreproc = testX.reshape(testX.shape[0], -1).T / 255.
    
    history = model(trainXpreproc, trainY, testXpreproc, testY)
    
    costs = np.squeeze(history['costs'])
    plt.plot(costs)
    plt.ylabel('cost')
    plt.xlabel('iterations (per hundreds)')
    plt.title("Learning rate =" + str(history["learning_rate"]))
    plt.show()
    