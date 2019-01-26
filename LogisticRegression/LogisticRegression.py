import numpy as np
import matplotlib.pyplot as plt
from datasetLoader import loadDataset

if __name__ == "__main__":
    trainX, trainY, testX, testY, classes = loadDataset()
    
    mTrain = len(trainX)
    mTest = len(testX)
    imgShape = trainX[0].shape
    
    print ("Number of training examples: {0}".format(mTrain))
    print ("Number of testing examples: {0}".format(mTest))
    print ("Shape of each image: {0}".format(imgShape))