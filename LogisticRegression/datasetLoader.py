import numpy as np
import h5py
    
PATH_TO_TRAIN = r'datasets/train_catvnoncat.h5'
PATH_TO_TEST = 'datasets/test_catvnoncat.h5'

def loadDataset():
    trainDataset = h5py.File(PATH_TO_TRAIN, "r")
    trainX = np.array(trainDataset["train_set_x"][:]) 
    trainY = np.array(trainDataset["train_set_y"][:])

    testDataset = h5py.File(PATH_TO_TEST, "r")
    testX = np.array(testDataset["test_set_x"][:]) # your test set features
    testY = np.array(testDataset["test_set_y"][:]) # your test set labels

    classes = np.array(testDataset["list_classes"][:]) # the list of classes
    
    trainY = trainY.reshape((1, trainY.shape[0]))
    testY = testY.reshape((1, testY.shape[0]))
    
    return trainX, trainY, testX, testY, classes

