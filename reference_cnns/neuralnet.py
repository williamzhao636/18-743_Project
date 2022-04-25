import sys
import math
import numpy as np

class OneHiddenNN(object):
    def __init__(self, numIn, numHidden, numOut, numEpoch, learningRate, initFlag=1):
        self.numInputs = numIn
        self.numOutputs = numOut
        self.numHidden = numHidden
        self.numEpoch = numEpoch
        self.learningRate = learningRate

        # Initialize depending on flag
        # Can use np.random.rand() and np.zeros for initializing
        # Initialize bias terms to 0
        if initFlag == 1:
            alpha = np.random.rand(numHidden, numIn + 1)
            alpha[:, 0] = np.zeros(numHidden)
            beta = np.random.rand(numOut, numHidden + 1)
            beta[:, 0] = np.zeros(numOut)
        elif initFlag == 2:
            alpha = np.zeros((numHidden, numIn + 1), dtype=float)
            beta = np.zeros((numOut, numHidden + 1), dtype=float)
        else:
            print("Error: Bad initialization")
            alpha = None
            beta = None

        self.alpha = alpha
        self.beta = beta
        self.alpha = np.array([[1, 1, 2, -3, 0, 1, -3], [1, 3, 1, 2, 1, 0, 2], [1, 2, 2, 2, 2, 2, 1], [1, 1, 0, 2, 1, -2, 2]])
        self.beta = np.array([[1, 1, 2, -2, 1], [1, 1, -1, 1, 2], [1, 3, 1, -1, 1]])

        # Initialize derivatives
        self.dalpha = np.zeros((numHidden, numIn), dtype=float)
        self.dbeta = np.zeros((numOut, numHidden), dtype=float)

        # Initialize activations
        x = np.zeros(numIn, dtype=float)
        a = np.zeros(numHidden, dtype=float)
        z = np.zeros(numHidden, dtype=float)
        b = np.zeros(numOut, dtype=float)
        yhat = np.zeros(numOut, dtype=float)
        activations = [x, a, z, b, yhat]
        self.activations = activations

    # Sigmoid function
    def sigmoid(self, vector):
        return 1.0 / (1.0 + np.exp(-vector))

    # Define sigmoid derivative for clarity
    def dsigmoid(self, vector):
        return self.sigmoid(vector) * (1.0 - self.sigmoid(vector))

    # Loss function -> Need to make it average in SGD
    def crossEntropy(self, y, yhat):
        return np.dot(y, np.log(yhat))

    # Propogate forward and remember calculations in self.activations
    # returns yhat so we can predict
    def forward_prop(self, inputs):
        self.activations[0] = inputs
        
        vectorA = np.dot(self.alpha, inputs)
        self.activations[1] = vectorA
        
        vectorV = self.sigmoid(vectorA)
        self.activations[2] = vectorV

        dummyV = np.ones((self.numHidden + 1, 1))
        dummyV[1:] = vectorV
        vectorB = np.dot(self.beta, dummyV)
        self.activations[3] = vectorB
        
        vectorYhat = np.exp(vectorB)
        sumBs = np.sum(vectorYhat)
        vectorYhat = (1.0 / sumBs) * vectorYhat
        self.activations[4] = vectorYhat
        return vectorYhat

    # Predict given yhat, pick largest value with argmax, auto picks lowest index
    # Index is also our output
    def predict(self, yhat):
        largest = np.argmax(yhat)
        return largest
        
    # Propogate backwards
    def back_prop(self, y, yhat, inputs):
        dldb = (np.sum(y) * yhat) - y
        dummyV = np.ones((self.numHidden + 1, 1))
        dummyV[1:] = self.activations[2]
        dldBeta = np.dot(dldb, np.transpose(dummyV))
        self.dbeta = dldBeta
        
        #dldz = np.transpose(np.dot(np.transpose(dldb), self.beta[:, 1:]))
        dldz = np.dot(np.transpose(self.beta[:, 1:]), dldb)
        dlda = dldz * self.dsigmoid(self.activations[1])
        dldAlpha = np.dot(dlda, np.transpose(inputs))
        self.dalpha = dldAlpha
    
    # Stochastic Gradient Descent
    def SKwonJiYong(self, trainData):
        for iter in self.numEpoch:
            # Do SGD
            for inout in trainData:
                x = inout[0]
                y = inout[1]
                yhat = self.forward_prop(x)
                self.back_prop(y, yhat, x)
                self.alpha = self.alpha + (self.learningRate * self.dalpha)
                self.beta = self.beta + (self.learningRate * self.dbeta)
        return None

# Get datapoints, put into arrays, getnumInput
def getDataPoints(trainData):
    # Open files to get data from
    fileData = open(trainData, "r")
    currInput = fileData.readline()
    count = 0
    dataPs = []

    # Keep track of number data points, construct dictionaries
    while currInput != "":
        dataPs.append(get_X_Y(currInput))
        currInput = fileData.readline()
        count += 1
    
    # Close files that were opened
    fileData.close()
    return (dataPs, count)

# Get input vector x and make label y a onehot encoding
def get_X_Y(inputData):
    currInput = inputData.strip()
    yandx = np.fromstring(currInput, dtype=float, sep=',')
    xsize = yandx.size

    # y is all zeros except for the index at label
    # We are guaranteed same output label space, so hardcode 10
    y = np.zeros((10, 1), dtype=float)
    y[int(yandx[0])] = 1.0

    # x is the rest of the labels
    # Add a 1 in front for bias terms
    yandx[0] = 1.0
    x = np.reshape(yandx, (xsize, -1))
    return (x, y)


if __name__ == '__main__':
    '''
    test = OneHiddenNN(4, 1, 1, 6, 3, 1)
    x = np.transpose(np.array([[1, 1, 1, 0, 0, 1, 1]]))
    y = np.transpose(np.array([[0, 1, 0]]))
    yhat = test.forward_prop(x)
    test.back_prop(y, yhat, x)
    print(get_X_Y("5,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,1,1,0,1,1,1,1,0,1,0,0,1,0,0,1,0,1,0,1,1,0,0,1,0,1,1,1,0,0,0,1,1,1,1,0,0,0,0,1,1,1,0,0,0,0,0,1,1,0,0,0,0,0,0,1,1,0,0,0,0,0,0,1,1,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0\n"))
    '''
    trainIn = sys.argv[1]
    valIn = sys.argv[2]
    trainOut = sys.argv[3]
    valOut = sys.argv[4]
    metricsOut = sys.argv[5]
    numEpoch = int(sys.argv[6])
    numHiddenUnits = int(sys.argv[7])
    initFlag = int(sys.argv[8])
    learningRate = float(sys.argv[9])

    (dataPs, countIn) = getDataPoints(trainIn)
    (testPs, countVal) = getDataPoints(valIn)
    
    training = OneHiddenNN(countIn, numHiddenUnits, 10, numEpoch, learningRate, initFlag)
    training.SKwonJiYong(dataPs)