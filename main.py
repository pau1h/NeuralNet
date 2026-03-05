import numpy as np
from array import array
from os.path  import join
import pandas as pd


train_data = pd.read_csv('mnist_train.csv')
test_data = pd.read_csv('mnist_test.csv')
train = train_data.to_numpy() #785 total, idx 0 is the label, so 784 is the actual length
test = test_data.to_numpy()
x_train = train[:, 1:] #selecting all rows and all columns except for the first one n x d
y_train = train[:, 0] #n x 1
epochs = 1

in_dim, H, out_dim, N = len(train[0]) - 1, 50, 10, len(train)

w1 = np.random.randn(in_dim, H) * np.sqrt(2/in_dim) #w1 connecting all input vectors to all hidden layers. so this is 784 x 50. w1 = array of weights where w1[0][0] is the weight from the first input to the first hidden layer node
w2 = np.random.randn(H, out_dim) * np.sqrt(2/H) #using kaiming initialization

def relu(x): return np.maximum(0, x)

def softmax(x): 
    z = x - np.max(x, axis=1, keepdims=True) #axis = 0 reduces across columns, 1 across rows, None is the entire matrix. 
    numerator = np.exp(z)
    denominator = np.sum(numerator, axis=1, keepdims=True)
    return numerator / denominator

def cross_entropy_loss(logits, y_true):
    #we need to get the max from all logits
    m = np.max(logits, axis=1, keepdims=True)
    #so now, for each row we need to take the logsumexp of the entirerow, and subtract that from the logit from the real class (logits[row][y_pred[row]])
    logsumexp = np.log(np.sum(np.exp(logits - m), axis=1, keepdims=True)) + m #subtract the max from all rows. this prevents an overflow when exponentiating. add it back in at the end
    return -logits[np.arange(len(y_true)), y_true] + logsumexp[:, 0] #selecting from logits where each row and column is the index of y_true, and y_true itself. returns the loss which is the actual logit of the correct class - the logsumexp from the rows logits. 


for epoch in range(epochs):
    #we need to calculate the result of the hidden layer, h is supposed to be x . w1. x = 60000 x 784, w = 784 x 50, so h should be 60000 x 50
    h = relu(x_train@w1) 
    print(h.shape)
    logits = h@w2 #y_pred is now 60000 x 10. We dont apply relu here because we want the logits for softmax
    loss = cross_entropy_loss(logits, y_train)

