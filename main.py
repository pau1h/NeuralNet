import numpy as np
from array import array
from os.path  import join
import pandas as pd
import matplotlib.pyplot as plt



#global vars
train_data = pd.read_csv('mnist_train.csv')
test_data = pd.read_csv('mnist_test.csv')
train = train_data.to_numpy() #785 total, idx 0 is the label, so 784 is the actual length
test = test_data.to_numpy()
x_train = train[:, 1:]/255 #selecting all rows and all columns except for the first one n x d
y_train = train[:, 0] #n x 1
x_test = test[:,1:]/255
y_test = test[:,0]
in_dim, H, out_dim, N = len(train[0]) - 1, 256, 10, len(train)
w1 = np.random.randn(in_dim, H) * np.sqrt(2/in_dim) #w1 connecting all input vectors to all hidden layers. so this is 784 x H. w1 = array of weights where w1[0][0] is the weight from the first input to the first hidden layer node
w2 = np.random.randn(H, out_dim) * np.sqrt(2/H) #using kaiming initialization
loss_history = []
acc_history = []


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


def start_training(learning_rate, epochs, w1, w2):
    for epoch in range(epochs):
        #we need to calculate the result of the hidden layer, h is supposed to be x . w1. x = 60000 x 784, w = 784 x H, so h should be 60000 x H
        z = x_train@w1
        h = relu(z) 
        logits = h@w2 #60000 x 10. We dont apply relu here because we want the logits for softmax
        preds = np.argmax(logits, axis=1)
        train_accuracy = np.mean(y_train == preds)
        L = cross_entropy_loss(logits, y_train)
        mean_L = np.mean(L)
        print(f'Training accuracy: Epoch {epoch}, {train_accuracy*100}%, Loss: {mean_L}')
        loss_history.append(mean_L)
        acc_history.append(train_accuracy)
        #now we need to calculate dloss / dlogits. this is 1/N * (softmax probabilities - y_onehot)
        probs = softmax(logits)
        y_onehot = np.zeros_like(logits)
        y_onehot[np.arange(N), y_train] = 1 #1 for correct label, 0 for everything else
        dlogits = (1/N)*(probs - y_onehot) #60000 x 10, w2 is H x 10
        dw2 = h.T@dlogits #Hx10
        dh = dlogits@w2.T #60000 x H
        dz = dh * (z>0) #propagating through relu
        dw1 = x_train.T@dz
        #update weights
        w1 -= learning_rate*dw1 
        w2 -= learning_rate*dw2
    return w1, w2


def main():
    global w1,w2
    w1, w2 = start_training(4e-1, 150, w1, w2)
    z = x_test@w1
    h = relu(z)
    logits = h@w2
    preds = np.argmax(logits, axis=1)
    test_accuracy = np.mean(preds == y_test)
    print(f"Test accuracy: {test_accuracy*100}")
    #plotting loss and accuracy 
    plt.plot(acc_history)
    plt.xlabel("Epoch")
    plt.ylabel("Training Accuracy")
    plt.title("Training Accuracy vs Epoch")
    plt.show()
    plt.plot(loss_history)
    plt.xlabel("Epoch")
    plt.ylabel("Cross Entropy Loss")
    plt.title("Training Loss vs Epoch")
    plt.show()



if __name__ == "__main__":
    main()



