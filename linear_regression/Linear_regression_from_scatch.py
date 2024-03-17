import numpy as np 

class LinearRegression:

    def __init__(self,lr = 0.001,n_iter = 1000):
        self.lr = lr
        self.n_iter = n_iter
        self.weights = None
        self.bias = None 


    def fit(self,X,y):
        N_samples,N_feature = X.shape
        self.weights = np.zeros(N_feature)
        self.bias = 0
        for _ in range(self.n_iter):
           
            y_hat = np.dot(X , self.weights) + self.bias 
            
            dw = (1/N_samples) * np.dot(X.T,(y_hat - y))
            db = (1/N_samples) * np.sum(y_hat - y)
            
            self.weights = self.weights - self.lr * dw
            self.bias = self.bias - self.lr * db
    

    def predict(self ,X):
        y_hat =np.dot(X , self.weights) + self.bias
        return y_hat