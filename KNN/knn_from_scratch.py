import numpy as np 





class KNN :

    def __init__(self , k = 3):
                   
        self.k = k

    def fit (self , X , y):
        self.X_train = X
        self.y_train = y       



    def predicate (self , X):
        y_hat = []
        for x in X :
            # Compute the Euclidean distance between x and each data point in X
            distances = np.sqrt(np.sum((self.X_train - x)**2, axis=1))

            # get the indicies of KNN 
            index = np.argsort(distances)[ :self.k]

            # Get the labels of the k nearest neighbors
            labels = self.y_train[index]

            # Predict the label of x based on the majority vote of the k nearest neighbors
            counts = np.bincount(labels)
            y_hat.append(np.argmax(counts))

        return y_hat



        

        

