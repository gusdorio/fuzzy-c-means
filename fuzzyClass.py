import numpy as np
import pandas as pd

class FuzzyCMeans:
    def __init__(self, X, n_clusters=3, max_iter=100, m=2, tol=1e-5, random_state=None):
        self.X = np.array(X)  # Ensure input is a NumPy array
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.m = m  # Fuzziness parameter
        self.tol = tol
        self.random_state = random_state
        self.cluster_centers_ = None
        self.U = None  # Membership matrix
        self.n_samples, _ = self.X.shape
        self._initialize_membership_matrix()
    
    def _initialize_membership_matrix(self):
        '''
        Initialize the membership matrix randomly and normalize it.
        
        parameters: None --> it's a holder initializing the membership matrix
        '''
        np.random.seed(self.random_state)
        self.U = np.random.rand(self.n_samples, self.n_clusters)
        self.U /= np.sum(self.U, axis=1, keepdims=True)  # Normalize so that sum of each row is 1
    
    def _compute_cluster_centers(self):
        '''
        Compute the cluster centers based on the membership matrix.

        parameters: None --> it's a holder initializing the cluster centers
        '''
        um = self.U ** self.m  # Compute U^m (degree of membership raised to fuzziness parameter)
        return (um.T @ self.X) / np.sum(um, axis=0, keepdims=True).T  # Formula for cluster centers
    
    def _update_membership_matrix(self):
        '''
        Initialize the membership matrix randomly and normalize it.
        
        parameters: None --> it's a holder initializing the membership matrix
        '''
        dist = np.linalg.norm(self.X[:, np.newaxis] - self.cluster_centers_, axis=2)  # Compute Euclidean distances
        dist = np.fmax(dist, np.finfo(np.float64).eps)  # Avoid division by zero
        power = -2 / (self.m - 1)
        
        # Compute new U using the formula:
        # u_ij = 1 / sum_k ( (d_ij / d_ik) ^ (2 / (m - 1)) )
        U_new = 1 / np.sum((dist[:, :, np.newaxis] / dist[:, np.newaxis, :]) ** power, axis=2)
        return U_new
    
    def fit(self):
        '''
        Fit the Fuzzy C-Means model to the data.

        parameters: None --> it's a holder for the fit and predict mothods
        Uses the formula application on the data to fit the model
        '''
        for _ in range(self.max_iter):
            prev_U = self.U.copy()
            self.cluster_centers_ = self._compute_cluster_centers()
            self.U = self._update_membership_matrix()
            
            # Check convergence (difference between old and new U is below tolerance)
            if np.linalg.norm(self.U - prev_U) < self.tol:
                break
    
    def predict(self):
        '''
        Predict the cluster assignments for the data.

        parameters: None --> it's a holder for the fit and predict mothods
        '''
        return np.argmax(self.U, axis=1)  # Assign each point to the cluster with highest membership
    
    def fit_predict(self):
        '''
        Function to be called on the program to algorithm application
        '''

        '''
        Make it return the pertinences of the samples to the clusters periodically
        '''
        self.fit()
        return self.predict()