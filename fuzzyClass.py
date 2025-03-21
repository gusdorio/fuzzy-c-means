import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

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
        self.fitted = False  # Track if model has been fitted
        self._initialize_membership_matrix()
        self._initialize_cluster_centers()
    
    def _initialize_membership_matrix(self):
        np.random.seed(self.random_state)
        self.U = np.random.rand(self.n_samples, self.n_clusters)
        self.U /= np.sum(self.U, axis=1, keepdims=True)  # Normalize
    
    #def _initialize_cluster_centers(self):
        # Inicializar os centroides usando uma estratégia mais dispersa (k-means++ style)
    #    np.random.seed(self.random_state)
    #    indices = np.random.choice(self.n_samples, self.n_clusters, replace=False)
    #    self.cluster_centers_ = self.X[indices]

    def _compute_cluster_centers(self):
        um = self.U ** self.m  # Compute U^m
        return (um.T @ self.X) / np.sum(um, axis=0, keepdims=True).T
        #new_centers = (um.T @ self.X) / np.sum(um, axis=0, keepdims=True).T
        #return new_centers + np.random.uniform(-0.01, 0.01, new_centers.shape)  # Evitar convergência ao mesmo ponto
    
    def _update_membership_matrix(self):
        dist = np.linalg.norm(self.X[:, np.newaxis] - self.cluster_centers_, axis=2)  # Euclidean distances
        dist = np.fmax(dist, np.finfo(np.float64).eps)  # Avoid division by zero
        power = -2 / (self.m - 1)
        U_new = 1 / np.sum((dist[:, :, np.newaxis] / dist[:, np.newaxis, :]) ** power, axis=2)
        return U_new
    
    def fit(self):
        for _ in range(self.max_iter):
            prev_U = self.U.copy()
            self.cluster_centers_ = self._compute_cluster_centers()
            self.U = self._update_membership_matrix()
            if np.linalg.norm(self.U - prev_U) < self.tol:
                break
        self.fitted = True
    
    def predict(self):
        if not self.fitted:
            raise ValueError("O modelo precisa ser ajustado antes de prever os clusters.")
        return self.U, self.cluster_centers_
    
    def fit_predict(self):
        self.fit()
        return self.predict()