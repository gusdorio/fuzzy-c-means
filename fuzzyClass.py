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
        self.U = None  # Membership matrix
        self.n_samples, _ = self.X.shape
        self.samples_dict = {}
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
        centroids = (um.T @ self.X) / np.sum(um, axis=0, keepdims=True).T  # Formula for cluster centers
        return centroids # Returns a matrix of the centroid for each cluster
    
    def _update_membership_matrix(self, centroids):
        '''
        Makes the distance calculation between each xi for each vk, them updates membership.
        
        parameters: 
                - centroids: the cluster centers to be used to update the membership matrix
        '''

        # Compute Euclidean distance for each Uij
        dist = np.linalg.norm(self.X[:, np.newaxis] - centroids, axis=2)  # Compute Euclidean distances
        dist = np.fmax(dist, np.finfo(np.float64).eps)  # Avoid division by zero
        # print(dist)
        power = -2 / (self.m - 1)
        
        # Compute new U using the formula:
        # u_ij = 1 / sum_k ( (d_ij / d_ik) ^ (2 / (m - 1)) )
        U_new = 1 / np.sum((dist[:, :, np.newaxis] / dist[:, np.newaxis, :]) ** power, axis=2)
        return U_new
    
    def fit(self):
        '''
        Fit the Fuzzy C-Means model to the data.

        parameters: None --> it's a holder for the fit and predict methods
        Uses the formula application on the data to fit the model
        '''
        for i in range(self.max_iter):
            prev_U = self.U.copy()
            centroids = self._compute_cluster_centers()
            self.U = self._update_membership_matrix(centroids)

            # Check convergence (difference between old and new U is below tolerance)
            diff = np.linalg.norm(self.U - prev_U, ord='fro')
            # print(f"Iteration {i}, difference = {diff}")                              # For comparing, if needed...
        
            if np.linalg.norm(self.U - prev_U) < self.tol:
                print(f"The configured tolerance for convergence was reached... (dif value: {self.tol})")
                print(f"Number of iterations: {i}")
                break

        print(f"The maximum convergence reached was: {diff}")
    
    def predict(self):
        '''
        Predict the cluster assignments for the data.

        **NEEDS BETTER DEFINITION...

        parameters: None --> it's a holder for the fit and predict methods
        '''

        return np.argmax(self.U, axis=1)  # Assign each point to the cluster with highest membership
    
    def generate_pertinence(self):
        '''
        Function to be called on the program to algorithm application
        '''

        self.fit()

        # Generates the pertinence dict
        xi_n = 0
        for sample in self.U:
            xi_n += 1
            sample_dict = {}   # Initialize the dictionary for each sample
            k_n = 0
            for sample_k in sample:
                k_n += 1
                sample_dict[f"k{k_n}"] = sample_k
            self.samples_dict[f"x{xi_n}"] = sample_dict   # Assign the accumulated dictionary to the sample key

        print("\n")
        print("Application > To see pertinence, simple call self.samples_dict or choose your desired sample.")
        