import pandas as pd

class Fuzzy:

    def __init__(self, dataset: pd, k=0):

        self.dataset = dataset
        self.k = k
        self_clusters = []

    def dataset_mean(self):
        return self.dataset.mean()

    def sample_distance_from_mean(self):
        """ Expects to calculate each sample x_ij from the mean of the dataset. """
        pass

    def cluster_generator(self, dataset):
        """
        Algorithm to generate clusters by probability of distances between samples and mean (?).
        """
        rules = {
            1: "to define",
            2: "to define",
        }

        def sample_standard_deviation(x_ij, cluster):
            """
            Calculate the standard deviation of a sample x_ij from a cluster.
            x_ij: sample from dataset
            cluster: (?)
            """
            pass

        def centroid_generator(cluster):
            """
            Generate cluster centroids (based on sample_standard_deviation)
            cluster: (?)
            """
            pass

