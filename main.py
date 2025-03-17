import pandas as pd
import numpy as np

class Fuzzy:

    """
    The algorithm is intended to cluster using pre-defined clusters (which must be defined on dataset_k)
    """

    def __init__(self, ds: pd.DataFrame, ds_label: str):
        """
        Parameters:
            ds (pd.DataFrame): The dataset to be used in clustering.
            ds_label (str): The column name used to reference the samples label.
                Note: This must be explicity defined to initialize clusters.
        """

        self.ds = ds
        self.ds_label = ds_label

        self.clusters = [] # List of np.arrays with samples from each cluster label (ds_label)
        self.ds_classes = ds[ds_label].unique()  # Array of dataset label names
        self.ds_params = ds.columns # Array of dataset param names (**NEED TO CUT ds_label COLUMN...)

        self.cluster_gen()


    def cluster_gen(self):
        """
        Algorithm to generate clusters by ds_label.
        """

        for specie in self.ds_classes:
            filtered_data = self.ds[self.ds[self.ds_label] == specie]
            print(f"Species '{specie}' has {filtered_data.shape[0]} samples")
            specie_data = filtered_data.to_numpy()
            self.clusters.append(specie_data)

        print(f"Created {len(self.clusters)} cluster arrays")

    def restrictions(self):
        """
        Define restrictions based on each K cluster.
        """
    
        restrictions = {"min": 0, 
                        "max": 0,
                        "mean": 0,}

        # Needed to define statistical params to assimilate each cluster restrictions
        for cluster in self.clusters:
            """
            To define rule restrictions; some examples would be:
                - define Ki(min,max) values for EACH cluster params;
                - define mean;
                ... (?)
            
            """

            pass

        return restrictions
    
    def pertinence_generator(self):
        """
        Generate new pertinencies, based on existent Ks. It is expected to be what create intermediary pertinencies between two defined ones.
        """

        pass

    def sample_classifier(self, sample: np.array):
        """
        Simulator of sample pertinencies, to validate the model.

         ** sample must be from the size of the dataset columns...
        """


        pass