import numpy as np

class Base(object):
    """
    Base class for inferring migration matrices from count data.
    This class provides methods to infer a migration matrix starting from a matrix of counts and non-missing total counts. 
    Both `counts` and `totcounts` should have the shape: NDxNtrajxT, where:
    - ND: Number of dimensions
    - Ntraj: Number of trajectories
    - T: Number of time points
    
    Attributes:
        counts (list): A list representing the counts matrix.
        totcounts (list): A list representing the total counts matrix.
        A (None or matrix): The inferred migration matrix, initialized to None.
    
    Methods:
        infer():
            Runs the inference algorithm and returns the inferred migration matrix.
        predict():
            Placeholder for a prediction method. Currently returns None.
    """
    
    def __init__(self, counts=[], totcounts=[]):
        """
        Initializes the instance with counts and totcounts.
        
        Parameters:
        counts (list, optional): A list of counts. Defaults to an empty list.
        totcounts (list, optional): A list of total counts. Defaults to an empty list.
        """
        
        self.counts=counts
        self.totcounts=totcounts
        self.A=None
        
    def infer(self):
        """ 
        This is the main function that runs the inference
        It returns the inferred matrix.
        """
        return None