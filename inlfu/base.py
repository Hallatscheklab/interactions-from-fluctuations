import numpy as np

class Base(object):
    """
    
    Class that infers migration matrix starting from a matrix of counts and non_missing tot_counts
    both counts and tot_counts should have shape: NDxNtrajxT 
    tot_counts directly refers to the number of non-missing counts for that specific allele
    
    """
    
    def __init__(self, counts=[], totcounts=[]):
        self.counts=counts
        self.totcounts=totcounts
        
    def infer(self):
        
        """ 
        This is the main function that runs the inference
        It returns the inferred matrix.
        """
        return None
    
    def predict(self):
        
        """ 
        Do we want to have this function?
        """
        return None