import numpy as np
from influ.base import Base
import cvxpy as cp

class F2(Base):
    """
    Class that infers migration matrix starting from a matrix of counts and non_missing tot_counts.
    Both counts and tot_counts should have shape: NDxNtrajxT.
    tot_counts directly refers to the number of non-missing counts for that specific allele.
    
    Methods
    -------
    infer(chunks_size=1000, n=5, fraction=0.8)
        Runs the whole analysis and returns the inferred matrix.
    """
        
    def infer(self, chunks_size=1000, n=5, fraction=0.8):
        """
        Infers the matrix A by processing data in chunks and averaging the results over multiple iterations.
        
        Parameters:
            chunks_size (int): The size of each data chunk to process. Default is 1000.
            n (int): The number of iterations to perform. Default is 5.
            fraction (float): The fraction of data to use in each iteration. Default is 0.8.
        
        Returns:
            numpy.ndarray: The inferred matrix A, averaged over all iterations.
        """
        
        self.As,self.F0,self.F1=[],[],[]
        self.chunks_size=chunks_size
        for i in range(n):
            self.F_X1_X0, self.F_X0_X0, self.F_X1_X0_std, self.F_X0_X0_std = from_counts_to_F2s(
                counts=self.counts,
                tot_counts=self.tot_counts,
                fraction=fraction,
                chunks_size=chunks_size
            )
            self.F0.append(self.F_X0_X0)
            self.F1.append(self.F_X1_X0)
            self.As.append(infer_A(self.F_X1_X0,self.F_X0_X0))
        return np.mean(self.As,axis=0).copy()
    
def from_counts_to_F2s(counts,totcounts,fraction:float=1.,chunks_size:int=5000) -> None:
    """
    This function builds the F2 and F2' matrices needed for inference.
    It divides the genome into blocks of equal size and then performs a weighted average across blocks 
    by weighting with the number of non-missing alleles within the block.
    The function calculates the following:
    - F_X1_X0: Matrix of F2 values between consecutive time points.
    - F_X0_X0: Matrix of F2 values within the same time point.
    - F_X1_X0_std: Standard deviation of F2 values between consecutive time points.
    - F_X0_X0_std: Standard deviation of F2 values within the same time point.
    The function uses the Patterson et al. correction factor to adjust for sampling variance.
    """  
    ND,Ntraj,T=counts.shape 
    freq=counts/totcounts
    correction_factor = freq*(1-freq)/(totcounts-1) # Patterson et al correction
    X0,X1=freq[:,:,:-1],freq[:,:,1:]
    correction_factor0,correction_factor1 = correction_factor[:,:,:-1],correction_factor[:,:,1:]
    tot_counts0,tot_counts1 = totcounts[:,:,:-1],totcounts[:,:,1:]
    F_X1_X0=np.zeros((X0.shape[-1],freq.shape[0],freq.shape[0]))
    F_X0_X0=np.zeros((X0.shape[-1],freq.shape[0],freq.shape[0]))
    F_X1_X0_std=np.zeros((X0.shape[-1],freq.shape[0],freq.shape[0]))
    F_X0_X0_std=np.zeros((X0.shape[-1],freq.shape[0],freq.shape[0]))
    for i in range(freq.shape[0]):
        for j in range(freq.shape[0]):        
                for t in range(T-1):
                # Compute F2 values
                selection_lineages=np.logical_and(tot_counts1[i,:,t]>1,tot_counts0[j,:,t]>1)
                f10=(X1[i,:,t]-X0[j,:,t])**2  - correction_factor1[i,:,t]- correction_factor0[j,:,t]
                F_X1_X0[t,i,j],F_X1_X0_std[t,i,j]=mean_chunks(f10,chunks_size,selection_lineages,fraction=fraction)
                selection_lineages=np.logical_and(tot_counts0[i,:,t]>1,tot_counts0[j,:,t]>1)
                if i!=j: f00=(X0[i,:,t]-X0[j,:,t])**2 - correction_factor0[i,:,t]- correction_factor0[j,:,t]
                else: f00=(X0[i,:,t]-X0[j,:,t])**2
                F_X0_X0[t,i,j],F_X0_X0_std[t,i,j]=mean_chunks(f00,chunks_size,selection_lineages,fraction=fraction)
    return F_X1_X0, F_X0_X0, F_X1_X0_std, F_X0_X0_std
                
def infer_A(F_X1_X0,F_X0_X0):
    """
    Infer the matrix A from the given fluctuation matrices F_X1_X0 and F_X0_X0.
    
    Parameters:
    -----------
    F_X1_X0 : numpy.ndarray
        A 3-dimensional array of shape (d, n, n) representing the fluctuation matrix at time t+1.
    F_X0_X0 : numpy.ndarray
        A 3-dimensional array of shape (d, n, n) representing the fluctuation matrix at time t.
    
    Returns:
    --------
    numpy.ndarray
        A 2-dimensional array of shape (n, n) representing the inferred matrix A.
    
    Raises:
    -------
    AssertionError
        If the input F_X1_X0 does not have 3 dimensions.
    """

    assert len(F_X1_X0.shape)==3
    results=[]
    n=F_X1_X0.shape[1]
    d=F_X1_X0.shape[0]
    for i in range(n):

        # Define and solve the CVXPY problem.
        a = cp.Variable(n)
        B=np.zeros((d,n))
        for k in range(n):
            B[:,k] = F_X1_X0[:,i,k] -  F_X1_X0[:,i,i]
        M=np.zeros((d,n,n))
        for j in range(n):
            for k in range(n):
                M[:,j,k] = F_X0_X0[:,j,k] - F_X0_X0[:,j,i]

        B=B.flatten()
        M=M.transpose(1,0,2).reshape(n,n*d)
        
        #Rescaling for numerical stability
        rescale=np.mean([np.mean(B),np.mean(M)])
        B*=1/rescale
        M*=1/rescale

        cost = cp.sum_squares(B - a@M)
        prob = cp.Problem(cp.Minimize(cost),[a>=1e-6,np.ones(n) @ a==1])
        prob.solve()
        results.append(a.value)
    return np.array(results)

def weighted_avg_and_std(values, weights):
    """
    Calculate the weighted average and standard deviation.
    
    Parameters:
        values (array-like): The data values.
        weights (array-like): The weights associated with the data values.
        fraction (float): fraction of chunks to use for resampling step
    
    Returns:
        tuple: A tuple containing the weighted average and the weighted standard deviation.
    """
    average = np.average(values, weights=weights)
    variance = np.average((values-average)**2, weights=weights)
    return average,np.sqrt(variance)

def mean_chunks(l, n, selection, fraction): 
    """
    Calculate the weighted mean of chunks of a list.

    Parameters:
        l (list or array-like): The input list or array to be chunked and averaged.
        n (int): The size of each chunk.
        selection (list or array-like): A boolean list or array indicating which elements to include in the mean calculation for each chunk.
        fraction: fraction of chunks to use for sample
        
    Returns:
      tuple: A tuple containing the weighted mean and the weighted standard deviation of the chunks.
    """
    r,w=[],[]
    for i in range(0, len(l), n):
        f=l[i:i + n]
        sel=selection[i:i + n]
        w.append(np.sum(sel))
        r.append(f[sel].mean())
    bootstrap = np.random.choice(np.arange(len(r)),int(len(r)*fraction))
    r=np.array(r)[bootstrap]
    w=np.array(w)[bootstrap]
    return weighted_avg_and_std(r,w/np.sum(w)) 