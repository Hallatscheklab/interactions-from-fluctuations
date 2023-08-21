import numpy as np
from influ.base import Base
import cvxpy as cp

class F2(Base):
    """
    
    Class that infers migration matrix starting from a matrix of counts and non_missing tot_counts
    both counts and tot_counts should have shape: NDxNtrajxT 
    tot_counts directly refers to the number of non-missing counts for that specific allele
    
    """
        
    def infer(self,chunks_size=5000):
        
        """ 
        This is the main function that runs the whole analysis.
        It returns the inferred matrix.
        """
        self.chunks_size=chunks_size
        self.from_counts_to_F2s()
        self.A=self.infer_A(self.F_X1_X0,self.F_X0_X0)
        return self.A.copy()

    def from_counts_to_F2s(self):
        """ 
        This function builds the F2 and F2' matrices needed for inference
        it returns an array of matrices, where the first index is T-1 long
        """
        def mean_chunks(l, n): 
            return np.array([np.mean(l[i:i + n],axis=0) for i in range(0, len(l), n)])

        # inference with the right amount of time points
        freq=self.counts/self.totcounts
        correction_factor = freq*(1-freq)/(self.totcounts-1) # Patterson et al
        X0,X1=freq[:,:,:-1],freq[:,:,1:]
        correction_factor0,correction_factor1 = correction_factor[:,:,:-1],correction_factor[:,:,1:]
        self.F_X1_X0=np.zeros((X0.shape[-1],freq.shape[0],freq.shape[0]))
        self.F_X0_X0=np.zeros((X0.shape[-1],freq.shape[0],freq.shape[0]))
        self.F_X1_X0_std=np.zeros((X0.shape[-1],freq.shape[0],freq.shape[0]))
        self.F_X0_X0_std=np.zeros((X0.shape[-1],freq.shape[0],freq.shape[0]))
        for i in range(freq.shape[0]):
            for j in range(freq.shape[0]):
                self.F_X1_X0[:,i,j]= np.mean(mean_chunks((X1[i]-X0[j])**2  - correction_factor0[i]- correction_factor1[j],self.chunks_size),axis=0)
                self.F_X1_X0_std[:,i,j]= np.std(mean_chunks((X1[i]-X0[j])**2  - correction_factor0[i]- correction_factor1[j],self.chunks_size),axis=0)

                if i!=j:
                    self.F_X0_X0[:,i,j]= np.mean(mean_chunks((X0[i]-X0[j])**2 - correction_factor0[i]- correction_factor0[j],self.chunks_size),axis=0)
                    self.F_X0_X0_std[:,i,j]= np.std(mean_chunks((X0[i]-X0[j])**2 - correction_factor0[i]- correction_factor0[j],self.chunks_size),axis=0)

                else:
                    self.F_X0_X0[:,i,j]= np.mean(mean_chunks((X0[i]-X0[j])**2,self.chunks_size),axis=0)
                    self.F_X0_X0_std[:,i,j]= np.mean(mean_chunks((X0[i]-X0[j])**2,self.chunks_size),axis=0)
    
                    
    def infer_A(self,F_X1_X0,F_X0_X0):
        """ 
        Here we compute the migration matrix
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