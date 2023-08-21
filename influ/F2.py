import numpy as np
from influ.base import Base
import cvxpy as cp

class F2(Base):
    """
    
    Class that infers migration matrix starting from a matrix of counts and non_missing tot_counts
    both counts and tot_counts should have shape: NDxNtrajxT 
    tot_counts directly refers to the number of non-missing counts for that specific allele
    
    """
        
    def infer(self,chunks_size=1000):
        
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
        Divides the genome into blocks of equal size and then does wighted average across block 
        by weighting with the number of non-missing alleles within the block
        """
        def weighted_avg_and_std(values, weights):
            average = np.average(values, weights=weights)
            variance = np.average((values-average)**2, weights=weights)
            return average,np.sqrt(variance)

        def mean_chunks(l, n, selection): 
            r,w=[],[]
            for i in range(0, len(l), n):
                f=l[i:i + n]
                sel=selection[i:i + n]
                w.append(np.sum(sel))
                r.append(f[sel].mean())
            return weighted_avg_and_std(np.array(r),np.array(w)/np.sum(w))    
        
        ND,Ntraj,T=self.counts.shape
        freq=self.counts/self.totcounts
        correction_factor = freq*(1-freq)/(self.totcounts-1) # Patterson et al
        X0,X1=freq[:,:,:-1],freq[:,:,1:]
        correction_factor0,correction_factor1 = correction_factor[:,:,:-1],correction_factor[:,:,1:]
        tot_counts0,tot_counts1 = self.totcounts[:,:,:-1],self.totcounts[:,:,1:]
        self.F_X1_X0=np.zeros((X0.shape[-1],freq.shape[0],freq.shape[0]))
        self.F_X0_X0=np.zeros((X0.shape[-1],freq.shape[0],freq.shape[0]))
        self.F_X1_X0_std=np.zeros((X0.shape[-1],freq.shape[0],freq.shape[0]))
        self.F_X0_X0_std=np.zeros((X0.shape[-1],freq.shape[0],freq.shape[0]))
        for i in range(freq.shape[0]):
            for j in range(freq.shape[0]):        
                 for t in range(T-1):
                    selection_lineages=np.logical_and(tot_counts1[i,:,t]>1,tot_counts0[j,:,t]>1)
                    f10=(X1[i,:,t]-X0[j,:,t])**2  - correction_factor1[i,:,t]- correction_factor0[j,:,t]
                    self.F_X1_X0[t,i,j],self.F_X1_X0_std[t,i,j]=mean_chunks(f10,self.chunks_size,selection_lineages)
                    selection_lineages=np.logical_and(tot_counts0[i,:,t]>1,tot_counts0[j,:,t]>1)
                    if i!=j: f00=(X0[i,:,t]-X0[j,:,t])**2 - correction_factor0[i,:,t]- correction_factor0[j,:,t]
                    else: f00=(X0[i,:,t]-X0[j,:,t])**2
                    self.F_X0_X0[t,i,j],self.F_X0_X0_std[t,i,j]=mean_chunks(f00,self.chunks_size,selection_lineages)
                    
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