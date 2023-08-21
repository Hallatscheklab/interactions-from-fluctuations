import scipy.odr as odr
import numpy as np
from influ.base import Base

class DMD(base):
    """
    
    Class that infers migration matrix starting from a matrix of counts and non_missing tot_counts
    both counts and tot_counts should have shape: NDxNtrajxT 
    tot_counts directly refers to the number of non-missing counts for that specific allele
    
    """
    
    def __init__(self, counts=[], totcounts=[]):
        self.counts=counts
        self.totcounts=totcounts
        
    def infer(self,we=1,lam=0):
        """ 
        This is the main function that runs the inference
        It returns the inferred matrix.
        """
        if we==1:
            self.A=self.lindyn_qp(self.counts/self.totcounts,lam=0)
        else:
            self.A=self.TLS_scipy(self.counts/self.totcounts,we=we)
        return self.A.copy()

    
    def predict(self):
        
        """ 
        Do we want to have this function?
        """
        return None




    def TLS_scipy(self, B, we=1,wd=1):
        ND = B.shape[-1]
        X=B[:-1,:,:].reshape((-1,ND), order='F').T 
        Xt=B[1:,:,:].reshape((-1,ND), order='F').T
        def odr_line(B, x):return  B @ x
        a=[]
        for i in range(ND):
            quadr = odr.Model(odr_line)
            mydata = odr.Data(X, Xt[i],we=we,wd=wd)
            myodr = odr.ODR(mydata, quadr, beta0=np.ones(ND))
            output = myodr.run()
            a.append(output.beta)
        return np.array(a)



    def lindyn_qp(self, B, lam=0):
    #For each row i of A, find Aopt_ij = argmin_{A_ij} sum_t (B2_it - sum_j B1_jt A_ij)^2  + lam*T*\sum_{j\neq i} A_ij
    #with A_ij>0 and sum_j A_ij=1
    #lam (>=0) is the LASSO-regularization parameter 
    # Feb22: A_ii is allowed to be negative

        ND = B.shape[-1]
        B1=B[:-1,:,:].reshape((-1,ND), order='F').T 
        B2=B[1:,:,:].reshape((-1,ND), order='F').T
        solvers.options['show_progress'] = False
        ND, NT = B2.shape
        Aopt=np.zeros((ND,ND)) 
        P=2*matrix(np.dot(B1,B1.T))
        Amin=0.0001
        h = matrix(np.full(ND,-1*Amin))
        A = matrix(np.full(ND,1.0), (1,ND))
        b = matrix(1.0)
        for i in range(ND):
            vec_lasso=np.array([lam*NT]*ND)
            vec_lasso[i]=0
            q=-2*matrix(np.matmul(B1,B2[i])-vec_lasso)
            G = matrix(np.diag(np.full(ND,-1.0)))
            sol=solvers.qp(P, q, G, h, A, b)
            Aopt[i]=np.array([sol['x'][i] for i in range(len(q))])
        return Aopt