#EM algorithm Kalman

import numpy as np
from numpy.linalg import inv as minv
from cvxopt import matrix, solvers
import random
import os
import time
from influ.base import Base

class Kalman(Base):

    def infer(self,frac=0.5, Ne_old=1000,em_step_max=100,terminate_th=0.001,infer_samplenoise=True,noisemode=2,ridge=0.,penalty_mode='L2'):
        '''
        counts = (ND,Nlin,T) -> B = (T, Nlin, ND) 
        counts_deme = (ND,lin,T)
        em_step_max: Max of EM steps
        terminate_th: Terminate if the likelihood improvement < terminate_th
        '''
        self.lnLH_record=[]
        self.counts_deme=self.totcounts.copy()
        self.B=(self.counts/self.counts_deme).transpose([2,1,0])
        self.T, self.Nlins, self.ND=self.B.shape
        self.frac=frac
        self.A_LS=self.lindyn_qp(lam=0)
        np.random.seed((os.getpid() * int(time.time())) % 123456789)
        self.Arand = np.array( [np.random.dirichlet(([1]*self.ND),size=1)[0] for i in range(self.ND)])
        self.A_old=np.copy(self.frac*self.A_LS+ (1.-self.frac)*self.Arand) # Start from a mixture of A_LS & homogenous
        self.Csn_old=np.ones(self.ND)
        self.Ne_old=np.array([Ne_old]*self.ND)
        return self.EM(em_step_max=em_step_max,terminate_th=terminate_th,infer_samplenoise=infer_samplenoise,noisemode=noisemode, 
                       ridge=ridge,penalty_mode=penalty_mode)[1].copy()
        
    def EM(self,em_step_max=100,terminate_th=0.001,infer_samplenoise=True,noisemode=2, ridge=0.0,penalty_mode='L2'):
        ND=self.ND
        for step in range(em_step_max):
            lnLH_all=0

            # Quantities writtein as E[O] in Bishop's textbook normalized by heterozygosity
            self.Ezz_all =np.zeros((ND,ND,ND),dtype='float32') #i,j,k component = sum_{lin,t} E[zt-1 zt-1]_jk/H_it, t =1,...,T-1. H it = f_it(1-f_it)
            self.Ezz_pre_all =np.zeros((ND,ND,ND),dtype='float32') # sum_{lin,t} E[zt zt-1]_jk/H_it, t =1,...,T-1.
            self.Ezz_all_shift =np.zeros((ND,ND,ND),dtype='float32') # sum_{lin,t} E[zt zt]_jk/H_it, t =1,...,T-1.
            Csn_new=np.zeros(ND,dtype='float32') # Strength of sample noise

            for lin in range(self.Nlins):
                mu_star=self.B[0,lin,:]
                V_star= np.diag([self.Csn_old[i]*(mu_star[i]*(1-mu_star[i]))/self.counts_deme[i,lin,0] for i in range(ND)])
                x=self.B[:,lin,:].T # NDxT
                mu_filter, V_filter,P_filter, lnLH=self.Kfilter(x, mu_star, V_star, self.A_old, self.counts_deme[:,lin], self.Ne_old, self.Csn_old, noisemode=noisemode)    
                mu_smoother, V_smoother, J = self.Ksmoother(self.A_old, mu_filter, V_filter,P_filter)
                Ez, Ezz_pre, Ezz=    self.Expvals(mu_smoother, V_smoother, J)# Quantities writtein as E[O] in Bishop's textbook 

                hetero =np.zeros((ND,self.T))
                if noisemode==0: hetero =x*(1-x)  # Hetetozygosity, ND*T
                elif noisemode==1: hetero =np.ones((ND,self.T))*np.mean(x)*(1-np.mean(x))
                elif noisemode==2: 
                    for i in range(ND): hetero[i] = np.ones(self.T)*np.mean(x[i])*(1-np.mean(x[i]))
                hetero[:,0]=np.nan # The first point is not used.
                Eztr=Ez.transpose().copy()# x,Eztr,hetero:NDxT
                for i in range(ND):
                    invhetero= np.diag(1./hetero[i,1:])
                    self.Ezz_all[i]+=np.sum(np.tensordot(invhetero, Ezz[:-1],axes=(1,0)),axis=0)
                    self.Ezz_all_shift[i] +=np.sum(np.tensordot(invhetero, Ezz[1:],axes=(1,0)),axis=0)
                    self.Ezz_pre_all[i]+=np.sum(np.tensordot(invhetero, Ezz_pre[1:],axes=(1,0)),axis=0)
                    Csn_new[i]+=np.sum((x[i,1:]*x[i,1:]-2*x[i,1:]*Eztr[i,1:]+ Ezz[1:,i,i])*self.counts_deme[i,lin,1:]/hetero[i,1:])
                lnLH_all+=lnLH

            self.Ezz_all*=1.0/(self.Nlins*(self.T-1))
            self.Ezz_pre_all*=1.0/(self.Nlins*(self.T-1))
            self.Ezz_all_shift*=1.0/(self.Nlins*(self.T-1))
            Csn_new*=1.0/(self.Nlins*(self.T-1))

            #M-step (Update parameters)
            if penalty_mode=='L2':
                A_new=np.copy(self.EM_A(self.Ezz_all, self.Ezz_pre_all,'cstr',ridge=ridge))
            elif penalty_mode=='L1':
                A_new=np.copy(self.EM_A_L1(self.Ezz_all, self.Ezz_pre_all,'cstr',ridge=ridge))
            DA =  np.max(np.abs(A_new-self.A_old))
            self.A_old = np.copy(A_new)

            Ne_new=np.copy(self.EM_Neff(self.A_old,  self.Ezz_all,self.Ezz_all_shift, self.Ezz_pre_all))
            DNe = np.max(np.abs(Ne_new-self.Ne_old)/self.Ne_old)
            self.Ne_old = np.copy(Ne_new)
            for i in range(ND): 
                if Csn_new[i]<1: Csn_new[i]=1
            if not infer_samplenoise:
                for i in range(ND): Csn_new[i]=1
            DCsn = np.max(np.abs(Csn_new-self.Csn_old)/self.Csn_old)
            self.Csn_old = Csn_new.copy()
            
            self.lnLH_record.append(lnLH_all)

            if step>1:
                if self.lnLH_record[-1] - self.lnLH_record[-2]<0:
                    print("LH decreses @")
                    if np.abs(self.lnLH_record[-1] - self.lnLH_record[-2])/np.abs(self.lnLH_record[-1]) >0.01:
                        print('Error?')   
            if DA<terminate_th and DNe<0.025 and DCsn<0.025:
                print("terminate at step={}, DA={}, ratioDNe={}".format(step,np.round(DA,5),np.round(DNe,5))+', ratioDCsn='+str(np.round(DCsn,5)))
                break
        
        return self.lnLH_record, self.A_old, self.Ne_old, self.A_LS, self.Csn_old

    def logGauss(self,x, mu, cov):
        vec=np.array([x-mu])
        lamlist=np.linalg.eigvals(cov)
        sumloglam=0
        for i in  lamlist:
            sumloglam+=np.log(abs(i))
        k = len(x)
        logGauss=(-0.5 *vec @ minv(cov) @ vec.T)[0,0] - 0.5*k*np.log(2*3.1415926535)  -0.5*sumloglam
        return logGauss

    def filter_initial(self,x0, mu_pre, V_pre,  Sigma):
        M = len(Sigma)
        I = np.identity(M)

        K0 = V_pre@minv( (V_pre) + Sigma)
        mu0 = mu_pre + K0@(x0-mu_pre)
        V0 =(I-K0)@V_pre 

        normal_mean = mu_pre
        normal_cov =  V_pre + Sigma
        lnc_0= self.logGauss(x0, normal_mean, normal_cov)
        return mu0, V0, lnc_0

    def filter_later(self,x_n, mu_old, V_old, A,  Sigma,Gamma):
        M = len(Sigma)
        I = np.identity(M)

        P_old = A@V_old@A.T + Gamma
        K_n = P_old@minv( (P_old) + Sigma)
        mu_n = A@ mu_old + K_n @(x_n-A@mu_old)
        V_n =(I-K_n)@P_old 

        normal_mean = A@mu_old
        normal_cov =  P_old + Sigma
        lnc_n = self.logGauss(x_n, normal_mean, normal_cov)
        return mu_n, V_n, P_old, lnc_n 


    def Kfilter(self,x, mu_star, V_star, A,  counts_deme, Ne, Csn, noisemode):
        '''
        Kalman Filter Algorithm. Assumes Hetereskedastic noise: at each time we have a different Sigma/Gamma.
        x:time series of vector ( ND times T)
        mu_star, V_star: the initial hidden-state distribuion
        A:evolution matrix (ND times ND)
        C:deterministic realtion between hidden states and observables, here, C=I.
        Nsample: controls the noise in emission
        Ne: controls the noise in hedden-state transition 
        '''
        ND = len(A)
        lnLH=0
        mu_filter=[]
        V_filter=[]
        P_filter=[]
        Ginv_filter=[]

        Sigma = np.diag([ Csn[i]*mu_star[i]*(1-mu_star[i])/counts_deme[i,0]  for i in range(ND)])
        mu, V, lnc_0=self.filter_initial(x[:,0], mu_star, V_star, Sigma)
        lnLH+=lnc_0
        mu_filter.append(mu)
        V_filter.append(V)
        freqave=np.mean(x)
        for t in range(1,len(x[0])):
            Amu=A@mu
            if noisemode==0:
                Sigma = np.diag([Csn[i]*x[i,t]*(1-x[i,t])/(counts_deme[i,t])  for i in range(ND)])
                Gamma = np.diag([ x[i,t]*(1-x[i,t])/Ne[i] for i in range(ND)])
            elif noisemode==1:
                Sigma = np.diag([Csn[i]*freqave*(1-freqave)/(counts_deme[i,t])  for i in range(ND)])
                Gamma = np.diag([ freqave*(1-freqave)/Ne[i] for i in range(ND)])
            elif noisemode==2:
                xaux = np.mean(x,axis=1)
                Sigma = np.diag([Csn[i]* xaux[i]*(1-xaux[i])/(counts_deme[i,t])  for i in range(ND)])
                Gamma = np.diag([ xaux[i]*(1-xaux[i])/Ne[i] for i in range(ND)])
            mu, V, P, lnc_n= self.filter_later(x[:,t], mu, V, A,  Sigma,Gamma)
            lnLH+=lnc_n
            mu_filter.append(mu)
            V_filter.append(V)
            P_filter.append(P)

        P_filter.append(A@V@A.T + Gamma)
        return np.array(mu_filter),np.array(V_filter), np.array(P_filter),  lnLH


    def Ksmoother (self,A, mu_filter, V_filter,P_filter):
        '''
        Kalman Smoother Algorithm
        '''
        T = len(mu_filter)
        ND =len(A)
        mu_smoother=np.zeros((T,ND))
        V_smoother=np.zeros((T,ND,ND))
        J=np.zeros((T,ND,ND))
        mu_smoother[-1]=mu_filter[-1]
        V_smoother[-1]=V_filter[-1]
        J[-1] =  V_filter[-1]@A.transpose()@minv(P_filter[-1])
        for t in reversed(range(len(mu_filter)-1)):
            J[t] =  V_filter[t]@A.transpose()@minv(P_filter[t])
            mu_smoother[t] = mu_filter[t] + J[t] @ (mu_smoother[t+1] - A@mu_filter[t] )
            V_smoother[t] = V_filter[t] + J[t]@(V_smoother[t+1] - P_filter[t])@J[t].transpose()
        return mu_smoother, V_smoother, J

    def Expvals(self,mu_smoother, V_smoother, J):
        T,ND =mu_smoother.shape
        Ez = np.copy(mu_smoother)
        Ezz_pre = np.zeros((T,ND,ND))
        Ezz = np.zeros((T,ND,ND))
        for t in range(T):
            Ezz[t] = V_smoother[t] + np.array([mu_smoother[t]]).transpose()@np.array([mu_smoother[t]])
        for t in range(1,T): 
            Ezz_pre[t] = V_smoother[t]@J[t-1].transpose() + np.array([mu_smoother[t]]).transpose()@np.array([mu_smoother[t-1]])
        Ezz_pre[0]=100000 #Ezz_pre[0] is not defined 
        # NOTE: Typo in the expression E[z[t]z[t-1]] in Bishop's textbook: Correctly, E[z_n z_{n-1}^T] = \hat V_n J_{n-1}^T + \hat\mu_n \hat\mu_{n-1}^T
        return Ez, Ezz_pre, Ezz

    def EM_A(self,Ezz_all, Ezz_pre_all, mode,ridge=0):
        solvers.options['show_progress'] = False
        ND = len(Ezz_all)
        A_new =np.zeros((ND,ND))
        qpA = matrix(np.full(ND,1.0), (1,ND))
        #Ridge
        XTX=np.mean(Ezz_all,axis=0)
        lam = np.mean(XTX)*ridge
        Amin=1e-6
        h = matrix(np.full(ND,-1*Amin))
        b = matrix(1.0)
        for i in range(ND):
            G = matrix(np.diag(np.full(ND,-1.0)))        
            scale=np.mean([np.mean(Ezz_all[i]),np.mean(Ezz_pre_all[i,i])])
            P=matrix(np.copy((Ezz_all[i]+lam)/scale).astype(np.double))
            q=-1*matrix(np.copy(Ezz_pre_all[i,i]/scale).astype(np.double))
            if mode=='cstr':sol=solvers.qp(P, q, G, h, qpA, b)
            else:sol=solvers.qp(P, q) #no constraints
            A_new[i]=np.array([sol['x'][j] for j in range(len(q))])

        return A_new    
    
    def EM_A_L1(self,Ezz_all, Ezz_pre_all, mode,ridge=0,):
        solvers.options['show_progress'] = False
        ND = len(Ezz_all)
        A_new =np.zeros((ND,ND))
        qpA = matrix(np.full(ND,1.0), (1,ND))
        #Ridge
        XTX=np.mean(Ezz_all,axis=0)
        lam = np.mean(XTX)*ridge
        Amin=1e-6
        h = matrix(np.full(ND,-1*Amin))
        b = matrix(1.0)
        for i in range(ND):
            G = matrix(np.diag(np.full(ND,-1.0)))        
            scale=np.mean([np.mean(Ezz_all[i]),np.mean(Ezz_pre_all[i,i])])
            P=matrix(np.copy((Ezz_all[i])/scale).astype(np.double))
            L1_i_removed=np.ones(ND)
            L1_i_removed[i]=0.0
            if lam>0:q=-1*matrix(np.copy((Ezz_pre_all[i,i]-lam*L1_i_removed)/scale).astype(np.double))
            else:q=-1*matrix(np.copy(Ezz_pre_all[i,i]/scale).astype(np.double))
            if mode=='cstr':sol=solvers.qp(P, q, G, h, qpA, b)
            else:sol=solvers.qp(P, q) #no constraints
            A_new[i]=np.array([sol['x'][j] for j in range(len(q))])
        return A_new

    def EM_Neff(self,A, Ezz_all, Ezz_all_shift,Ezz_pre_all):
        ND = len(A)
        Ne_new =np.zeros(ND)
        for i in range(ND):
            aux_sum = Ezz_all_shift[i,i,i] + -2*Ezz_pre_all[i,i]@A[i] + A[i]@Ezz_all[i]@A[i]
            Ne_new[i]=1/aux_sum
        return Ne_new

    def lindyn_qp(self,lam=0):
        '''
        For each row i of A, find Aopt_ij = argmin_{A_ij} sum_t (B2_it - sum_j B1_jt A_ij)^2  + lam*T*\sum_{j\neq i} A_ij
        with A_ij>0 and sum_j A_ij=1
        lam (>=0) is the LASSO-regularization parameter
        '''
        B=self.B
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