import numpy as np
import matplotlib.pyplot as plt

def WF_sim(Npop,counts_per_demeweek, Csn, ND, T, A=None,Ntraj=100, freqini=[],sigma=0.5,low=-3,high=-1):
    
    if type(Npop)==int:
        NS=np.array([Npop]*ND) #Controls the strength of stochasticity (in transition probability)
    else:# if the list of Npop is provided
        NS = Npop 
    
    B=np.zeros((T,Ntraj,ND))
    
    if freqini ==[]: 
        # random allele frequencies
        X0=10**(np.random.uniform(low=low, high=high,size=Ntraj))
        for l in range(Ntraj): 
            for i in range(ND):
                X=np.clip(X0[l]+np.random.randn()*np.sqrt(X0[l])*sigma,0,1)
                B[0,l,i]=np.random.binomial(NS[i], X, size=1)/NS[i]
        
    for t in range(1,T):
        aux = B[t-1,:,:].T
        aux = np.matmul(A,aux)
        for l in range(Ntraj): 
            for i in range(ND):
                B[t,l,i] =np.random.binomial(NS[i],aux[i,l], size=1)/NS[i]
 
    # Add sampling error
    if type(counts_per_demeweek)==int:
        counts_deme=np.ones((ND,T))*counts_per_demeweek
    else:
        counts_deme=counts_per_demeweek
    counts=np.zeros(B.shape)
    for t in range(T):
            for j in range(ND):
                for l in range(Ntraj):
                    counts[t,l,j]  =np.random.binomial(int(counts_deme[j,t]/Csn[j]), B[t,l,j], size=1)*Csn[j]
    counts=counts.transpose([2,1,0])

    return A,counts,B



def calc_A_mean_low_up(res_A_mcmc,alpha):
    ND = len(res_A_mcmc[0])
    A_mean=np.zeros((ND,ND))
    A_low=np.zeros((ND,ND))
    A_up=np.zeros((ND,ND))
    for row in range(ND):
        for col in range(ND):
            A_mean[row,col] = np.mean(res_A_mcmc[:,row,col])
            [lower,upper]=CI(res_A_mcmc[:,row,col],alpha=alpha) # alpha=0.5 corresponds to the upper/lower quartiles 
            A_low[row,col]=lower
            A_up[row,col]=upper
            
    return A_mean,A_low,A_up

def CI(data,alpha):
    sortdata=np.sort(data)
    return [sortdata[round(0.5*alpha*len(data))],sortdata[-round(0.5*alpha*len(data))]]

def plot_mat_heatmap_offdiag(adj, mat_mean,mat_low,mat_up,vmax,vmin, plt_title, index, mode,filename,outpath='fig/',figsize=None):
    Path(outpath+filename+'/').mkdir(parents=True, exist_ok=True)
    
    def square(x_cell,y_cell):
        cl='red'
        delta=0.025
        plt.hlines(y=y_cell+delta, xmin=x_cell+delta, xmax=x_cell+1-delta, linewidth=2, color=cl)
        plt.hlines(y=y_cell+1-delta, xmin=x_cell+delta, xmax=x_cell+1-delta, linewidth=2, color=cl)
        plt.vlines(x=x_cell+delta, ymin=y_cell+delta, ymax=y_cell+1-delta, linewidth=2, color=cl)
        plt.vlines(x=x_cell+1-delta, ymin=y_cell+delta, ymax=y_cell+1-delta, linewidth=2, color=cl)

    maxlen = max([len(i) for i in index])
    if maxlen >10:
        rot_x = 90
    else:
        rot_x =0
    kw={"labels":index,"fontsize":20,"rotation":rot_x}
    kw_y={"labels":index,"fontsize":20,"rotation":0}
    
    if figsize!=None:
        fig, ax = plt.subplots(figsize=(figsize[0],figsize[1]))
    
    ND = len(index)
    
    mat_mean_offdiag=np.copy(mat_mean)
    for i in range(ND):
        mat_mean_offdiag[i,i]= -1 #np.min(mat_mean)-0.01
        
    cmap = copy.copy(plt.get_cmap("YlGnBu"))
    cmap.set_under('Darkgray')
    
   
    ax=sns.heatmap(mat_mean_offdiag,# annot=mat_label,
                   linewidth=0.3, fmt='',cmap=cmap, cbar=True,vmax=vmax,vmin=vmin
                  )

    plt.xticks(ticks=np.arange(0.5,ND+0.5,1), **kw)
    plt.yticks(ticks=np.arange(0.5,ND+0.5,1), **kw_y)
    plt.xlabel('FROM',fontsize=20)
    plt.ylabel('TO',fontsize=20)
    plt.title(plt_title,fontsize=20)    
    for pair in adj:
        ind1=index.index(pair[0])
        ind2=index.index(pair[1])
        square(x_cell=ind1,y_cell=ind2)
        square(x_cell=ind2,y_cell=ind1)
        
    # use matplotlib.colorbar.Colorbar object
    cbar = ax.collections[0].colorbar
    cbar.ax.tick_params(labelsize=20)

    vmean=0.5*(vmax+vmin)
    for r in range(ND):
        for c in range(ND):
            if mat_mean[r,c] >vmean:
                cl='white'
            else:
                cl ='black'
            ax.annotate(str(np.round(mat_mean[r,c],2)), (c+0.25, r+0.4), fontsize=20, color=cl)
            ax.annotate('['+str(np.round(mat_low[r,c],2))+', '+str(np.round(mat_up[r,c],2))+']', (c+0.1, r+0.8), fontsize=12, color=cl)
    plt.savefig(outpath+filename+'/'+'Aheatmap'+filename+'.pdf',bbox_inches='tight')  
    plt.show() 
    
    
    
def plot_mat_heatmap_with_diag(mat_mean,mat_low,mat_up, plt_title, index,filename,outpath='fig/',figsize=None, vm=None):
    Path(outpath+filename+'/').mkdir(parents=True, exist_ok=True)

    kw={"labels":index,"fontsize":12,"rotation":90}
    kw_y={"labels":index,"fontsize":12,"rotation":0}
    
    if figsize!=None:
        fig, ax = plt.subplots(figsize=(figsize[0],figsize[1]))
    
    ND = len(index)
    mat_label=make_txt_heatmap(mat_mean,mat_up,mat_low)  
    
    if vm !=None:
        ax=sns.heatmap(mat_mean, annot=mat_label,vmin=vm[0],vmax=vm[1], fmt='', cmap="YlGnBu",cbar=True)
    else:
        ax=sns.heatmap(mat_mean, annot=mat_label, fmt='', cmap="YlGnBu",cbar=True)
    plt.xticks(ticks=np.arange(0.5,ND+0.5,1), **kw)
    plt.yticks(ticks=np.arange(0.5,ND+0.5,1), **kw_y)
    plt.xlabel('FROM',fontsize=15)
    plt.ylabel('TO',fontsize=15)
    plt.title(plt_title,fontsize=15)
    plt.savefig(outpath+filename+'/'+'Aheatmap_withdiag_'+filename+'.pdf',bbox_inches='tight')  
    plt.show()  
    
def make_txt_heatmap(mean, up, low, mode='with_err'):
    
    label=[]
    ND = len(mean)
    for i in range(ND):
        if mode =='with_err':
            aux=[str(np.round(mean[i,j],2))+'\n ['+str(np.round(low[i,j],2))
                 +', '+str(np.round(up[i,j],2))+']' for j in range(ND)]
        else:
            aux = [str(np.round(mean[i,j],2))  for j in range(ND)]
        label.append(aux)
    return np.array(label)