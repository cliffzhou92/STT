import numpy as np
import pandas as pd
import anndata
import scvelo as scv
import scanpy as sc
import cellrank as cr
from cellrank.tl.estimators import GPCCA
from cellrank.tl.kernels import ConnectivityKernel


def dynamical_analysis(sc_object,sc_object_aggr, n_states = None, n_states_seq = None, weight_connectivities=0.2, n_components=20, thresh_ms_gene = 0, use_spatial = False, spa_weight = 0.5, spa_conn_key = 'spatial'):
    """
    Perform STT dynamical analysis on a single-cell transcriptomics dataset.

    Parameters:
    -----------
    sc_object : AnnData object
        Annotated data matrix with rows for cells and columns for genes.
    sc_object_aggr : AnnData object
        Annotated data matrix with rows for cell aggregates and columns for genes.
    n_states : int, optional
        Number of macrostates to compute. If None, n_states_seq must be provided.
    n_states_seq : list of int, optional
        Sequence of number of macrostates to compute. If None, n_states must be provided.
    weight_connectivities : float, optional
        Weight of gene expression similarity connectivities in computing the transition matrix.
    n_components : int, optional
        Number of components to compute in the Schur decomposition.
    thresh_ms_gene : float, optional
        Threshold for selecting genes based on their mean squared expression.
    use_spatial : bool, optional
        Whether to use spatial information in computing the transition matrix.
    spa_weight : float, optional
        Weight of spatial similarity in computing the transition matrix.
    spa_conn_key : str, optional
        Key for accessing the spatial connectivities in the AnnData object.

    Returns:
    --------
    None
    """
    gene_select = sc_object.var['r2'][sc_object.var['r2']>thresh_ms_gene].index.tolist()
    gene_subset = [gene+'_u' for gene in gene_select]+gene_select
    
    kernel = cr.tl.transition_matrix(sc_object_aggr, weight_connectivities=weight_connectivities,  n_jobs=-1,scheme = 'dot_product',gene_subset = gene_subset )
    
    if use_spatial:
        spa_kernel = ConnectivityKernel(sc_object,conn_key=spa_conn_key+'_connectivities')
        spa_kernel.compute_transition_matrix()
        kernel = (1-spa_weight)*kernel + spa_weight*spa_kernel
    print(kernel)    
    g_fwd = GPCCA(kernel)
    
    g_fwd.compute_schur(n_components=n_components)
    
    if n_states == None:
        g_fwd.compute_macrostates(n_states=n_states_seq,use_min_chi = True)
    else:
        g_fwd.compute_macrostates(n_states=n_states)    
    
    P_cg = g_fwd.coarse_T.to_numpy()
    P_cg[P_cg<0] = 1e-16
    row_sums = P_cg.sum(axis=1)
    P_hat = P_cg / row_sums[:, np.newaxis]
    
    sc_object.uns['da_out'] = {}
    sc_object.uns['da_out']['P_hat'] = P_hat
    sc_object.uns['da_out']['mu_hat'] = g_fwd.coarse_stationary_distribution.to_numpy()
    sc_object.uns['da_out']['membership'] = g_fwd.macrostates_memberships.X
    sc_object.uns['da_out']['gene_select'] = gene_select
    sc_object.obsm['rho'] = g_fwd.macrostates_memberships.X
    
    sc_object_aggr.obs['attractor'] = sc_object.obs['attractor'].values
    sc_object_aggr.uns['gene_subset'] = gene_subset


def construct_tenstor(adata, rho):
    # tensor N_c*N_g*2*K
    K = rho.shape[1]
    par = np.zeros((adata.shape[1],K+1))
    tensor_v = np.zeros((adata.shape[0],adata.shape[1],2,K))
    r2 = np.zeros(adata.shape[1])
 
    for i in range(adata.shape[1]):
        U = adata.layers['Mu'][:,i]
        S = adata.layers['Ms'][:,i]
        
        if 'toarray' in dir(U):
            U = U.toarray().flatten()
            S = S.toarray().flatten()
        
        m_c = np.zeros(K)
        U_c_var = np.zeros(adata.shape[0])
        
        for c in range(K):
            m_c[c] = np.average(U,weights = rho[:,c])
        
        for k in range(adata.shape[0]):
            U_c_var[k] = np.inner((U[k]-m_c)**2,rho[k,:])
        
        par[i,K]= np.inner(U,S)/np.sum(U**2+U_c_var)  #beta
        par[i,:K] = m_c*par[i,K] #alpha
        
        U_beta = par[i,K]*U
        var_reg = np.sum((par[i,K]*U-S)**2)+np.sum(((U_beta[:,np.newaxis]-par[i,:K])**2)*rho)
        var_all = adata.shape[0]*(np.var(S)+np.var(U_beta))
        
        r2[i] = 1- var_reg/var_all
    
    U = adata.layers['Mu']
    S = adata.layers['Ms']

    if 'toarray' in dir(U):
        U = U.toarray()
        S = S.toarray()

    for i in range(K):        
        tensor_v [:,:,0,i] = (par[:,i]- U * par[:,K])
        tensor_v [:,:,1,i] = (U * par[:,K] - S)
        
        
    adata.uns['par'] = par
    adata.obsm['tensor_v'] = tensor_v
    adata.var['r2'] = r2
    
def aver_velo(tensor_v,membership):
    (N_c,N_g,_,_) = np.shape(tensor_v)
    velo = np.zeros((N_c,N_g,2))
    
    for i in range(N_c):
        for j in range(N_g):
            for k in range(2):
                velo[i,j,k] = np.dot(tensor_v[i,j,k,:] ,membership[i,:])
    
    return velo

def dynamical_iteration(adata, n_states=None, n_states_seq=None, n_iter=10, return_aggr_obj=False, weight_connectivities=0.2, n_components=20, n_neighbors=100, thresh_ms_gene=0, thresh_entropy=0.1, use_spatial=False, spa_weight=0.5, spa_conn_key='spatial', stop_cr='abs'):
    """
    Perform dynamical iteration on the given AnnData object.

    Parameters:
    -----------
    adata: AnnData object
        Annotated data matrix.
    n_states: int, optional (default: None)
        Number of attractor states.
    n_states_seq: list, optional (default: None)
        List of number of attractor states for each iteration.
    n_iter: int, optional (default: 10)
        Number of iterations.
    return_aggr_obj: bool, optional (default: False)
        Whether to return the aggregated object.
    weight_connectivities: float, optional (default: 0.2)
        Weight of connectivities.
    n_components: int, optional (default: 20)
        Number of components.
    n_neighbors: int, optional (default: 100)
        Number of neighbors.
    thresh_ms_gene: int, optional (default: 0)
        Threshold for mean spliced gene expression.
    thresh_entropy: float, optional (default: 0.1)
        Threshold for entropy.
    use_spatial: bool, optional (default: False)
        Whether to use spatial information.
    spa_weight: float, optional (default: 0.5)
        Weight of spatial information.
    spa_conn_key: str, optional (default: 'spatial')
        Key for spatial connectivities.
    stop_cr: str, optional (default: 'abs')
        Stopping criterion for iteration.

    Returns:
    --------
    None
    """
    
    adata.uns['da_out']={}
    rho = pd.get_dummies(adata.obs['attractor']).to_numpy()
    construct_tenstor(adata,rho = rho)
    adata.obsm['tensor_v_aver'] = aver_velo(adata.obsm['tensor_v'],rho) 
    adata.obsm['rho'] =rho 
    
    U = adata.layers['unspliced']
    S = adata.layers['spliced']

    if 'toarray' in dir(U):
        U = U.toarray()
        S = S.toarray()
    
    X_all = np.concatenate((U,S),axis = 1)
    
    sc_object_aggr = anndata.AnnData(X=X_all)
    sc_object_aggr.var_names = [gene+'_u' for gene in adata.var_names.to_list()]+adata.var_names.to_list()
    
    velo = adata.obsm['tensor_v_aver']
    sc_object_aggr.layers['velocity']= np.concatenate((velo[:,:,0],velo[:,:,1]),axis = 1)
    sc_object_aggr.layers['spliced'] = X_all
    
    sc_object_aggr.obs['entropy'] = -np.sum(rho*np.log(rho+1e-8),axis = 1)
    entropy = 0
    sc_object_aggr.var['highly_variable'] = True
    
    for i in range(n_iter):
        velo_orig = adata.obsm['tensor_v_aver']
        rho_orig =  rho
        entropy_orig = entropy
        
        sc.tl.pca(sc_object_aggr,use_highly_variable = True)
        sc.pp.neighbors(sc_object_aggr,n_neighbors = n_neighbors)
        dynamical_analysis(adata, sc_object_aggr, n_states = n_states,n_states_seq=n_states_seq, weight_connectivities=weight_connectivities, n_components = n_components,thresh_ms_gene = thresh_ms_gene, use_spatial = use_spatial, spa_weight = spa_weight, spa_conn_key = spa_conn_key)
        
        rho = adata.obsm['rho']
        adata.obs['attractor'] =  np.argmax(rho,axis = 1)
        adata.obs['attractor'] = adata.obs['attractor'].astype('category')
        
        construct_tenstor(adata,rho = rho)
        adata.obsm['tensor_v_aver'] = aver_velo(adata.obsm['tensor_v'],rho)
        sc_object_aggr.layers['velocity']= np.concatenate((adata.obsm['tensor_v_aver'][:,:,0],adata.obsm['tensor_v_aver'][:,:,1]),axis = 1)
        sc_object_aggr.var['highly_variable'] = [genes in sc_object_aggr.uns['gene_subset'] for genes in sc_object_aggr.var_names]
        
        
        entropy = -np.sum(rho*np.log(rho+1e-8),axis = 1)
        
        if stop_cr == 'rel':
            err_ent = np.quantile(np.abs(entropy-entropy_orig)/np.abs(entropy_orig),0.75)
        
        if stop_cr == 'abs':
            err_ent = np.amax(np.abs(entropy-entropy_orig))
        
        print(err_ent)
        
        if err_ent < thresh_entropy:
            break
    
    adata.obs['entropy'] = entropy
    adata.uns['gene_subset'] = sc_object_aggr.uns['gene_subset']
    sc_object_aggr.layers['Ms'] = np.concatenate((adata.layers['Mu'],adata.layers['Ms']),axis = 1)
    sc_object_aggr = sc_object_aggr[:,sc_object_aggr.uns['gene_subset']]
    adata.obs['speed'] = np.linalg.norm(sc_object_aggr.layers['velocity'],axis=1)
    adata.layers['velo'] = adata.obsm['tensor_v_aver'][:,:,1]
    
    if return_aggr_obj:
        sc_object_aggr.obs['entropy'] = adata.obs['entropy'].values
        sc_object_aggr.obs['speed'] = adata.obs['speed'].values
        sc_object_aggr.obs['attractor'] = adata.obs['attractor'].values
        return sc_object_aggr
    