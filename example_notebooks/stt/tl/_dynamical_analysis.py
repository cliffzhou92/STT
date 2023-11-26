import numpy as np
import pandas as pd
import anndata
import scvelo as scv
import scanpy as sc
import cellrank as cr
from cellrank.tl.estimators import GPCCA
from cellrank.tl.kernels import ConnectivityKernel
from sklearn.model_selection import train_test_split


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
    gene_select = sc_object.var['r2_test'][sc_object.var['r2_test']>thresh_ms_gene].index.tolist()
    gene_subset = [gene+'_u' for gene in gene_select]+gene_select
    r2_keep_test = sc_object.var['r2_test'][sc_object.var['r2_test']>thresh_ms_gene]
    r2_keep_train = sc_object.var['r2_train'][sc_object.var['r2_test']>thresh_ms_gene]   


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
    sc_object.uns['r2_keep_train'] = r2_keep_train
    sc_object.uns['r2_keep_test'] = r2_keep_test



def construct_tenstor(adata, rho, portion = 0.8):
    # tensor N_c*N_g*2*K
    K = rho.shape[1]
    par = np.zeros((adata.shape[1],K+1))
    tensor_v = np.zeros((adata.shape[0],adata.shape[1],2,K))
    r2_train = np.zeros(adata.shape[1])
    r2_test = np.zeros(adata.shape[1])
 
    U = adata.layers['Mu']
    S = adata.layers['Ms']
    if 'toarray' in dir(U):
        U = U.toarray()
        S = S.toarray()

    label_all = adata.obs['attractor']
    categories = np.unique(label_all)
    U_train, U_test, S_train, S_test, rho_train, rho_test,label,_ = train_test_split(U, S, rho, label_all, test_size=1-portion, random_state=42)

    for i in range(adata.shape[1]):
        u_train = U_train[:,i]
        s_train = S_train[:,i]
        u_test = U_test[:,i]
        s_test = S_test[:,i]
        
        if 'toarray' in dir(u_train):
            u_train = u_train.toarray().flatten()
            s_train = s_train.toarray().flatten()
        
        # Initialize an empty list to store indices for each category
        indices_per_category = []

        for category in categories:
            # Get indices for the current category
            cat_indices = np.where(label == category)[0]

            # Extract values of u_train and s_train for the current category
            u_train_cat = u_train[cat_indices]
            s_train_cat = s_train[cat_indices]
            if len(u_train_cat[u_train_cat>0])>0:
                u_q = u_train_cat[u_train_cat>0]
            else:
                u_q = u_train_cat
            if len(s_train_cat[s_train_cat>0])>0:
                s_q = s_train_cat[s_train_cat>0]
            else:
                s_q = s_train_cat



            # Compute 10% and 90% quantiles for each array within the current category
            u_10, u_90 = np.quantile(u_q, [0.1, 0.9])
            s_10, s_90 = np.quantile(s_q, [0.1, 0.9])

            # Find indices where both arrays are within their respective 10%-90% quantiles for the current category
            selected_indices = cat_indices[np.where((u_train_cat >= u_10) & (u_train_cat <= u_90) & (s_train_cat >= s_10) & (s_train_cat <= s_90))[0]]

            # Add these indices to the list
            indices_per_category.append(selected_indices)

        # Union of all selected indices across categories
        indices = np.unique(np.concatenate(indices_per_category))
        if len(indices>=10):
            # Extract the values from u_train and s_train using the indices
            u_train_selected = u_train[indices]
            s_train_selected = s_train[indices]
            rho_train_selected = rho_train[indices,:]

            m_c = np.zeros(K)
            U_c_var = np.zeros(u_train_selected.shape[0])
            
            for c in range(K):
                if np.max(rho_train_selected[:,c])>0:
                    m_c[c] = np.average(u_train_selected,weights = rho_train_selected[:,c])
                else:
                    m_c[c] = 0

            
            for k in range(u_train_selected.shape[0]):
                U_c_var[k] = np.inner((u_train_selected[k]-m_c)**2,rho_train_selected[k,:])
            
            par[i,K]= np.inner(u_train_selected,s_train_selected)/np.sum(u_train_selected**2+U_c_var)  #beta
            par[i,:K] = m_c*par[i,K] #alpha
            
            U_beta_train = par[i,K]*u_train
            U_beta_test = par[i,K]*u_test
            var_reg_train = np.sum((par[i,K]*u_train-s_train)**2)+np.sum(((U_beta_train[:,np.newaxis]-par[i,:K])**2)*rho_train)
            var_reg_test = np.sum((par[i,K]*u_test-s_test)**2)+np.sum(((U_beta_test[:,np.newaxis]-par[i,:K])**2)*rho_test)
            var_all_train = U_train.shape[0]*(np.var(s_train)+np.var(U_beta_train))
            var_all_test = U_test.shape[0]*(np.var(s_test)+np.var(U_beta_test))
            
            r2_train[i] = 1- var_reg_train/var_all_train
            r2_test[i] = 1- var_reg_test/var_all_test
        else:
            r2_train[i] = -100
            r2_test[i] = -100

        

    for i in range(K):        
        tensor_v [:,:,0,i] = (par[:,i]- U * par[:,K])
        tensor_v [:,:,1,i] = (U * par[:,K] - S)
        
        
    adata.uns['par'] = par
    adata.obsm['tensor_v'] = tensor_v
    adata.var['r2_train'] = r2_train
    adata.var['r2_test'] = r2_test
    
def aver_velo(tensor_v,membership):
    (N_c,N_g,_,_) = np.shape(tensor_v)
    velo = np.zeros((N_c,N_g,2))
    
    for i in range(N_c):
        for j in range(N_g):
            for k in range(2):
                velo[i,j,k] = np.dot(tensor_v[i,j,k,:] ,membership[i,:])
    
    return velo

def dynamical_iteration(adata, n_states=None, n_states_seq=None, n_iter=10, return_aggr_obj=True, weight_connectivities=0.2, n_components=20, n_neighbors=100, thresh_ms_gene=0, thresh_entropy=0.1, use_spatial=False, spa_weight=0.5, spa_conn_key='spatial', monitor_mode=False):
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
        ent_diff_rel = np.abs(entropy-entropy_orig)/np.abs(entropy_orig)
        err_diff_abs = np.abs(entropy-entropy_orig)
        ent_diff_rel_series = pd.Series(ent_diff_rel)
        err_diff_abs_series = pd.Series(err_diff_abs)

        quantiles = [0, 0.25, 0.5, 0.75, 1]
        print("\nQuantiles for entropy relative difference with last iteration:")
        print(ent_diff_rel_series.quantile(quantiles))

        print("\nQuantiles for entropy absolute difference with last iteration:")
        print(err_diff_abs_series.quantile(quantiles))

        err_ent = np.quantile(ent_diff_rel,0.75)
        
        print("\nQuantiles for entropy absolute difference with last iteration:")
        
        if monitor_mode:
            user_input = input("Do you want to continue? (y/n): ").strip().lower()
            if user_input == 'n':
                    print("Exiting the loop.")
                    break
            elif user_input != 'y':
                    print("Invalid input, please enter 'y' to continue or 'n' to exit.")
        else:
            err_ent = np.quantile(ent_diff_rel,0.75)
            if err_ent < thresh_entropy:
                print("Entropy difference is below the threshold, exiting the loop.")
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
    