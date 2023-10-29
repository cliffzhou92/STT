#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr  2 01:26:55 2021

@author: cliffzhou
"""

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib import cm

import scipy
from numpy.linalg import inv
from sklearn.mixture import GaussianMixture

import networks as nw
import pyemma.msm as msm

import anndata
import scvelo as scv
import scanpy as sc

import cellrank as cr
from cellrank.tl.estimators import GPCCA
from cellrank.tl.kernels import ConnectivityKernel

import plotly.graph_objects as go
from collections import defaultdict

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
    
def construct_landscape(sc_object,thresh_cal_cov = 0.3, scale_axis = 1.0, scale_land = 1.1, N_grid = 100, coord_key = 'X_umap'):
    
    
    mu_hat = sc_object.uns['da_out']['mu_hat']
    rho = sc_object.obsm['rho']
    projection = sc_object.obsm[coord_key][:,0:2]
    
    
    labels = np.argmax(rho,axis = 1)
    K = max(labels)+1
    
    centers = []
    for i in range(K):
        index = labels==i
        p = np.mean(projection[index], axis=0)
        centers.append(p)
    centers = np.array(centers)
        
    trans_coord = np.matmul(rho,centers)
    
    centers = []
    for i in range(K):
        index = labels==i
        p = np.mean(trans_coord[index], axis=0)
        centers.append(p)
    centers = np.array(centers)
    
    mu = np.zeros((K,2))
    precision = np.zeros((K,2,2))    
    
    for i in range(K):
        member_id = np.array(labels == i) 
        stable_id = rho[:,i]>thresh_cal_cov
        select_id = np.logical_or(member_id,stable_id)
        coord_select = trans_coord[select_id,]
        mu[i,:] = np.mean(coord_select,axis = 0)
        precision[i,:,:] = inv(np.cov(coord_select.T))
    

    gmm = GaussianMixture(n_components = K, weights_init=mu_hat, means_init = mu,precisions_init=precision, max_iter = 5,reg_covar = 1e-03)
    gmm.fit(trans_coord)
    land_cell = -gmm.score_samples(trans_coord)

    coord_min,coord_max = trans_coord.min(axis = 0),trans_coord.max(axis = 0)
    x_grid = scale_axis* np.linspace(coord_min[0],coord_max[0], N_grid)
    y_grid = scale_axis* np.linspace(coord_min[1],coord_max[1], N_grid)
    xv,yv = np.meshgrid(x_grid,y_grid)

    pos = np.empty(xv.shape + (2,))
    pos[:, :, 0] = xv; pos[:, :, 1] = yv
    land_value = -gmm.score_samples(pos.reshape(-1,2)).reshape(N_grid,N_grid)
    land_max_thresh = scale_land*np.max(land_cell)
    land_value[land_value>land_max_thresh] = np.nan

    
    sc_object.uns['land_out'] = {}
    sc_object.uns['land_out']['land_value'] = land_value
    sc_object.uns['land_out']['grid_x'] = xv
    sc_object.uns['land_out']['grid_y'] = yv
    sc_object.uns['land_out']['trans_coord'] = trans_coord
    sc_object.uns['land_out']['cluster_centers'] = centers
    sc_object.obs['land_cell'] = land_cell
    

    
def plot_landscape(sc_object,show_colorbar = False, dim = 2, size_point = 3, alpha_land = 0.5, alpha_point = 0.5,  color_palette_name = 'Set1', contour_levels = 15, elev=10, azim = 4):
    land_value = sc_object.uns['land_out']['land_value']
    xv = sc_object.uns['land_out']['grid_x']
    yv = sc_object.uns['land_out']['grid_y']
    trans_coord = sc_object.uns['land_out']['trans_coord'] 
    land_cell = sc_object.obs['land_cell']
    
    K = sc_object.obsm['rho'].shape[1]
    labels =sc_object.obs['attractor'].astype(int)
    
    color_palette = sns.color_palette(color_palette_name, K)
    cluster_colors = [color_palette[x] for x in labels]
    
    if dim == 2:
        plt.contourf(xv, yv, land_value, levels=contour_levels, cmap="Greys_r",zorder=-100, alpha = alpha_land)
        plt.scatter(*trans_coord.T, s=size_point, linewidth=0, c=cluster_colors, alpha=alpha_point)

    else:
        ax = plt.axes(projection='3d')
        ax.scatter(*trans_coord.T,land_cell,s=size_point, linewidth=0, c=cluster_colors, alpha=alpha_point)
        ax.plot_surface(xv, yv,land_value,rstride=1, cstride=1, linewidth=0, antialiased=True,cmap="Greys_r", alpha = alpha_land, vmin = 0, vmax=np.nanmax(land_value), shade = True)
        ax.grid(False)
        ax.axis('off')
        ax.view_init(elev=elev, azim=azim)
        
    if show_colorbar:
        plt.colorbar()
        
        


def infer_lineage(sc_object,si=0,sf=1,method = 'MPFT',flux_fraction = 0.9, size_state = 0.1, size_point = 3, alpha_land = 0.5, alpha_point = 0.5, size_text=20, show_colorbar = False, color_palette_name = 'Set1', contour_levels = 15):
    
    
    K = sc_object.obsm['rho'].shape[1]
    centers = sc_object.uns['land_out']['cluster_centers']

    

    P_hat = sc_object.uns['da_out']['P_hat']
    M = msm.markov_model(P_hat)
    mu_hat = M.pi
    
    if method == 'MPFT':
        Flux_cg = np.diag(mu_hat.reshape(-1)).dot(P_hat)
        max_flux_tree = scipy.sparse.csgraph.minimum_spanning_tree(-Flux_cg)
        max_flux_tree = -max_flux_tree.toarray()
        #for i in range(K):
        #    for j in range(i+1,K):   
        #        max_flux_tree[i,j]= max(max_flux_tree[i,j],max_flux_tree[j,i])
        #        max_flux_tree[j,i] = max_flux_tree[i,j]
         
        nw.plot_network(max_flux_tree, pos=centers, state_scale=size_state, state_sizes=mu_hat, arrow_scale=2.0,arrow_labels= None, arrow_curvature = 0.2, ax=plt.gca(),max_width=1000, max_height=1000)
        plot_landscape(sc_object, show_colorbar = show_colorbar, size_point = size_point, alpha_land = alpha_land, alpha_point = alpha_point,  color_palette_name = color_palette_name)
        plt.axis('off')   

        
    if method == 'MPPT':
        
        #state_reorder = np.array(range(K))
        #state_reorder[0] = si
        #state_reorder[-1] = sf
        #state_reorder[sf+1:-1]=state_reorder[sf+1:-1]+1
        #state_reorder[1:si]=state_reorder[1:si]-1
        if isinstance(si,int):
            si = list(map(int, str(si)))
        
        if isinstance(sf,int):
            sf = list(map(int, str(sf)))
        
        
        tpt = msm.tpt(M, si, sf)
        Fsub = tpt.major_flux(fraction=flux_fraction)
        Fsubpercent = 100.0 * Fsub / tpt.total_flux
        
       
        plot_landscape(sc_object, show_colorbar = show_colorbar, size_point = size_point, alpha_land = alpha_land, alpha_point = alpha_point,  color_palette_name = color_palette_name, contour_levels = contour_levels)
        nw.plot_network(Fsubpercent, state_scale=size_state*mu_hat,pos=centers, arrow_label_format="%3.1f",arrow_label_size = size_text,ax=plt.gca(), max_width=1000, max_height=1000)
        plt.axis('off')   

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

def dynamical_iteration(adata, n_states = None, n_states_seq = None, n_iter = 10, return_aggr_obj=False, weight_connectivities=0.2, n_components=20, n_neighbors = 100,thresh_ms_gene = 0, thresh_entropy = 0.1, use_spatial = False, spa_weight = 0.5, spa_conn_key = 'spatial', stop_cr = 'abs'):
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
    

def plot_top_genes(adata, top_genes = 6, ncols = 2, figsize = (8,8), color_map = 'tab10',hspace = 0.5,wspace = 0.5):
    K = adata.obsm['rho'].shape[1]
    cmp = sns.color_palette(color_map, K)
    U = adata.layers['Mu']
    S = adata.layers['Ms']
    
    gene_sort = adata.var['r2'].sort_values(ascending=False).index.tolist()

    # calculate number of rows
    nrows = top_genes // ncols + (top_genes % ncols > 0)

    plt.figure(figsize=figsize)
    plt.subplots_adjust(hspace=hspace,wspace = wspace)
    
    for gene_id in range(top_genes):

        gene_name =  gene_sort[gene_id]
        ind_g = adata.var_names.tolist().index(gene_name)

        par = adata.uns['par'][ind_g,:]
        alpha = par[0:K]
        beta = par[K]


        ax = plt.subplot(nrows, ncols, gene_id + 1)
        scv.pl.scatter(adata, x = S[:,ind_g],y = U[:,ind_g],color = 'attractor',show=False,alpha = 0.5,size = 50,ax=ax)
        ax.axline((0, 0), slope=1/beta,color = 'k')
        ax.set_title(gene_name)
        for i in range(K):
            ax.axline((0, alpha[i]/beta), slope=0, color = cmp[i])

def plot_genes_list(adata, genelist, ncols = 2, figsize = (8,8), color_map = 'tab10',hspace = 0.5,wspace = 0.5):
    K = adata.obsm['rho'].shape[1]
    cmp = sns.color_palette(color_map, K)
    U = adata.layers['Mu']
    S = adata.layers['Ms']
    
    # calculate number of rows
    nrows = len(genelist) // ncols + (len(genelist) % ncols > 0)

    plt.figure(figsize=figsize)
    plt.subplots_adjust(hspace=hspace,wspace = wspace)
    
    for gene_id in range(len(genelist)):

        gene_name =  genelist[gene_id]
        ind_g = adata.var_names.tolist().index(gene_name)

        par = adata.uns['par'][ind_g,:]
        alpha = par[0:K]
        beta = par[K]


        ax = plt.subplot(nrows, ncols, gene_id + 1)
        scv.pl.scatter(adata, x = S[:,ind_g],y = U[:,ind_g],color = 'attractor',show=False,alpha = 0.5,size = 20,ax=ax)
        ax.axline((0, 0), slope=1/beta,color = 'k')
        ax.set_title(gene_name)
        for i in range(K):
            ax.axline((0, alpha[i]/beta), slope=0, color = cmp[i])
    
def plot_tensor_single(adata, adata_aggr = None, state = 'joint', attractor = None, basis = 'umap', color ='attractor', color_map = None, size = 20, alpha = 0.5, ax = None, show = None, filter_cells = False, member_thresh = 0.05):
    
    if attractor == None:
        velo =  adata.obsm['tensor_v_aver'].copy()
        title = 'All attractors'
    else:
        velo = adata.obsm['tensor_v'][:,:,:,attractor].copy()
        color = adata.obsm['rho'][:,attractor] 
        title = 'Attractor '+str(attractor)
        color_map = 'coolwarm'
        if filter_cells:
            cell_id_filtered = adata.obsm['rho'][:,attractor] < member_thresh
            velo[cell_id_filtered,:,:] = 0
    
    adata_aggr = adata_aggr[:,adata_aggr.uns['gene_subset']]
    gene_select = [x in adata.uns['gene_subset'] for x in adata.var_names]
    adata = adata[:,gene_select]
    print(adata)
        
    if state == 'spliced':
        adata.layers['vs'] = velo[:,gene_select,1]
        scv.tl.velocity_graph(adata, vkey = 'vs', xkey = 'Ms', gene_subset = adata.uns['gene_subset'],n_jobs = -1)
        scv.pl.velocity_embedding_stream(adata, vkey = 'vs', basis=basis, color=color, title = title+','+'Spliced',color_map = color_map, size = size, alpha = alpha, ax = ax, show = show)
    if state == 'unspliced':
        adata.layers['vu'] = velo[:,gene_select,0]
        scv.tl.velocity_graph(adata, vkey = 'vu', xkey = 'Mu', gene_subset =adata.uns['gene_subset'],n_jobs = -1)
        scv.pl.velocity_embedding_stream(adata, vkey = 'vu',basis=basis, color=color, title = title+','+'Unspliced',color_map = color_map, size = size, alpha = alpha, ax = ax, show = show)
    if state == 'joint':
        print("check that the input includes aggregated object")
        adata_aggr.layers['vj'] = np.concatenate((velo[:,gene_select,0],velo[:,gene_select,1]),axis = 1)
        scv.tl.velocity_graph(adata_aggr, vkey = 'vj', xkey = 'Ms',n_jobs = -1)
        scv.pl.velocity_embedding_stream(adata_aggr, vkey = 'vj',basis=basis, color=color, title = title+','+'Joint',color_map = color_map, size = size, alpha = alpha, ax = ax, show = show)
        
        
def plot_tensor(adata, adata_aggr, list_state =['joint','spliced','unspliced'], list_attractor ='all', basis = 'umap',figsize = (8,8),hspace = 0.2,wspace = 0.2, color_map = None,size = 20,alpha = 0.5, filter_cells = False, member_thresh = 0.05):
    
    if list_attractor == 'all':
        list_attractor =[None]+list(range(len(adata.obs['attractor'].unique())))
    
    nrows = len(list_state)
    ncols = len(list_attractor)
    plt.figure(figsize=figsize)
    plt.subplots_adjust(hspace=hspace,wspace = wspace)
    fig_id = 1
    
    
    for state in list_state:
        for attractor in list_attractor:
            if state !='joint':
                basis_plot = basis+'_aggr'
            else:
                basis_plot = basis    
            ax = plt.subplot(nrows, ncols, fig_id)
            fig_id+=1
            plot_tensor_single(adata, adata_aggr, attractor = attractor, state = state, basis = basis_plot,ax = ax,show = False, member_thresh = member_thresh, filter_cells = filter_cells, size = size, alpha = alpha) 
            
def plot_para_hist(adata, bins = 20, log = True,figsize = (8,8)):
    gene_select = [x in adata.uns['gene_subset'] for x in adata.var_names]
    par = adata.uns['par']
    K = par.shape[1]
    fig, axs = plt.subplots(1, K, sharex=True, sharey=True, tight_layout=False, squeeze = True, figsize = figsize)

    if log:
        par = np.log10(par)
    
    for i in range(K):
        if i<K-1:
            title_name = 'alpha'+str(i)
            color = None
        else:
            title_name = 'beta'
            color = 'g'
        axs[i].hist(par[:,i],density = True, bins = bins, color = color)
        axs[i].set_xlabel('log(parameter)')
        axs[i].set_ylabel('density')
        axs[i].set_title(title_name)


def plot_sankey(vector1, vector2):
    # Ensure both vectors have the same length.
    assert len(vector1) == len(vector2)

    label_dict = defaultdict(lambda: len(label_dict))
    
    # Generate a palette of colors.
    palette = sns.color_palette('husl', n_colors=len(set(vector1 + vector2))).as_hex()
    color_list = []

    for label in vector1 + vector2:
        label_id = label_dict[label]
        if len(color_list) <= label_id:
            color_list.append(palette[label_id % len(palette)])

    source = [label_dict[label] for label in vector1]
    target = [label_dict[label] for label in vector2]
    value = [1] * len(vector1)  # Assume each pair has a value of 1.
    
    # Color the links according to their target node.
    link_color = [color_list[target[i]] for i in range(len(target))]

    # Create the Sankey diagram.
    fig = go.Figure(data=[go.Sankey(
        node=dict(
            pad=15,
            thickness=20,
            line=dict(color="black", width=0.5),
            label=list(label_dict.keys()),
            color=color_list
        ),
        link=dict(
            source=source,
            target=target,
            value=value,
            color=link_color  # Color the links.
        )
    )])

    fig.update_layout(title_text="Sankey Diagram", font_size=20)
    fig.show()        
                
def compute_tensor_similarity(adata, adata_aggr, pathway1, pathway2, state = 'spliced', attractor = None):
    if attractor == None:
        velo =  adata.obsm['tensor_v_aver'].copy()
    else:
        velo = adata.obsm['tensor_v'][:,:,:,attractor].copy()
    
    if state == 'spliced':
        vkey = 'vs'
        xkey = 'Ms'
    if state == 'unspliced':
        vkey = 'vu'
        xkey = 'Mu'
    if state == 'joint':
        print("check that the input includes aggregated object") # some problem needs fixed
        adata_aggr.layers['vj'] = np.concatenate((velo[:,:,0],velo[:,:,1]),axis = 1)
        vkey = 'vj'
        xkey = 'Ms'

    scv.tl.velocity_graph(adata, vkey = vkey, xkey = xkey, gene_subset = pathway1,n_jobs = -1)
    tpm1 = adata.uns[vkey+'_graph'].toarray()
    scv.tl.velocity_graph(adata, vkey = vkey, xkey = vkey, gene_subset = pathway2,n_jobs = -1)
    tpm2 = adata.uns[vkey+'_graph'].toarray()
    return np.corrcoef(m1.reshape(-1),m2.reshape(-1))[0,1]