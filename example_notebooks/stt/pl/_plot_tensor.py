import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib import cm
import networks as nw
import plotly.graph_objects as go
from collections import defaultdict
import scvelo as scv

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
