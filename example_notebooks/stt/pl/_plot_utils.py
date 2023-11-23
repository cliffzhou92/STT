import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib import cm

import plotly.graph_objects as go
from collections import defaultdict
import scvelo as scv

def plot_top_genes(adata, top_genes = 6, ncols = 2, figsize = (8,8), color_map = 'tab10',hspace = 0.5,wspace = 0.5):
    K = adata.obsm['rho'].shape[1]
    cmp = sns.color_palette(color_map, K)
    U = adata.layers['Mu']
    S = adata.layers['Ms']
    
    gene_sort = adata.var['r2_test'].sort_values(ascending=False).index.tolist()

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
    return np.corrcoef(tpm1.reshape(-1),tpm2.reshape(-1))[0,1]