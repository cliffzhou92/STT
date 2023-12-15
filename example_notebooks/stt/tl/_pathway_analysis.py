import numpy as np
import umap
import scvelo as scv
from sklearn.decomposition import PCA
import gseapy as gp
from scipy.cluster.hierarchy import linkage, fcluster
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

def optimal_kmeans(data, k_range):
    # Function to determine the optimal number of clusters for KMeans
    best_score = -1
    best_k = 2

    for k in k_range:
        kmeans = KMeans(n_clusters=k, random_state=42)
        cluster_labels = kmeans.fit_predict(data)
        silhouette_avg = silhouette_score(data, cluster_labels)

        if silhouette_avg > best_score:
            best_score = silhouette_avg
            best_k = k

    # Fitting KMeans with the optimal number of clusters
    kmeans = KMeans(n_clusters=best_k, random_state=42)
    final_labels = kmeans.fit_predict(data)
    return final_labels

def compute_pathway(adata,adata_aggr,db_name):
    pathway = gp.parser.download_library(name = db_name)
    tpm_dict = {}
    pathway_select = {}
    temp = []
    gene_select = [x in adata.uns['gene_subset'] for x in adata.var_names]
    #velo =  adata.obsm['tensor_v_aver'].copy()
    #adata_aggr.layers['vj'] = np.concatenate((velo[:,gene_select,0],velo[:,gene_select,1]),axis = 1)


    for key in pathway.keys():
        gene_list = [x.capitalize() for x in pathway[key]] 
        gene_select = [x for x in gene_list if x in adata_aggr.var_names]
        if len(gene_select)>2 and gene_select not in temp:
                scv.tl.velocity_graph(adata_aggr, vkey = 'vj', xkey = 'Ms', n_jobs = -1)
                tpm_dict[key] = adata_aggr.uns['vj_graph'].toarray().reshape(-1)
                pathway_select[key] = gene_select
                temp.append(gene_select)
    
    adata.uns['pathway_select'] = pathway_select

    # compute correlation
    arr = np.stack(list(tpm_dict.values()))
    cor = np.corrcoef(arr)
    # dimensionality reduction
    pca = PCA(n_components=10)
    pca_embedding = pca.fit_transform(cor)
    # Perform UMAP on the PCA embedding
    umap_reducer = umap.UMAP(random_state=42)
    umap_embedding = umap_reducer.fit_transform(pca_embedding)
    # Perform hierarchical clustering
    adata.uns['pathway_embedding'] = umap_embedding
    c_labels = optimal_kmeans(umap_embedding,range(3,8))
    adata.uns['pathway_labels'] = c_labels

    