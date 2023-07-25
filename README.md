# STT

## Introduction
We present a spatial transition tensor (STT) method that utilizes mRNA splicing and spatial transcriptomes through a multiscale dynamical model to characterize multi-stability in space. By learning a four-dimensional transition tensor and spatial-constrained random walk, STT reconstructs cell-state specific dynamics and spatial state-transitions via both short-time local tensor streamlines between cells and long-time transition paths among attractors. Overall, STT provides a consistent multiscale description of single-cell transcriptome data across multiple spatiotemporal scales. 


<img src="https://github.com/cliffzhou92/STT/blob/main/img/Picture1.png" width="800">

## Get Started
Install the dependencies by ``pip install -r requirements.txt``, and change directory to the ``./example_notebooks``

## Basic Usage
```python
import sctt as st
adata.obs['attractor'] =  # initialize the attractor, can use leiden or original annotation
adata_aggr = st.dynamical_iteration(adata,n_states =K, n_iter = 20, weight_connectivities = 0.5, n_neighbors = 100, n_components = 20,thresh_ms_gene = 0,thresh_entropy = 0.1)
# n_states: number of attractors
# n_iter: maximum of iteration
# thresh_entropy: the threshold of maximum entropy difference between iterations to halt iteration, default is 0.1
# weight_connectivities: the weight of diffusion kernel as opposed to velocity kernel, default is 0.5
# n_neighbors: number of neghbors used to constrcut cellular random walk, default is 100
# n_component: number of eigen components to use in GPCCA decomposion, default is 20
# thresh_ms_gene: the threshold of minimum multi-stability score of genes to include when constructing random walk, default is 0
st.infer_lineage(adata,si=4,sf=3) # infer and plot the transition path
st.plot_tensor(adata, adata_aggr,  basis = 'trans_coord',list_attractor = [0,1,2,3]) # plot the transition tensor components
st.plot_top_genes(adata, top_genes = 10) # plot the U-S diagram of top genes with highest multi-stability score

```
The full tutorials are provided as example notebooks below.
## Example Notebooks
**System** | **Data Source** | **Notebook File**
------------| -------------- | ------------
Toggle-switch | [Simulation Data](https://github.com/cliffzhou92/STT/blob/main/data/toggle_switch/generating_toggle_data.ipynb) in this study | [notebook](https://github.com/cliffzhou92/STT/blob/main/example_notebooks/example_toggle.ipynb)
EMT circuit | [Simulation Data](https://github.com/cliffzhou92/STT/tree/main/data/emt_sim/Generating_Dataset.ipynb) in this study |[notebook](https://github.com/cliffzhou92/scTT/blob/main/example_notebooks/example_emt_circuit.ipynb)
EMT of A549 cell lines |[Cook et al.](https://www.nature.com/articles/s41467-020-16066-2)|[notebook](https://github.com/cliffzhou92/STT/blob/main/example_notebooks/example-emt.ipynb)
Erythroid lineage in mouse gastrulation |[Pijuan-Sala et al.](https://www.nature.com/articles/s41586-019-0933-9) and [scVelo](https://scvelo.readthedocs.io/scvelo.datasets.gastrulation_erythroid/)|[notebook](https://github.com/cliffzhou92/STT/blob/main/example_notebooks/example-mouse_eryth.ipynb)
Adult human bone marrow | [Setty et al.](https://www.nature.com/articles/s41587-019-0068-4) and [scVelo](https://scvelo.readthedocs.io/scvelo.datasets.bonemarrow/)| [notebook](https://github.com/cliffzhou92/STT/blob/main/example_notebooks/example-bone_marrow.ipynb)
Developing Mouse Brain | [Manno et al.](https://www.nature.com/articles/s41586-021-03775-x) and [SIRV](https://zenodo.org/record/6798659)| [notebook](https://github.com/cliffzhou92/STT/blob/main/example_notebooks/example-mouse_brain-spatial.ipynb)
Developing Chicken Heart | [Mantri et al.](https://www.nature.com/articles/s41467-021-21892-z) and [SIRV](https://zenodo.org/record/6798659)| [notebook](https://github.com/cliffzhou92/STT/blob/main/example_notebooks/example-chicken_heart.ipynb)

