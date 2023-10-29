import numpy as np
import scipy
from numpy.linalg import inv
from sklearn.mixture import GaussianMixture
import pyemma.msm as msm

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