import os
import ot
import random
import numpy as np
import pandas as pd
import scanpy as sc
import matplotlib.pyplot as plt

from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.neighbors import NearestNeighbors

from alignment import st_pairwise_align

import torch
import torch.nn.functional as F
from torch_geometric.utils import negative_sampling

def seed_everything(seed=2024):
    random.seed(seed)    # Python random module
    np.random.seed(seed) # Numpy module
    os.environ['PYTHONHASHSEED'] = str(seed) # Env variable
    
    torch.manual_seed(seed)  # Torch
    torch.cuda.manual_seed(seed)  # CUDA
    torch.cuda.manual_seed_all(seed)  # For multi-GPU setups
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    print(f'Seeding all randomness with seed={seed}')
    
# Function to generate masked data by randomly setting a percentage of non-zero values to zero
def generate_masked_data(adata, perc, seed, verbose=True):
    seed_everything(seed)
    nonzero_indices = adata.X.nonzero()
    nonzero_count = len(nonzero_indices[0])
    total_mask = np.zeros(nonzero_count, dtype=bool)

    if perc > 0:
        num_to_mask = int(nonzero_count * perc)
        current_mask = np.zeros(nonzero_count, dtype=bool)
        current_mask[:num_to_mask] = True
        np.random.shuffle(current_mask)
        
        rows, cols = nonzero_indices
        unique_rows = np.unique(rows)
        
        # Apply the initial mask and print statistics
        masked_data_temp = adata.copy()
        mask_indices_temp = (rows[current_mask], cols[current_mask])
        masked_data_temp.X[mask_indices_temp] = 0
        zero_gene_cells = np.sum(np.sum(masked_data_temp.X, axis=1) == 0)
        zero_cell_genes = np.sum(np.sum(masked_data_temp.X, axis=0) == 0)
        
        if verbose:
            print("Initial Masking:")
            print(f"  - Initial masked count: {current_mask.sum()}")
            print(f"  - Cells with no expressed genes: {zero_gene_cells}")
            print(f"  - Genes with no expressed cells: {zero_cell_genes}")

        # Loop to ensure each cell retains at least one non-zero gene
        iteration = 0
        while True:
            unmasked_indices = []
            iteration += 1
            
            # Ensure each cell retains at least one non-zero gene
            for row in unique_rows:
                row_indices = np.where(rows == row)[0]
                if all(current_mask[row_indices]):  # If all genes in a cell are masked
                    # Randomly select one gene in this cell to unmask
                    unmask_index = np.random.choice(row_indices)
                    current_mask[unmask_index] = False
                    unmasked_indices.append(unmask_index)
            
            # Break the loop if no adjustments are needed
            if not unmasked_indices:
                break
            
            # Re-mask an equal number of genes in other locations to maintain correct masking ratio
            available_indices = np.where(~current_mask)[0]
            additional_mask_indices = np.random.choice(available_indices, len(unmasked_indices), replace=False)
            current_mask[additional_mask_indices] = True
            
            # Update masked data after each iteration for printing
            masked_data_temp = adata.copy()
            mask_indices_temp = (rows[current_mask], cols[current_mask])
            masked_data_temp.X[mask_indices_temp] = 0
            zero_gene_cells = np.sum(np.sum(masked_data_temp.X, axis=1) == 0)
            zero_cell_genes = np.sum(np.sum(masked_data_temp.X, axis=0) == 0)

            if verbose:
                print(f"Iteration {iteration}:")
                print(f"  - Masked count: {current_mask.sum()}")
                print(f"  - Cells with no expressed genes: {zero_gene_cells}")
                print(f"  - Genes with no expressed cells: {zero_cell_genes}")
        
        total_mask = current_mask
        mask_indices = (rows[total_mask], cols[total_mask])
        masked_data = adata.copy()
        masked_data.X[mask_indices] = 0
    else:
        masked_data = adata.copy()

    # Final density output
    if verbose:
        print("Final Masking:")
        print("Percentage:", perc, "Density:", len(masked_data.X.nonzero()[0]) / (masked_data.shape[0] * masked_data.shape[1]))

    return masked_data

def generate_gene_masked_data(adata, perc, seed):
    seed_everything(seed)
    num_genes = adata.shape[1]
    num_to_mask = int(num_genes * perc)  # Calculate the number of genes to mask
    
    # Randomly select genes to mask
    mask_indices = np.random.choice(num_genes, num_to_mask, replace=False)
    
    masked_data = adata.copy()
    # Drop the selected genes by removing these columns
    masked_data = masked_data[:, np.setdiff1d(np.arange(num_genes), mask_indices)]
    
    print("perc:", perc, "density:", len(masked_data.X.nonzero()[0]) / (masked_data.shape[0] * masked_data.shape[1]))
    return masked_data

def mclust_R(adata, num_cluster, modelNames='EEE', used_obsm='STAGATE', random_seed=2024):
    np.random.seed(random_seed)
    import rpy2.robjects as robjects
    robjects.r.library("mclust")

    import rpy2.robjects.numpy2ri
    rpy2.robjects.numpy2ri.activate()
    r_random_seed = robjects.r['set.seed']
    r_random_seed(random_seed)
    rmclust = robjects.r['Mclust']
    
    res = rmclust(rpy2.robjects.numpy2ri.numpy2rpy(adata.obsm[used_obsm]), num_cluster, modelNames)
    mclust_res = np.array(res[-2])

    adata.obs['mclust'] = mclust_res
    adata.obs['mclust'] = adata.obs['mclust'].astype('int')
    adata.obs['mclust'] = adata.obs['mclust'].astype('category')
    return adata

# Function for Leiden clustering with binary search for optimal resolution
def leiden_clustering(adata, num_clusters, use_rep='X_pca', random_seed=2024, tol=0.01, visualize=False):
    np.random.seed(random_seed)
    sc.pp.neighbors(adata, use_rep=use_rep)
    
    low_res, high_res = 0.1, 2.0  # Initial resolution range
    best_resolution, best_clustering = None, None
    best_cluster_count_diff = float('inf')

    while (high_res - low_res) > tol:
        mid_res = (low_res + high_res) / 2
        sc.tl.leiden(adata, resolution=mid_res, random_state=random_seed)
        cluster_count = adata.obs['leiden'].nunique()
        cluster_count_diff = abs(cluster_count - num_clusters)

        # Visualization if enabled
        if visualize:
            plt.rcParams["figure.figsize"] = (5, 5)
            sc.pl.spatial(adata, color=["leiden"], title=[f'Leiden Clustering at resolution {mid_res:.4f}'])

        if cluster_count == num_clusters:
            best_resolution = mid_res
            best_clustering = adata.obs['leiden'].copy()
            break

        if cluster_count_diff < best_cluster_count_diff:
            best_cluster_count_diff = cluster_count_diff
            best_resolution = mid_res
            best_clustering = adata.obs['leiden'].copy()

        # Adjust resolution range based on cluster count
        if cluster_count > num_clusters:
            high_res = mid_res
        else:
            low_res = mid_res

    adata.obs['leiden'] = best_clustering.astype('category')

    # Final visualization if enabled
    if visualize:
        sc.pl.spatial(adata, color=["leiden"], title=[f'Final Leiden Clustering at resolution {best_resolution:.4f}'])

    return adata

def refine_label(adata, radius=50, key='label'):
    n_neigh = radius
    new_type = []
    old_type = adata.obs[key].values
    
    #calculate distance
    position = adata.obsm['spatial']
    distance = ot.dist(position, position, metric='euclidean')
        
    n_cell = distance.shape[0]
    
    for i in range(n_cell):
        vec  = distance[i, :]
        index = vec.argsort()
        neigh_type = []
        for j in range(1, n_neigh+1):
            neigh_type.append(old_type[index[j]])
        max_type = max(neigh_type, key=neigh_type.count)
        new_type.append(max_type)
        
    new_type = [str(i) for i in list(new_type)]    
    #adata.obs['label_refined'] = np.array(new_type)
    
    return new_type

import copy
from scipy.sparse import spmatrix
from sklearn.model_selection import train_test_split

def split_and_preserve_data(adata, val_test_split_ratio=0.1):
    """
    Split adata.X into train, val, and test sets, preserving val and test values.
    """
        
    # Make a deep copy of the original dense matrix
    original_x = copy.deepcopy(adata.X.todense())
    
    # Get all non-zero element indices
    non_zero_indices = np.transpose(np.nonzero(adata.X.todense()))
    
    # Calculate the number of elements to use for val and test sets (10% each)
    val_test_count = int(val_test_split_ratio * len(non_zero_indices))
    
    # Randomly select indices for val and test
    selected_indices = np.random.permutation(len(non_zero_indices))
    val_indices = selected_indices[:val_test_count]
    test_indices = selected_indices[val_test_count:2 * val_test_count]
    
    # Create masks for val and test
    val_mask = np.full(adata.X.shape, False)
    test_mask = np.full(adata.X.shape, False)
    
    val_mask[non_zero_indices[val_indices, 0], non_zero_indices[val_indices, 1]] = True
    test_mask[non_zero_indices[test_indices, 0], non_zero_indices[test_indices, 1]] = True
    
    # Set the values in validation and test positions to zero in adata.X
    adata.X[val_mask] = 0
    adata.X[test_mask] = 0
    
    return adata, val_mask, test_mask, original_x

def print_density(adata, label=""):
    """
    Calculate and print the density of non-zero entries in adata.X.
    Density is defined as the ratio of non-zero entries to total entries.
    """
    # Convert sparse matrix to dense format if necessary
    if isinstance(adata.X, (np.matrix, np.ndarray)):
        data = adata.X
    else:
        data = adata.X.toarray()
    
    non_zero_count = np.count_nonzero(data)
    total_count = data.size  # Total number of elements
    density = non_zero_count / total_count
    print(f"Density of {label}: {density:.4f} ({non_zero_count} / {total_count})")
    return density

from torch_geometric.data import Data

def process_data(adata):
    
    spatial_graph_radius_cutoff = 150
    gene_graph_knn_cutoff = 5
    gene_graph_pca_num = 50
    
    # adata.obs['Ground Truth'] = list(ground_truth_layer)
    spatial_graph_coor = pd.DataFrame(adata.obsm['spatial'])
    spatial_graph_coor.index = adata.obs.index
    spatial_graph_coor.columns = ['imagerow', 'imagecol']
    id_cell_trans = dict(zip(range(spatial_graph_coor.shape[0]), np.array(spatial_graph_coor.index)))
    cell_to_index = {id_cell_trans[idx]: idx for idx in id_cell_trans}
    nbrs = NearestNeighbors(radius=spatial_graph_radius_cutoff).fit(spatial_graph_coor)
    distances, indices = nbrs.radius_neighbors(spatial_graph_coor, return_distance=True)

    spatial_graph_KNN_list = [pd.DataFrame(zip([it]*indices[it].shape[0], indices[it], distances[it])) for it in range(indices.shape[0])]
    spatial_graph_KNN_df = pd.concat(spatial_graph_KNN_list)
    spatial_graph_KNN_df.columns = ['Cell1', 'Cell2', 'Distance']
    Spatial_Net = spatial_graph_KNN_df.copy()
    Spatial_Net = Spatial_Net[Spatial_Net['Distance']>0]
    Spatial_Net['Cell1'] = Spatial_Net['Cell1'].map(id_cell_trans)
    Spatial_Net['Cell2'] = Spatial_Net['Cell2'].map(id_cell_trans)
    print(f'The spatial graph contains {Spatial_Net.shape[0]} edges, {adata.n_obs} cells.')
    print(f'{Spatial_Net.shape[0]/adata.n_obs:.4f} neighbors per cell on average.')
    adata.uns['Spatial_Net'] = Spatial_Net

    spatial_graph_edge_index = []
    for idx, row in Spatial_Net.iterrows():
        cell1 = cell_to_index[row['Cell1']]
        cell2 = cell_to_index[row['Cell2']]
        spatial_graph_edge_index.append([cell1, cell2])
        spatial_graph_edge_index.append([cell2, cell1])
        
    spatial_graph_edge_index = torch.tensor(spatial_graph_edge_index, dtype=torch.long).t().contiguous()
    
    pca = PCA(n_components=gene_graph_pca_num)
    if hasattr(adata.X, 'todense'):
        pca_data = pca.fit_transform(np.array(adata.X.todense().astype(np.float32)))
    else:
        pca_data = pca.fit_transform(adata.X.astype(np.float32))
    nbrs_gene = NearestNeighbors(n_neighbors=gene_graph_knn_cutoff).fit(pca_data)
    distances, indices = nbrs_gene.kneighbors(pca_data, return_distance=True)
    
    gene_graph_KNN_list = [pd.DataFrame(zip([it]*indices[it].shape[0], indices[it], distances[it])) for it in range(indices.shape[0])]
    gene_graph_KNN_df = pd.concat(gene_graph_KNN_list)
    gene_graph_KNN_df.columns = ['Cell1', 'Cell2', 'Distance']
    Gene_Net = gene_graph_KNN_df.copy()
    Gene_Net['Cell1'] = Gene_Net['Cell1'].map(id_cell_trans)
    Gene_Net['Cell2'] = Gene_Net['Cell2'].map(id_cell_trans)
    print(f'The gene graph contains {Gene_Net.shape[0]} edges, {adata.n_obs} cells.')
    print(f'{Gene_Net.shape[0]/adata.n_obs:.4f} neighbors per cell on average.')
    gene_graph_edge_index = []
    for idx, row in Gene_Net.iterrows():
        cell1 = cell_to_index[row['Cell1']]
        cell2 = cell_to_index[row['Cell2']]
        gene_graph_edge_index.append([cell1, cell2])
        gene_graph_edge_index.append([cell2, cell1])  # Add reversed edge since the graph is undirected
    gene_graph_edge_index = torch.tensor(gene_graph_edge_index, dtype=torch.long).t().contiguous()
    
    # Create a PyTorch tensor for node features
    if hasattr(adata.X, 'todense'):
        x = torch.from_numpy(adata.X.todense().astype(np.float32))
    else:
        x = torch.from_numpy(adata.X.astype(np.float32))
    original_x = x.clone()  # Save the original features
    
    edge_index = torch.cat([spatial_graph_edge_index, gene_graph_edge_index], dim=1)
    edge_type = torch.cat([torch.zeros(spatial_graph_edge_index.size(1), dtype=torch.long),
                            torch.ones(gene_graph_edge_index.size(1), dtype=torch.long)], dim=0)
    data = Data(x=x, edge_index=edge_index, edge_type=edge_type)
    
    return data
   
    
def save_results_to_csv(results_dict, dataset_name, results_dir):
    """
    Save results for a dataset to a CSV file.
    Each row represents a different percentage, with multiple results for each metric.
    """
    file_path = os.path.join(results_dir, f'dataset_{dataset_name}_full.csv')
    
    # Convert dictionary to DataFrame
    df = pd.DataFrame.from_dict(results_dict, orient='index')
    df.to_csv(file_path)
    print(f"Results saved to {file_path}")
    
def evaluate(predicted_x, original_x, test_mask):
    """
    Evaluate the performance using L1 distance, Cosine Similarity, and RMSE.
    
    Parameters:
    - predicted_x: NumPy array containing the predicted values.
    - original_x: NumPy array containing the original ground truth values.
    - test_mask: NumPy array (boolean) indicating the test positions.
    
    Returns:
    - l1_distance: L1 distance (mean absolute error).
    - cosine_sim: Cosine similarity.
    - rmse: Root Mean Square Error (RMSE).
    """
    # Ensure inputs are NumPy arrays
    if not isinstance(predicted_x, np.ndarray):
        predicted_x = np.array(predicted_x)
    if not isinstance(original_x, np.ndarray):
        original_x = np.array(original_x)
    if not isinstance(test_mask, np.ndarray):
        test_mask = np.array(test_mask)
    
    # Extract the values at the positions indicated by test_mask
    predicted_values = predicted_x[test_mask]
    original_values = original_x[test_mask]

    # Convert to PyTorch tensors for calculations
    predicted_values = torch.tensor(predicted_values, dtype=torch.float32)
    original_values = torch.tensor(original_values, dtype=torch.float32)
    
    # Ensure the shapes match
    assert predicted_values.shape == original_values.shape, "Shape mismatch between predicted and original values"
    
    # Calculate L1 distance (Mean Absolute Error)
    l1_distance = F.l1_loss(predicted_values, original_values).item()
    
    # Calculate Cosine Similarity
    cosine_sim = F.cosine_similarity(predicted_values, original_values, dim=0).mean().item()
    
    # Calculate Root Mean Square Error (RMSE)
    rmse = torch.sqrt(F.mse_loss(predicted_values, original_values)).item()
    
    return l1_distance, cosine_sim, rmse