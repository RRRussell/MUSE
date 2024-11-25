import umap
import numpy as np
import pandas as pd
import scanpy as sc

import seaborn as sns
import matplotlib.pyplot as plt

import torch
import torch.optim as optim
import torch.nn.functional as F

from sklearn import metrics
from sklearn.metrics import confusion_matrix
from torch_geometric.utils import negative_sampling

def evaluate_transfer_matrix(adata_0, adata_1, T, savefig_path=None, dataset="10X_DLPFC"):
    if dataset == "10X_DLPFC":
        labels_0 = adata_0.obs['ground_truth'].astype('category')
        labels_1 = adata_1.obs['ground_truth'].astype('category')
    elif dataset == "Merfish_MouseBrainAging":
        labels_0 = adata_0.obs['tissue'].astype('category')
        labels_1 = adata_1.obs['tissue'].astype('category')

    # Combine categories to visualize all mappings, even between different layers
    all_categories = labels_0.cat.categories.union(labels_1.cat.categories)
    labels_0 = labels_0.cat.set_categories(all_categories, ordered=True)
    labels_1 = labels_1.cat.set_categories(all_categories, ordered=True)

    # Find the best match for each node based on the transfer matrix
    aligned_indices = np.argmax(T, axis=1)
    aligned_labels = labels_1.iloc[aligned_indices].values
    original_labels = labels_0.values

    # Create a DataFrame to track original and aligned labels
    alignment_results = pd.DataFrame({
        'Original': pd.Categorical(original_labels, categories=all_categories),
        'Aligned': pd.Categorical(aligned_labels, categories=all_categories)
    })

    # Create a confusion matrix to compare original and aligned labels, including all categories
    conf_matrix = confusion_matrix(alignment_results['Original'], alignment_results['Aligned'], labels=all_categories)

    # Calculate the percentage alignment for each layer
    percentages = conf_matrix / conf_matrix.sum(axis=1, keepdims=True) * 100

    # Visualization of the percentage-based confusion matrix with colorbar from 0 to 100
    plt.figure(figsize=(4, 3))
    sns.heatmap(percentages, annot=True, fmt=".1f", cmap="Blues", 
                vmin=0, vmax=100,  
                xticklabels=all_categories, yticklabels=all_categories)  # Ensure x and y tick labels match respective axis
    plt.xlabel('Aligned Label')
    plt.ylabel('Original Label')
    plt.title('Percentage Confusion Matrix of Spatial Domain Alignment')
    if savefig_path:
        plt.savefig(savefig_path, bbox_inches='tight')
    plt.show()

    # Compute alignment accuracy as the sum of diagonal percentages (correct mappings)
    alignment_score = np.trace(percentages) / np.sum(percentages)
    print(f"Alignment Score (Accuracy): {alignment_score:.2f}")

    return alignment_score

# def evaluate_transfer_matrix(adata_0, adata_1, T, savefig_path=None, dataset="10X_DLPFC"):
#     """
#     Evaluate the alignment between two slices using the transfer matrix T.

#     Parameters:
#     - adata_0: AnnData object for slice A.
#     - adata_1: AnnData object for slice B.
#     - T: Transfer matrix from slice A to slice B.

#     Returns:
#     - alignment_score: A quantitative measure of alignment accuracy.
#     """
#     if dataset == "10X_DLPFC":
#         labels_0 = adata_0.obs['ground_truth'].astype('category')
#         labels_1 = adata_1.obs['ground_truth'].astype('category')
#     elif dataset == "Merfish_MouseBrainAging":
#         labels_0 = adata_0.obs['tissue'].astype('category')
#         labels_1 = adata_1.obs['tissue'].astype('category')

#     # Find the best match for each node based on the transfer matrix
#     aligned_indices = np.argmax(T, axis=1)
#     aligned_labels = labels_1.iloc[aligned_indices].values
#     original_labels = labels_0.values

#     # Create a DataFrame to track original and aligned labels
#     alignment_results = pd.DataFrame({
#         'Original': pd.Categorical(original_labels, categories=labels_0.cat.categories),
#         'Aligned': pd.Categorical(aligned_labels, categories=labels_0.cat.categories)
#     })

#     # Create a confusion matrix to compare original and aligned labels
#     conf_matrix = confusion_matrix(alignment_results['Original'], alignment_results['Aligned'], labels=labels_0.cat.categories)

#     # Calculate the percentage alignment for each layer
#     percentages = conf_matrix / conf_matrix.sum(axis=1, keepdims=True) * 100

#     # Visualization of the percentage-based confusion matrix with colorbar from 0 to 100
#     plt.figure(figsize=(4, 3))
#     sns.heatmap(percentages, annot=True, fmt=".1f", cmap="Blues", 
#                 vmin=0, vmax=100,  # Set colorbar scale from 0 to 100
#                 xticklabels=labels_0.cat.categories, yticklabels=labels_0.cat.categories)
#     plt.xlabel('Aligned Label')
#     plt.ylabel('Original Label')
#     plt.title('Percentage Confusion Matrix of Spatial Domain Alignment')
#     if savefig_path:
#         plt.savefig(savefig_path, bbox_inches='tight')
#     plt.show()

#     # Compute alignment accuracy as the sum of diagonal percentages (correct mappings)
#     alignment_score = np.trace(percentages) / np.sum(percentages)
#     print(f"Alignment Score (Accuracy): {alignment_score:.2f}")

#     return alignment_score
    
from sklearn.metrics import accuracy_score, f1_score, adjusted_rand_score

def evaluate_alignment_score(adata_0, adata_1, T):
    labels_0 = adata_0.obs['ground_truth'].values
    labels_1 = adata_1.obs['ground_truth'].values
    
    aligned_indices = np.argmax(T, axis=1)
    aligned_labels = labels_1[aligned_indices]

    accuracy = accuracy_score(labels_0, aligned_labels)
    f1 = f1_score(labels_0, aligned_labels, average='weighted')
    ari = adjusted_rand_score(labels_0, aligned_labels)

    print(f"Accuracy: {accuracy:.4f}")
    print(f"F1-Score: {f1:.4f}")
    print(f"Adjusted Rand Index: {ari:.4f}")

    return accuracy, f1, ari

