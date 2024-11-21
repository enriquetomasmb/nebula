import asyncio
import ipaddress
import logging
import os
from scipy.spatial.distance import pdist, squareform

import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

from nebula.core.selectors.selector import Selector
import torch
from nebula.core.utils.locker import Locker


class DistributionSelector(Selector):
    def __init__(self, config=None):
        super().__init__(config)
        self.config = config
        self.stop_training = False
        self.already_activated = False
        self.final_list = False
        self.number_votes = 100000
        self.threshold = 0
        self.rounds_without_training = 0
        logging.info("[DistributionSelector] Initialized")
        
        self.embedding_lock = Locker(name="embedding_lock", async_lock=True)
        self.own_embedding = None
        self.own_embedding_cov = None
        self.node_embeddings = {}
        self.node_embeddings_cov = {}
        self.selected_nodes = set()
        self.threshold_multiplier = 1.0 # Multiplier for the standard deviation to define the dynamic threshold for similarity (default: 1.0, equivalent to mean + std)
        
    async def get_embeddings(self, model, dataloader):
        logging.info("[DistributionSelector] Getting embeddings")
        idx = self.config.participant['device_args']['idx']
        logging.info(f"[DistributionSelector] Participant index: {idx}")
        # timeout = round(np.random.uniform(0, 20), 2)
        timeout = idx * 5
        logging.info(f"[DistributionSelector] Sleeping for {timeout} seconds")
        await asyncio.sleep(timeout)
        logging.info(f"[DistributionSelector] Waking up after {timeout} seconds")
        logging.info(f"Model is in {'eval' if model.training else 'train'} mode")
        logging.info(f"Tamaño del DataLoader: {len(dataloader)} batches")
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # device = torch.device('cpu')
        logging.info(f"Device: {device}")
        logging.info(f"Is CUDA available: {torch.cuda.is_available()}")
        if device.type == 'cuda':
            logging.info(f"Current CUDA device: {torch.cuda.current_device()}")
            logging.info(f"Device name: {torch.cuda.get_device_name(device)}")
        model = model.to(device)
        model.eval()
        embeddings_list = []
        with torch.no_grad():
            for batch_idx, (images, labels) in enumerate(dataloader):
                try:
                    logging.info(f"Procesando batch {batch_idx+1}/{len(dataloader)}")
                    images = images.to(device)
                    outputs = model(images)
                    outputs = outputs.view(outputs.size(0), -1)  # Flatten embeddings
                    # Print class distribution
                    class_counts = torch.bincount(labels)
                    logging.info(f"Class distribution in batch {batch_idx+1}: {class_counts}")
                    embeddings_list.append(outputs.cpu())
                except Exception as e:
                    logging.error(f"Error al procesar el batch {batch_idx+1}: {e}")
                    raise e
        embeddings = torch.cat(embeddings_list, dim=0)
        self.own_embedding = embeddings.mean(dim=0)
        self.own_embedding_cov = torch.cov(embeddings.T)
        del model, embeddings_list, embeddings
        
        # embeddings_np = embeddings.cpu().numpy()
        # Apply PCA to reduce dimensionality
        #pca = PCA(n_components=50)
        #embeddings_pca = pca.fit_transform(embeddings_np)
        #embeddings_pca = torch.from_numpy(embeddings_pca)
        #self.own_embedding = embeddings_pca.mean(dim=0)
        #self.own_embedding_cov = torch.cov(embeddings_pca.T)
        #del model, embeddings_list, embeddings, embeddings_np, embeddings_pca
        return self.own_embedding, self.own_embedding_cov
    
    async def add_embedding(self, node, embedding, embedding_cov, current_neighbors):
        logging.info(f"[DistributionSelector] Received embedding from {node}")
        self.node_embeddings[node] = embedding
        self.node_embeddings_cov[node] = embedding_cov
        logging.info(f"[DistributionSelector] Current embeddings: {len(self.node_embeddings)}")
        logging.info(f"[DistributionSelector] Current neighbors: {len(current_neighbors)}")
        if len(self.node_embeddings) == len(current_neighbors):
            if self.own_embedding is None:
                logging.info(f"[DistributionSelector] Own embedding not computed yet")
                while self.own_embedding is None or self.own_embedding_cov is None:
                    logging.info(f"[DistributionSelector] Waiting for own embedding")
                    await asyncio.sleep(1)
            logging.info(f"[DistributionSelector] Including own embedding")
            self.node_embeddings[self.config.participant['network_args']['addr']] = self.own_embedding
            self.node_embeddings_cov[self.config.participant['network_args']['addr']] = self.own_embedding_cov
            logging.info(f"[DistributionSelector] All embeddings received")
            def sort_key(item):
                ip, port = item[0].split(':')
                return (ipaddress.ip_address(ip), int(port))
            self.node_embeddings = {k: v for k, v in sorted(self.node_embeddings.items(), key=sort_key)}
            self.node_embeddings_cov = {k: v for k, v in sorted(self.node_embeddings_cov.items(), key=sort_key)}
            logging.info(f"[DistributionSelector] Sorted embeddings")
            logging.info(f"[DistributionSelector] Embeddings (keys): {self.node_embeddings.keys()}")
            logging.info(f"[DistributionSelector] Embeddings:\n{self.node_embeddings}")
            await self.get_similarity_embeddings()
            await self.embedding_lock.release_async()
            
    def compute_cosine_similarity(self, embeddings):
        norm_embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
        similarity_matrix = np.dot(norm_embeddings, norm_embeddings.T)
        logging.info("Cosine similarity matrix computed.")
        return similarity_matrix
    
    def compute_distance_similarity(self, embeddings):
        distance_matrix = squareform(pdist(embeddings, 'euclidean'))
        # Normalize distances and convert to similarity scores
        max_distance = distance_matrix.max()
        if max_distance == 0:
            similarity_matrix = np.ones_like(distance_matrix)
        else:
            similarity_matrix = 1 - (distance_matrix / max_distance)
        logging.info("Distance-based similarity matrix computed.")
        return similarity_matrix
    
    def compute_bhatta_similarity(self, embeddings, covariances):
        # Stack embeddings and covariances into tensors
        embeddings = torch.stack(embeddings)  # Shape: (N, D)
        covariances = torch.stack(covariances)  # Shape: (N, D, D)
        
        N, D = embeddings.shape
        
        # Compute pair-wise mean differences
        mean_diff = embeddings.unsqueeze(1) - embeddings.unsqueeze(0)  # Shape: (N, N, D)
        
        # Compute average covariances
        cov_avg = 0.5 * (covariances.unsqueeze(1) + covariances.unsqueeze(0))  # Shape: (N, N, D, D)
        
        # Compute log determinant of average covariances
        cov_avg_logdet = torch.logdet(cov_avg)  # Shape: (N, N)
        
        # Compute log determinants of individual covariances
        cov_logdet = torch.logdet(covariances)  # Shape: (N,)
        cov_logdet_sum = 0.5 * (cov_logdet.unsqueeze(1) + cov_logdet.unsqueeze(0))  # Shape: (N, N)
        
        # Calculate term1
        term1 = 0.5 * cov_avg_logdet - cov_logdet_sum  # Shape: (N, N)
        
        # Compute inverse of average covariances
        cov_avg_inv = torch.inverse(cov_avg)  # Shape: (N, N, D, D)
        
        # Compute Mahalanobis distance (term2)
        mean_diff = mean_diff.unsqueeze(-1)  # Shape: (N, N, D, 1)
        term2 = 0.125 * (mean_diff.transpose(-2, -1) @ cov_avg_inv @ mean_diff).squeeze()  # Shape: (N, N)
        
        # Compute Bhattacharyya distance
        bhatta_dist = term1 + term2  # Shape: (N, N)
        
        # Compute Bhattacharyya similarity
        similarity_matrix = torch.exp(-bhatta_dist)  # Shape: (N, N)
        
        # Convert to NumPy array if needed
        similarity_matrix = similarity_matrix.cpu().numpy()
        
        logging.info("Bhattacharyya similarity matrix computed.")
        del embeddings, covariances, mean_diff, cov_avg, cov_avg_logdet, cov_logdet, cov_logdet_sum, term1, cov_avg_inv, term2, bhatta_dist
        return similarity_matrix
        
    async def get_similarity_embeddings(self, method='cosine'):
        logging.info("[DistributionSelector] Getting similarity between embeddings")
        nodes = list(self.node_embeddings.keys())
        own_node = self.config.participant['network_args']['addr']
        
        if not self.node_embeddings:
            logging.warning("No embeddings available for similarity computation.")
            return
        
        if self.node_embeddings_cov and len(self.node_embeddings) != len(self.node_embeddings_cov):
            logging.error("Inconsistent number of embeddings and covariance matrices.")
            return
                
        if method == 'cosine':
            embeddings = torch.stack([embedding.detach() for embedding in self.node_embeddings.values()]).numpy()
            similarity_matrix = self.compute_cosine_similarity(embeddings)
        elif method == 'distance':
            embeddings = torch.stack([embedding.detach() for embedding in self.node_embeddings.values()]).numpy()
            similarity_matrix = self.compute_distance_similarity(embeddings)
        elif method == 'bhatta':
            embeddings = [embedding.detach() for embedding in self.node_embeddings.values()]
            covariances = [cov.detach().to(embeddings[0].device) for cov in self.node_embeddings_cov.values()]
            similarity_matrix = self.compute_bhatta_similarity(embeddings, covariances)
        else:
            raise ValueError(f"Method {method} not supported")
        
        
        image_path = os.path.join(self.config.participant['tracking_args']['log_dir'], self.config.participant['scenario_args']['name'], f"participant_{self.config.participant['device_args']['idx']}_{method}_similarity_matrix.png")
        fig, ax = plt.subplots(figsize=(8, 8))
        cax = ax.matshow(similarity_matrix, cmap='viridis')
        fig.colorbar(cax)
        plt.title(f"{method.capitalize()} Similarity Matrix")
        plt.xticks(range(len(nodes)), nodes, rotation=90, fontsize=6)
        plt.yticks(range(len(nodes)), nodes, fontsize=6)
        plt.xlabel("Nodes")
        plt.ylabel("Nodes")
        plt.tight_layout()
        plt.savefig(image_path)
        plt.close()
        logging.info(f"similarity_matrix saved in {image_path}")
        
        logging.info(f"similarity_matrix: {similarity_matrix}")
        
        for i, node in enumerate(nodes):
            # For cosine similarity, higher values mean more similar
            # For distance-based similarity, higher values also mean more similar (since we invert the distances)
            similarity_scores = similarity_matrix[i]
            sorted_indices = np.argsort(-similarity_scores)
            sorted_nodes = np.array([nodes[idx] for idx in sorted_indices])
            sorted_scores = similarity_scores[sorted_indices]
            
            similar_nodes_with_scores = [f"{sorted_nodes[j]} [{sorted_scores[j]:.4f}]" for j in range(len(sorted_nodes))]
            logging.info(f"Node {node} has similar nodes: {', '.join(similar_nodes_with_scores)}")
                        
            if node == own_node:
                logging.info(f"--- [Own node '{node}'] ---")
                # Exclude own node
                similar_nodes = [n for n in sorted_nodes if n != own_node]
                similar_scores = np.delete(sorted_scores, np.where(sorted_nodes == own_node))
                similar_nodes_with_scores = [f"{similar_nodes[j]} [{similar_scores[j]:.4f}]" for j in range(len(similar_nodes))]
                logging.info(f"Own node '{own_node}' similar nodes with scores: {', '.join(similar_nodes_with_scores)}")
                
                # Compute mean and standard deviation of similarity scores
                mean_similarity = np.mean(similar_scores)
                std_similarity = np.std(similar_scores)
                
                # Define dynamic threshold
                threshold = mean_similarity + self.threshold_multiplier * std_similarity
                logging.info(f"Dynamic threshold for node '{own_node}': {threshold:.4f}")
                
                # Select nodes above the dynamic threshold
                above_threshold_indices = np.where(similar_scores >= threshold)[0]
                if len(above_threshold_indices) > len(similar_nodes) // 2:
                    selected_nodes = [similar_nodes[idx] for idx in above_threshold_indices]
                    logging.info(f"Own node '{own_node}' has similar nodes above dynamic threshold: {selected_nodes}")
                else:
                    selected_nodes = similar_nodes[:max(1, len(similar_nodes) // 2)]
                    logging.info(f"No nodes above dynamic threshold for own node. Selecting top similar nodes: {selected_nodes}")
                
        # Update the final selected nodes
        self.selected_nodes.update(selected_nodes)
        if own_node not in self.selected_nodes:
            self.selected_nodes.add(own_node)
        logging.info(f"[DistributionSelector] Final selected nodes: {self.selected_nodes}")
        
    def get_selected_nodes(self):
        logging.info(f"[DistributionSelector.get_selected_nodes] Selected nodes: {self.selected_nodes}")
        return self.selected_nodes 

    async def node_selection(self, node):
        logging.info(f"[DistributionSelector.node_selection] Selected nodes: {self.selected_nodes}")
        return self.selected_nodes
