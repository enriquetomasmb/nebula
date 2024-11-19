import logging
import os
from statistics import mean, stdev
import random
from scipy.spatial.distance import pdist, squareform

import numpy as np
import matplotlib.pyplot as plt

from nebula.core.selectors.selector import Selector
from nebula.core.utils.helper import cosine_metric
import torch
from sklearn.metrics.pairwise import cosine_similarity
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
        self.node_embeddings = {}
        self.selected_nodes = set()
        
    async def get_embeddings(self, model, dataloader):
        logging.info("[DistributionSelector] Getting embeddings")
        logging.info(f"Model is in {'eval' if model.training else 'train'} mode")
        logging.info(f"Tamaño del DataLoader: {len(dataloader)} batches")
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logging.info(f"Device: {device}")
        logging.info(f"Is CUDA available: {torch.cuda.is_available()}")
        logging.info(f"Current CUDA device: {torch.cuda.current_device()}")
        logging.info(f"CUDA device name: {torch.cuda.get_device_name(device)}")
        model = model.to(device)
        model.eval()
        embedding_sum = None
        total_samples = 0
        with torch.no_grad():
            for batch_idx, (images, labels) in enumerate(dataloader):
                try:
                    logging.info(f"Procesando batch {batch_idx+1}/{len(dataloader)}")
                    images = images.to(device)
                    outputs = model(images)
                    batch_size = outputs.size(0)
                    # Print class distribution
                    class_counts = torch.bincount(labels)
                    logging.info(f"Class distribution in batch {batch_idx+1}: {class_counts}")
                    
                    if embedding_sum is None:
                        embedding_sum = outputs.sum(dim=0)
                    else:
                        embedding_sum += outputs.sum(dim=0)
                    total_samples += batch_size
                except Exception as e:
                    logging.error(f"Error al procesar el batch {batch_idx+1}: {e}")
                    raise e
        mean_embedding = embedding_sum / total_samples
        self.own_embedding = mean_embedding.cpu()
        return self.own_embedding
    
    async def add_embedding(self, node, embedding, current_neighbors):
        logging.info(f"[DistributionSelector] Received embedding from {node}")
        self.node_embeddings[node] = embedding
        logging.info(f"[DistributionSelector] Current embeddings: {len(self.node_embeddings)}")
        logging.info(f"[DistributionSelector] Current neighbors: {len(current_neighbors)}")
        if len(self.node_embeddings) == len(current_neighbors):
            logging.info(f"[DistributionSelector] Including own embedding")
            self.node_embeddings[self.config.participant['network_args']['addr']] = self.own_embedding
            logging.info(f"[DistributionSelector] All embeddings received")
            # Sort embeddings by node name
            self.node_embeddings = {k: v for k, v in sorted(self.node_embeddings.items(), key=lambda item: item[0])}
            logging.info(f"[DistributionSelector] Sorted embeddings")
            logging.info(f"[DistributionSelector] Embeddings (keys): {self.node_embeddings.keys()}")
            logging.info(f"[DistributionSelector] Embeddings:\n{self.node_embeddings}")
            self.get_similarity_embeddings()
            await self.embedding_lock.release_async()
            
    def compute_cosine_similarity(self, embeddings):
        similarity_matrix = cosine_similarity(embeddings)
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
        
    def get_similarity_embeddings(self, method='cosine'):
        logging.info("[DistributionSelector] Getting similarity between embeddings")
        nodes = list(self.node_embeddings.keys())
        own_node = self.config.participant['network_args']['addr']
        
        if not self.node_embeddings:
            logging.warning("No embeddings available for similarity computation.")
            return
        
        embedding_shapes = [embedding.shape for embedding in self.node_embeddings.values()]
        if len(set(embedding_shapes)) > 1:
            logging.error("Embeddings have inconsistent shapes.")
            return
        
        embeddings = torch.stack([embedding.detach() for embedding in self.node_embeddings.values()]).numpy()
        
        # Check variance of embeddings
        var_path = os.path.join(self.config.participant['tracking_args']['log_dir'], self.config.participant['scenario_args']['name'], f"participant_{self.config.participant['device_args']['idx']}_embedding_variance.png")
        embedding_std = embeddings.std(axis=0)
        plt.plot(embedding_std)
        plt.title("Embedding Standard Deviation")
        plt.xlabel("Dimension")
        plt.ylabel("Standard Deviation")
        plt.savefig(var_path)
        plt.close()
        logging.info(f"Embedding variance saved in {var_path}")
        
        
        if method == 'cosine':
            similarity_matrix = self.compute_cosine_similarity(embeddings)
        elif method == 'distance':
            similarity_matrix = self.compute_distance_similarity(embeddings)
        else:
            raise ValueError(f"Method {method} not supported")
        
        
        image_path = os.path.join(self.config.participant['tracking_args']['log_dir'], self.config.participant['scenario_args']['name'], f"participant_{self.config.participant['device_args']['idx']}_{method}_similarity_matrix.png")
        fig, ax = plt.subplots()
        cax = ax.matshow(similarity_matrix, cmap='viridis')
        fig.colorbar(cax)
        plt.title(f"{method.capitalize()} Similarity Matrix")
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
                # Exclude own node
                similar_nodes = [n for n in sorted_nodes if n != own_node]
                similar_scores = np.delete(sorted_scores, np.where(sorted_nodes == own_node))
                similar_nodes_with_scores = [f"{similar_nodes[j]} [{similar_scores[j]:.4f}]" for j in range(len(similar_nodes))]
                logging.info(f"Own node '{own_node}' similar nodes with scores: {', '.join(similar_nodes_with_scores)}")
                
        threshold = 0.9
        
        # Select nodes above the threshold
        above_threshold_indices = np.where(similar_scores >= threshold)[0]
        if len(above_threshold_indices) > 0:
            selected_nodes = [similar_nodes[idx] for idx in above_threshold_indices]
            logging.info(f"Own node '{own_node}' has similar nodes above threshold: {selected_nodes}")
        else:
            selected_nodes = similar_nodes[:max(1, len(similar_nodes) // 2)]
            logging.info(f"No nodes above threshold for own node. Selecting top similar nodes: {selected_nodes}")
            
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
