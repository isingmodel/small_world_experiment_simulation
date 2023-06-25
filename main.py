import igraph as ig
import random
import numpy as np
from tqdm import tqdm


class SmallWorldSimulation:
    def __init__(self, num_nodes, p_vanish):
        self.num_nodes = num_nodes
        self.p_vanish = p_vanish
        self.graph, self.locations = self.create_network()

    def create_network(self):
        # Set parameters
        num_clusters_side = int(np.sqrt(self.num_nodes)/10)  # Number of clusters along one side of the grid
        num_clusters = int(num_clusters_side * num_clusters_side)  # Total number of clusters
        nodes_per_cluster = self.num_nodes // num_clusters  # Nodes in each cluster
        self.num_nodes = num_clusters * nodes_per_cluster

        in_cluster_prob = min(150 / nodes_per_cluster, 1)
        inter_cluster_edges = num_clusters * 10
        print(f"number of clusters: {num_clusters}")
        print(f"nodes per cluster: {nodes_per_cluster}")

        # Create separate clusters
        clusters = [ig.Graph.Erdos_Renyi(n=nodes_per_cluster,
                                         p=in_cluster_prob) for _ in range(num_clusters)]

        # Assign each cluster a location on a grid
        cluster_locations = {}
        for i in range(num_clusters_side):
            for j in range(num_clusters_side):
                cluster_id = i * num_clusters_side + j
                cluster_locations[cluster_id] = (i / (num_clusters_side - 1), j / (num_clusters_side - 1))

        # Assign locations to each node based on its cluster's location
        locations = {}
        for i, cluster in enumerate(clusters):
            for node in range(cluster.vcount()):
                node_global_id = i * nodes_per_cluster + node  # Global node ID across all clusters
                cluster_location = np.array(cluster_locations[i])
                node_location = cluster_location + np.random.normal(0, 0.01, 2)  # Nodes are close to their cluster
                locations[int(node_global_id)] = tuple(node_location)

        # Combine all clusters into a single graph
        g = clusters[0]
        for i in range(1, num_clusters):
            g = g + clusters[i]

        # Add inter-cluster edges to create a small-world effect
        for _ in tqdm(range(inter_cluster_edges)):
            # Choose two clusters randomly
            cluster1, cluster2 = np.random.choice(num_clusters, 2, replace=False)
            # Choose a node from each cluster
            node1 = np.random.choice(range(cluster1 * nodes_per_cluster, (cluster1 + 1) * nodes_per_cluster))
            node2 = np.random.choice(range(cluster2 * nodes_per_cluster, (cluster2 + 1) * nodes_per_cluster))
            # Add an edge between the chosen nodes
            g.add_edge(node1, node2)

        return g, locations


    def calculate_distance(self, node1, node2):
        # Calculate Euclidean distance between two nodes
        loc1 = np.array(self.locations[node1])
        loc2 = np.array(self.locations[node2])
        return max(np.linalg.norm(loc1-loc2), 0.0001)

    def simulate(self):
        source = random.randint(0, self.num_nodes - 1)
        target = random.randint(0, self.num_nodes - 1)
        while target == source:
            target = random.randint(0, self.num_nodes - 1)

        current_node = source
        prev_node = None
        steps = 0

        while True:
            # Mail vanishes with probability p_vanish
            if random.random() < self.p_vanish:
                return -1  # Mail vanished

            # Check if mail reached target
            if current_node == target:
                return steps  # Mail reached target
            
            neighbors = self.graph.neighbors(current_node)
                
            # Can't send mail back to the node that sent it to us
            if prev_node is not None and prev_node in neighbors:
                neighbors.remove(prev_node)
                
            if len(neighbors) == 0:
                return -2

            # Choose next node based on distance to target

            distances = [self.calculate_distance(neighbor, target) for neighbor in neighbors]
            probabilities = [1./dist for dist in distances]  # closer nodes have higher probability
            weights = [p/sum(probabilities) for p in probabilities]  # normalize probabilities

            prev_node = int(current_node)
            current_node = int(np.random.choice(neighbors, p=weights))
            
            steps += 1


if __name__ == "__main__": 
	num_nodes = 1000
	p_vanish = 0.05
	num_trials = 100

	simulation = SmallWorldSimulation(num_nodes, p_vanish)

	step_list = []
	for _ in tqdm(range(num_trials)):
	    steps = simulation.simulate()
	    step_list.append(steps)

	print(step_list)
	print(np.average([x for x in step_list if x > 1]))



