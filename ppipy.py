import numpy as np
import pandas as pd
import networkx as nx
import igraph as ig
import leidenalg
import scipy as sp
from scipy.linalg import eigh
import scipy.sparse.linalg as spla


def initialise_PPI_network(protein_network, essential_proteins, remove_essentials=False, threshold=700.0, 
                       largest_connected_component=False):
    '''
    Given a PPI network, and a list of
    essential proteins, return a new pruned network that meets
    some given specifications.
    
    Parameters
    ----------
    protein_network : NetworkX graph
        PPI network.
    
    essential_proteins : list
        List of essential proteins on the network.
    
    remove_essentials : bool, optional (default=False)
        If true, then the method will return a PPI network
        where all the essential proteins are removed, as
        well as all the corresponding edges.
    
    threshold : float, optional (default=700)
        Minimum threshold of viability of the edge
        connection in the network. If below this 
        threshold, then the node is removed, as well 
        as the corresponding edges.
    
    Returns
    -------
    H : NetworkX Graph
        The filtered PPI network.
    '''
    ## remove all necessary nodes ##
    if remove_essentials:
         for node in essential_proteins:
            protein_network.remove_node(node)
    edges_to_remove = []
    for edge in protein_network.edges:
        edge_weights = list(protein_network.get_edge_data(edge[0], edge[1]).values())
        if edge_weights[0] < threshold:
            edges_to_remove.append(edge)
    for edge in edges_to_remove:
        protein_network.remove_edge(edge[0], edge[1])
    ## now create a copy that is unweighted ##
    H = nx.create_empty_copy(protein_network)
    for edge in protein_network.edges:
        H.add_edge(edge[0], edge[1])
    ## fix node names ##
    mapping = []
    for node in protein_network.nodes:
        mapping.append(node.removeprefix('4932.'))
    H = nx.relabel_nodes(H, dict(zip(H, mapping)), copy=False) # <-- cute little python hack i just learnt! (i feel very proud of myself)
    ## (if it is needed) return the largest connected subgraph ##
    if largest_connected_component:
        return H.subgraph(max(nx.connected_components(H), key=len))
    return H


def initialise_yeast_PPI_network(remove_essentials=False, threshold=700.0, largest_connected_component=False):
    '''
    Initialises the yeast PPI network.

    Expects that the yeast PPI network is in the main 
    directory with name "4932.protein.links.v12.0.txt",
    as well as a list of essential proteins with name
    "EssentialProteins_YeastMine_cerevisiae.csv".
    
    Parameters
    ----------
    remove_essentials : bool, optional (default=False)
        If true, then the method will return a PPI network
        where all the essential proteins are removed, as
        well as all the corresponding edges.
    
    threshold : float, optional (default=700)
        Minimum threshold of viability of the edge
        connection in the network. If below this 
        threshold, then the node is removed, as well 
        as the corresponding edges.

    largest_connected_component: bool, optional (default=False)
        If true, return the largest connected component in
        the network.

    Returns
    -------
    H : NetworkX Graph
        The filtered yeast PPI network.
    '''
    protein_network = nx.read_weighted_edgelist("4932.protein.links.v12.0.txt",comments="#",nodetype=str)
    df = pd.read_table("EssentialProteins_YeastMine_cerevisiae.csv", sep=",", header = None)
    df.columns = ['primary identifier','secondary identifier','organism short name','symbol','description']
    essential_nodes = [n for n in protein_network.nodes() if n[5:] in list(df['secondary identifier'])]
    return initialise_PPI_network(protein_network, essential_nodes, remove_essentials=remove_essentials, threshold=threshold, largest_connected_component=largest_connected_component)


def rank_centralities(centrality_dict):
    '''
    Given a dict of centrality scores, returns an ordered 
    list of the centralities and nodes.

    Parameters
    ----------
    '''
    centrality_list = []
    return sorted(centrality_dict.items(), key=lambda item: item[1], reverse=True)


def get_adjacent_comms(G, node, coms):
  """takes a node, a graph and a list of communities, then returns the communities that are adjacent to the community the node is in."""
    # identify com for node:
    for com in coms:
        if node in com:
            com_of_interest = com
    adj_comms = []
    for node in com_of_interest:
        for neigh in G.neighbors(node):
            for comm in coms:
                if (neigh in comm) and (comm not in adj_comms):
                    adj_comms.append(comm)
    # this likely contains the community of interest, we want to remove it:
    for i in range(len(adj_comms)):
        if adj_comms[i] == com_of_interest:
            kill_i = i
    del adj_comms[kill_i]
    return adj_comms


def print_pairwise_shortest_paths(nodes, G):
    """takes a list of nodes and a network, then prints out the pairwise shortest paths between them as well as their lengths"""
    print("Pairwise shortest-paths between BCMB seeds:")
    # iterate over all nodes
    for i in range(len(nodes)):
        # iterate over every pair
        for j in range(i+1, len(nodes)):
            a, b = nodes[i], nodes[j]
            # check that they're in the graph and connected
            if a in G and b in G and nx.has_path(G, a, b):
                # determine shortest paths, as well as the length of them and how many there are.
                p = list(nx.all_shortest_paths(G, a, b))
                d = len(p[0]) - 1
                n = len(p)
                print(f"  node {a} to {b}: has distance: {d}, with {n} path(s): {p}")
            else:
                print(f"  node {a} to {b} has no path (disconnected after threshold), or either {a} or {b} are not in the graph.")


def leiden_katz_analysis(G0, bio_proteins, custom_betas, essential_proteins, resolution = 10.0, top = 10, SEED=1):
    """
    Runs Leiden community detection and Katz centrality analysis on a NetworkX graph.

    Parameters
    ----------
    G0 : networkx Graph
        Protein-protein interaction graph.
    bio_proteins : list[str]
        Proteins from bio group.
    custom_betas : dict[str, float]
        Custom beta values for specific proteins.
    essential_proteins : set[str]
        Set of essential proteins for annotation.
    resolution : float, optional
        Resolution parameter for Leiden algorithm (default=10.0).
    """

    # Check if the proteins are still in the graph
    print("Proteins in filtered graph?")
    for p in bio_proteins:
        print(f"  {p}: {p in G0}")

    print(f"\nGraph has {G0.number_of_nodes()} nodes and {G0.number_of_edges()} edges")

    # Convert to igraph as Leiden only works with igraph objects, not NetworkX graphs.
    g = ig.Graph.from_networkx(G0)

    # run Leiden community detection
    partition = leidenalg.find_partition(
        g,
        leidenalg.RBConfigurationVertexPartition,
        resolution_parameter=resolution,
        seed=SEED
    )

    # Extract partition into list of lists of node names
    partition_list = []
    for comm in partition:
        comm_nodes = [g.vs[idx]["_nx_name"] for idx in comm]
        partition_list.append(comm_nodes)

    print(f"\nLeiden found {len(partition_list)} communities (resolution={resolution})")
    # for i, comm in enumerate(partition_list[:5]):
    #     print(f"  Community {i}: size={len(comm)}")

    # Analyze only the communities with bio_proteins
    found_any = False
    for i, comm in enumerate(partition_list):
        hits = set(comm) & set(bio_proteins)
        if hits:
            found_any = True
            print(f"\nCommunity {i} contains {hits}, size={len(comm)}")

            # Make a subgraph of this community in NetworkX
            subG = G0.subgraph(comm)

            # Compute alpha for Katz centrality
            if subG.number_of_nodes() > 1 and subG.number_of_edges() > 0:
                A = nx.to_numpy_array(subG, nodelist=list(subG.nodes()))
                largest_eigval = np.real(spla.eigs(A, k=1, which="LR", return_eigenvectors=False)[0])
                alpha = 0.9 / largest_eigval
            else:
                A = nx.to_numpy_array(subG, nodelist=list(subG.nodes()))
                alpha = 0.005

            nodes = list(subG.nodes())
            n = len(nodes)

            # Make the custom beta vector
            b = np.ones(n)
            for j, node in enumerate(nodes):
                if node in custom_betas:
                    b[j] = float(custom_betas[node])

            # Solve Katz equation: (I - alpha*A)x = b
            I = np.eye(n)
            x = np.linalg.solve(I - alpha * A, b)

            # Normalize vector to avoid huge values
            x = x / np.linalg.norm(x, 2)

            katz = {nodes[j]: x[j] for j in range(n)}

            
            # Show the top nodes in each imprtant community
            top10 = sorted(katz.items(), key=lambda kv: kv[1], reverse=True)[:top]
            print("\nTop ",top," nodes by Katz centrality in this community:")
            for rank, (node, score) in enumerate(top10, start=1):
                status = " (essential)" if node in essential_proteins else ""
                print(f"  {rank}. {node}: {score:.6f}{status}")

            # Print proteins of interest explicitly
            print("\nProteins of interest in this community:")
            for p in bio_proteins:
                if p in katz:
                    status = " (essential)" if p in essential_proteins else ""
                    print(f"  {p}: {katz[p]:.6f}{status}")

    if not found_any:
        print("\n None of the target proteins were found in any Leiden community.")



