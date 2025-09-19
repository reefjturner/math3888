import numpy as np
import pandas as pd
import networkx as nx


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
