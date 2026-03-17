import numpy as np
import math
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.patches import Circle

def setup_three_point_graph(Pa, Pb, Pc, Pf=None):
    # Define three terminal points
    

    # Create graph and add nodes
    G = nx.Graph()
    G.add_node('Pa', pos=(Pa.x, Pa.y))
    G.add_node('Pb', pos=(Pb.x, Pb.y))
    G.add_node('Pc', pos=(Pc.x, Pc.y))

    if Pf is None:
        # Calculate distances between terminals
        PaPb = math.dist(Pa.coords[0], Pb.coords[0])
        PaPc = math.dist(Pa.coords[0], Pc.coords[0])
        PbPc = math.dist(Pb.coords[0], Pc.coords[0])
        
        edges = [
            (PaPb, 'Pa', 'Pb'),
            (PaPc, 'Pa', 'Pc'),
            (PbPc, 'Pb', 'Pc')
        ]
        edges.sort(key=lambda x: x[0])  # Sort by distance
        edges = edges[:2] 
        
        
    else:
        G.add_node('Pf', pos=(Pf.x, Pf.y))

        PaPf = math.dist(Pa.coords[0], Pf.coords[0])
        PbPf = math.dist(Pb.coords[0], Pf.coords[0])
        PcPf = math.dist(Pc.coords[0], Pf.coords[0])
        
        edges = [
            (PaPf, 'Pa', 'Pf'),
            (PbPf, 'Pb', 'Pf'),
            (PcPf, 'Pc', 'Pf')
        ]

    # Add edges from Pf to each terminal
    for dist, u, v in edges:
        G.add_edge(u, v, weight=dist)


    # finish setting up the graph and return it along with the points
    G = finalize_graph_creation(G)
    return G

def finalize_graph_creation(G, graph_side_length=1.0, scale_pos=False):
    """
    Standardize the graph G by assigning node types, coordinates, and edge lengths.
    Follows TSPLIB-like node labeling and coordinate mapping.

    :param G: NetworkX Graph object.
    :param end_node_mode: Mode for selecting terminal nodes ("4corners" or "convexhull").
    :param graph_side_length: Scaling factor for coordinates.
    :param scale_pos: Whether to apply the side length scaling to the 'pos' attribute.
    :return: Standardized NetworkX Graph G.
    """
    # Relabel nodes to strings for consistency
    mapping = {node: str(node) for node in G.nodes() if not isinstance(node, str)}
    G = nx.relabel_nodes(G, mapping)
    
    # Assign default types
    for node in G.nodes():
        if 'type' not in G.nodes[node]:
            G.nodes[node]['type'] = 'unknown_node'
    
    # Scale coordinates if requested
    if scale_pos:
        for node in G.nodes():
            pos = G.nodes[node]['pos']
            G.nodes[node]['pos'] = (pos[0] * graph_side_length, pos[1] * graph_side_length)
    
    # Compute edge lengths (EUC_2D convention)
    G = compute_dist_cartesian(G)
    
    return G

def compute_dist_cartesian(G):
    """
    Compute Euclidean distances (EUC_2D) for all edges in G based on 'pos'.
    
    :param G: NetworkX Graph object.
    :return: G with 'length' attribute on edges.
    """
    for u, v in G.edges():
        pos_u = G.nodes[u]['pos']
        pos_v = G.nodes[v]['pos']
        # Euclidean distance rounded to 5 decimals as per legacy behavior
        dist = round(math.dist(pos_u, pos_v), 5)
        G.edges[u, v]['length'] = dist
    return G

def get_pos(G):
    """
    Get the positions of the nodes in the graph for visualization or further processing.

    :return: Dictionary mapping node labels to (x, y) coordinates.
    """
    return nx.get_node_attributes(G, 'pos')

def dist(graph, node_a, node_b):
    """
    Calculate the Euclidean distance between two nodes in the graph.
            
    :param graph: NetworkX Graph object.
    :param node_a: Label of the first node.
    :param node_b: Label of the second node.
    :return: Euclidean distance rounded to 5 decimal places.
    """
    dx = np.abs(graph.nodes[node_a]['pos'][0] - graph.nodes[node_b]['pos'][0])
    dy = np.abs(graph.nodes[node_a]['pos'][1] - graph.nodes[node_b]['pos'][1])
    dist = np.round(np.sqrt(np.square(dx) + np.square(dy)), 5)
    return dist

def draw_graph(G, title="", disc_size=1.0, chosen_edges=None, extra_points=None, extra_discs=None, extra_discs_layers=1, show_nodes_labels=False, show_edges_label=False):
    """
    A professional visualization tool for the network graph G.
    Displays terminal nodes with extra nodes and communication radii.

    :param G: NetworkX Graph.
    :param title: Plot title.
    :param disc_size: Radius R for communication coverage discs.
    :param chosen_edges: List of edges (u, v) to highlight.
    :param extra_points: List or dictionary of coordinates for extra nodes.
    :param extra_discs: List of coordinates for extra coverage discs.
    :param extra_discs_layers: Number of concentric layers for the discs.
    """
    pos = get_pos(G)
    length = nx.get_edge_attributes(G, 'length')
    
    fig, ax = plt.subplots(figsize=(10, 10))
    # fig, ax = plt.figure(figsize=(8, 8))
    
                               
    # Draw  Nodes
    nx.draw_networkx_nodes(G, pos, nodelist=G.nodes(), node_size=300, 
                            node_color='white', edgecolors='black')

    # Draw Standard Edges
    nx.draw_networkx_edges(G, pos, width=0.5, alpha=0.5)
    
    # Highlight Edges
    if chosen_edges:
        nx.draw_networkx_edges(G, pos, edgelist=chosen_edges, edge_color="#2CBAFA", width=2.5)

    if show_nodes_labels:
        labels = {node: node for node in G.nodes()}
        nx.draw_networkx_labels(G=G, pos=pos, labels=labels, font_size=10, font_color='black')

    if show_edges_label:
            nx.draw_networkx_edge_labels(G=G, pos=pos, edge_labels=length)

    # Draw Extra Points
    if extra_points:
        if isinstance(extra_points, dict):
            # Handle Dictionary: { 'name': [x, y] }
            for name, coords in extra_points.items():
                ax.scatter(*coords, s=150, edgecolor='b', facecolor='none', marker='^')
                plt.text(coords[0], coords[1], name, fontsize=12, color='red')

        elif isinstance(extra_points, list):
            # Handle List: [[x1, y1], [x2, y2]]
            for point in extra_points:
                ax.scatter(*point, s=50, edgecolor='b', facecolor='b', marker='s')
            
    # Draw coverage Discs
    if extra_discs:
        for disc_center in extra_discs:
            for i in range(extra_discs_layers):
                circle = Circle(disc_center, radius=disc_size*(i+1), edgecolor='blue', facecolor='none', linestyle='--', alpha=0.3)
                ax.add_patch(circle)
    
    
    plt.axis('equal')
    ax.set_title(title)
    ax.set_aspect('equal')
    # plt.tight_layout()
    plt.show()

def draw_triangural(graph, title="", disc_size=1, chosen_edges=[], chosen_edges2=[], updated_edges=[], chosen_nodes=[], chosen_nodes2=[], extra_points=[], extra_points_dict={}, convex_discs=[], convex_discs2=[], show_node_label = True, show_edge_label=True):
    """
    Draws the input graph.

    Parameters:
        graph (NetworkX graph): Input graph.
        title (str): Title of the plot.
        edge_label (bool): Whether to display edge labels.

    Returns:
        None
    """
    node_pos = nx.get_node_attributes(graph, 'pos')
    edge_pos = node_pos
    label_pos = node_pos #pos_nudge(node_pos, 0.002, 0.002, nx.diameter(graph))
    length = nx.get_edge_attributes(graph, 'length')
    
    none_type_node = []
    repeater_nodes = []
    end_nodes = []
    
    node_labels = {}
    end_node_labels = {}
    repeater_node_labels = {}
    chosen_node_labels = {}
    chosen_node_labels2 = {}
    
    for node, nodedata in graph.nodes.items():
        if not nodedata['type']:
            none_type_node.append(node)
            if show_node_label:
                node_labels[node] = node
        else:
            if nodedata['type'] == 'repeater_node':
                repeater_nodes.append(node)
                if show_node_label:
                    repeater_node_labels[node] = node
            elif nodedata['type'] == 'steiner_node':
                repeater_nodes.append(node)
                if show_node_label:
                    repeater_node_labels[node] = node
            elif nodedata['type'] == 'end_node':
                end_nodes.append(node)
                if show_node_label:
                    end_node_labels[node] = node
            else:
                NotImplementedError(f"Unknowed node type: {nodedata['type']}")
        
        if node in chosen_nodes:
            chosen_node_labels[node] = node
        if node in chosen_nodes2:
            chosen_node_labels2[node] = node
    
    fig, ax = plt.subplots(figsize=(10, 10))
    
    #draw nodes
    if end_nodes:
        end_nodes = nx.draw_networkx_nodes(G=graph, pos=node_pos, nodelist=end_nodes, node_shape='s', node_size=750,
                                       node_color=[[1.0, 120 / 255, 0.]], label="End Node", linewidths=3)
        end_nodes.set_edgecolor(["k"])
        if end_node_labels:
            nx.draw_networkx_labels(G=graph, pos=label_pos, labels=end_node_labels, font_size=20, 
                                font_weight="bold", font_color="b", font_family='serif')
    if repeater_nodes:
        rep_nodes = nx.draw_networkx_nodes(G=graph, pos=node_pos, nodelist=repeater_nodes, node_size=350,
                                       node_color=[[1, 1, 1]], label="Repeater Node")
        rep_nodes.set_edgecolor(["k"])
        if repeater_node_labels:
            nx.draw_networkx_labels(G=graph, pos=label_pos, labels=repeater_node_labels, font_size=12, font_weight="bold") #12
        
    if chosen_nodes:
        chosen_nodes = nx.draw_networkx_nodes(G=graph, pos=node_pos, nodelist=chosen_nodes, node_size=750,
                                       node_color="#2CBAFA", label="Chosen Node")
        chosen_nodes.set_edgecolor(["k"])
        # if chosen_node_labels and show_node_label:
        #     nx.draw_networkx_labels(G=graph, pos=label_pos, labels=chosen_node_labels, font_size=20, 
        #                             font_weight="bold", font_color="k", font_family='serif')
            
    if chosen_nodes2:
        chosen_nodes2 = nx.draw_networkx_nodes(G=graph, pos=node_pos, nodelist=chosen_nodes2, node_size=750,
                                   node_color="#F98181", label="Chosen Node2")
        chosen_nodes2.set_edgecolor(["k"])
        # if chosen_node_labels2 and show_node_label:
        #     nx.draw_networkx_labels(G=graph, pos=label_pos, labels=chosen_node_labels2, font_size=20, 
        #                             font_weight="bold", font_color="k", font_family='serif')
    
    if extra_points:
        for point in extra_points:
            # extra_node = Circle(point, radius=0.2, edgecolor='b', facecolor='none')
            ax.scatter(*point, s=200, edgecolor='b', facecolor='none', marker='s')  # Change marker here
            
    if extra_points_dict:
        for point in extra_points_dict:
            # extra_node = Circle(point, radius=0.2, edgecolor='b', facecolor='none')
            plt.text(extra_points_dict[point][0], extra_points_dict[point][1], point, fontsize=12, color='red')
            ax.scatter(*extra_points_dict[point], s=200, edgecolor='b', facecolor='none', marker='s')  # Change marker here
            
    if convex_discs:
        for convex_i in range(1, convex_discs[0]+1):
            cover = Circle(convex_discs[1], radius=convex_i*disc_size, edgecolor='b', facecolor='none')
            ax.add_patch(cover)
    if convex_discs2:
        for convex_i in range(1, convex_discs2[0]+1):
            cover = Circle(convex_discs2[1], radius=convex_i*disc_size, edgecolor='y', facecolor='none')
            ax.add_patch(cover)
    
    #draw edges
    
    nx.draw_networkx_edges(G=graph, pos=node_pos, width=1, node_size=750)
    if show_edge_label:
        nx.draw_networkx_edge_labels(G=graph, pos=node_pos, edge_labels=length, font_size=8) #12
    if chosen_edges:
        nx.draw_networkx_edges(G=graph, pos=node_pos, edgelist=chosen_edges, edge_color="#2CBAFA", width=3, node_size=750)
    if chosen_edges2:
        nx.draw_networkx_edges(G=graph, pos=node_pos, edgelist=chosen_edges2, edge_color="#2CFFFA", width=3, node_size=750)
            
    ax.set_title(title, fontsize=15)   
    ax.set_aspect(1)
    ax.axis('equal') 
    ax.tick_params(left=True, bottom=True, labelleft=True, labelbottom=True, labelsize=20)
    plt.axis('on')
    fig.tight_layout()
    plt.show()

def import_tsplib_file(filepath):
    """
    Import node coordinates from a TSPLIB .tsp file.
    
    :param filepath: Path to the .tsp file.
    :return: List of (x, y) tuples.
    """
    coords = []
    with open(filepath, 'r') as f:
        in_node_section = False
        for line in f:
            line = line.strip()
            if line.startswith("NODE_COORD_SECTION"):
                in_node_section = True
                continue
            if line.startswith("EOF") or line.startswith("-1"):
                break
            if in_node_section:
                parts = line.split()
                if len(parts) >= 3:
                    coords.append((float(parts[1]), float(parts[2])))
    return coords


def write_tsplib_graph(G, name="network", filepath="output.tsp"):
    """
    Export the graph coordinates to a TSPLIB compatible file.
            
    :param G: NetworkX Graph object.
    :param name: Name of the problem instance.
    :param filepath: Output path for the .tsp file.
    """
    with open(filepath, 'w') as f:
        f.write(f"NAME : {name}\n")
        f.write("TYPE : TSP\n")
        f.write(f"DIMENSION : {G.number_of_nodes()}\n")
        f.write("EDGE_WEIGHT_TYPE : EUC_2D\n")
        f.write("NODE_COORD_SECTION\n")
        for i, node in enumerate(G.nodes(), 1):
            x, y = G.nodes[node]['pos']
            f.write(f"{i} {x:.5e} {y:.5e}\n")
        f.write("EOF\n")
