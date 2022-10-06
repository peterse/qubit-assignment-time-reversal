import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.colors as colors
import matplotlib as mpl


def make_nn_connected_from_inputs(width, height, noise_1q, noise_2q):
    """Construct a graph containing information on the qubit/gate noise.

        The graph will be a nearest-neighbor connected grid of dimensions
        (width, height). See `noisemaps/sample_noisemap` for a sample of the
        dictionary inputs.

        Args:
            width (int): Width of the 2D grid
            height (int): Height of the 2D grid
            noise_1q (dict): Dictionary where keys represent type of noise, and
                values are lists of noise parameters whose index corresponds to
                the qubit position in the (flattened) grid.
            noise_2q (dict): Nested dictionary where keys represent type of noise,
                and values are dictionaries mapping edge coordinates `(i, j)` to
                noise parameter values, where `i`, `j` correspond to the qubit
                indices in the (flattened) grid.

        Returns:
            G: Lattice graph containing noise parameters as node/edge attributes.
            original_coords: Positions of qubits _before_ being flattened into
                integers. This is necessary to recover relative positions of
                qubits on the lattice for plotting.
        """

    G = nx.generators.lattice.grid_2d_graph(height, width)

    # Impose a canonical numbering scheme on nodes: Qubits will be labeled from
    # left to right, top to bottom. Single qubit gate metrics will follow the same
    # convention. However we will use the original coords for plotting so save them.
    original_coords = list(G.nodes())

    # Assign the weights to edges and nodes
    for noise_type, noise_arr in noise_1q.items():
        for k, node in enumerate(G.nodes()):
            G.nodes[node].update({noise_type: noise_arr[k]})

    for noise_type, noise_dct in noise_2q.items():
        for e in G.edges():
            G.edges[e].update({noise_type: noise_dct[e]})

    return G, original_coords


def truncate_colormap(cmap, minval=0.0, maxval=1.0, n=100):
    """This will truncate a colormap that assumes a [0, 1] range.

    After truncation, the color that formerly corresponded to `0` will correspond
    to `minval`, while the color that formerly corresponded to `1` will be
    represented by `maxval`.
    """
    new_cmap = colors.LinearSegmentedColormap.from_list(
        'trunc({n},{a:.2f},{b:.2f})'.format(n=cmap.name, a=minval, b=maxval),
        cmap(np.linspace(minval, maxval, n)))
    return new_cmap


def plot_nn_graph_with_noise(G,
                             original_coords,
                             metric_1q=None,
                             metric_2q=None,
                             ax=None,
                             return_artists=False,
                             show_values=True,
                             nodescale=1,
                             edgescale=1,
                             nodefontscale=1,
                             cbar_rect=[0.88, 0.13, 0.075, 0.4],
                             label_cbar=True,
                             cbar_labelsz=16):
    """Plot a nearest-neighbor lattice graph.

    Args:
        G (nx.Graph): Graph to be plotted, containing noise parameters. Note
            that nodes are Int and edges are Tuple[int]!
        original_coords (Sequence[Tuple[int]]): Sequence of (row, col)
            coordinates describing the layout of G.nodes().
        ax: (plt.Axes) matplotlib ax to be plotted on
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 8))

    if metric_1q is None:
        metric_1q = 'single_qubit_depol'
    if metric_2q is None:
        metric_2q = 'two_qubit_depol'
    # Set up a regular grid for plotting
    scale = 0.1
    coords = []
    for (row, col) in original_coords:
        coords.append((scale * col, scale * row))

    fixed_positions = dict(zip(G.nodes(), coords))
    pos = nx.spring_layout(G, pos=fixed_positions, fixed=G.nodes())

    # First learn how large of a figure we are working with
    fig = ax.figure
    bbox = ax.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
    # Sizes in pixels
    axwidth, axheight = bbox.width * fig.dpi, bbox.height * fig.dpi

    # PLOTTING KWARGS
    # These are hard to control because networkx has its own automated
    # plotting system for graphs. Here I've tried to do my best to set up
    # scaling that results in graphs plotting reasonably regardless of
    # absolute axis size...
    node_font_size = 17 * axwidth / 600 * nodefontscale
    edge_font_size = 18 * axwidth / 600 * edgescale
    edge_width = axwidth / 20 * edgescale
    node_size = axwidth * 4.5 * nodescale

    # Assign (custom) labels to nodes. If you want to plot a different choice of
    # 1-qubit fidelity change the key controlling this attribute getter.
    node_labels = nx.get_node_attributes(G, metric_1q)
    node_labels_fmt = {k: f"{k}" for k, _ in node_labels.items()}
    if show_values:
        node_labels_fmt = {k: f"{k}\n{v:3.4f}" for k, v in node_labels.items()}

    nodevals = [node_labels.get(node) for node in G.nodes]
    rel_nodevals = [x / max(nodevals) for x in nodevals]
    ub = max(nodevals)
    node_cmap = truncate_colormap(
        plt.get_cmap('hot'),
        minval=0,
        # maxval=max(node_labels.values()) * 1.1)
        maxval=0.6,  # somehow this became a percentage.??
        n=1000)
    node_colors = np.array([node_cmap(x) for x in rel_nodevals])

    # Assign edge labels in a similar fashion.
    edge_labels = nx.get_edge_attributes(G, metric_2q)
    edgevals = [edge_labels.get(edge) for edge in G.edges]
    rel_edgevals = [x / max(edgevals) for x in edgevals]
    edge_labels_fmt = {}
    if show_values:
        edge_labels_fmt = {(x[0], x[1]): f"{edge_labels[x]:3.4f}"
                           for x in edge_labels}
    edge_widths = [edge_width for edge in G.edges()]

    edge_cmap = truncate_colormap(plt.get_cmap('hot'), minval=0, maxval=0.6)
    edge_colors = [edge_cmap(x) for x in rel_edgevals]

    node_label_artists = nx.draw_networkx_edge_labels(
        G,
        pos=pos,
        edge_labels=edge_labels_fmt,
        font_size=edge_font_size,
        font_color='w',
        ax=ax,
        bbox=dict(alpha=0),  # transparent edge labels
    )

    node_artists = nx.draw_networkx_nodes(G,
                                          pos=pos,
                                          node_size=node_size,
                                          node_color=node_colors,
                                          ax=ax)
    node_label_artists = nx.draw_networkx_labels(G,
                                                 pos=pos,
                                                 labels=node_labels_fmt,
                                                 font_color='w',
                                                 font_size=node_font_size,
                                                 ax=ax)
    edge_artists = nx.draw_networkx_edges(G,
                                          pos=pos,
                                          width=edge_widths,
                                          edge_color=edge_colors,
                                          ax=ax)
    xmin, xmax = ax.get_xlim()
    ymin, ymax = ax.get_ylim()
    buffer = .1  # buffer in axis size units
    xsz = xmax - xmin
    ysz = ymax - ymin
    ax.set_xlim(xmin - buffer * xsz, xmax + buffer * xsz)
    ax.set_ylim(ymin - buffer * ysz, ymax + buffer * ysz)
    ax.axis('off')

    # Create two axes for the colorbar on the same place.
    cax12 = plt.axes(cbar_rect)
    cax3 = cax12.twiny()
    # plot first colorbar
    nodemin, nodemax = min(nodevals), max(nodevals)
    edgemin, edgemax = min(edgevals), max(edgevals)
    norm = mpl.colors.Normalize(vmin=0, vmax=nodemax)
    sm = plt.cm.ScalarMappable(cmap=node_cmap, norm=norm)
    sm.set_array([])

    label = None
    if label_cbar:
        label = "Single-qubit gate average RB error"
    cbar12 = plt.colorbar(sm,
                          cax=cax12,
                          ticks=np.linspace(0, nodemax, 8),
                          label=label,
                          orientation='horizontal')

    # plt.colorbar(sm)
    def sci_notation(number, sig_fig=2):
        ret_string = "{0:.{1:d}e}".format(number, sig_fig)
        a, b = ret_string.split("e")
        b = int(b)
        return a + r"E" + str(b) + r""

    yticklabel_sz = cbar_labelsz - 4
    # move ticks and label of colorbar to the left
    cax12.xaxis.set_ticks_position('bottom')
    cax12.xaxis.set_label_position('bottom')
    left_unformatted = cax12.get_xticks()
    left_fmt = [sci_notation(x) for x in left_unformatted]
    left_fmt[0] = r"$0$"
    cax12.set_xticklabels(left_fmt, rotation=45, ha="right")
    cax12.xaxis.label.set_size(cbar_labelsz)
    cax12.tick_params(
        axis='x',
        which='major',
        labelsize=yticklabel_sz,
    )

    # adjust limits of right axis to match data range of 3rd plot
    cax3.set_xticks(np.arange(8))
    cax3.set_xticklabels(np.linspace(0, edgemax, 8))
    # cax3.set_ylim(0, edgemax)
    right_unformatted = np.linspace(0, edgemax, 8)
    right_fmt = [sci_notation(x) for x in right_unformatted]
    right_fmt[0] = r"$0$"
    cax3.set_xticklabels(right_fmt, rotation=45, ha="left")
    if label_cbar:
        cax3.set_xlabel(r"ISWAP$^{1/2}$ average XEB error per cycle",
                        size=cbar_labelsz)
    cax3.tick_params(axis='x', which='major', labelsize=yticklabel_sz)

    if return_artists:
        return [node_artists, edge_artists, node_label_artists]

    return ax, cax12, cax3


def plot_circuit_overlay(G, nodes, ax=None):
    """Overlay a circuit on the noise graph in some consistent way."""

    if ax is None:
        _, ax = plt.subplots(figsize=(8, 8))
    edges = [(nodes[i], nodes[i + 1]) for i in range(len(nodes) - 1)]
    subgraph = nx.Graph()
    subgraph.add_nodes_from(nodes)
    subgraph.add_edges_from(edges)

    artists = plot_nn_graph_with_noise(G,
                                       G.nodes,
                                       metric_1q="weight",
                                       metric_2q="weight",
                                       ax=ax,
                                       return_artists=True)

    # unlike on sycamore, no transpose needed for this lattice coord system
    scale = 0.1
    coords = []
    for (row, col) in nodes:
        coords.append((scale * col, scale * row))
    fixed_positions = dict(zip(nodes, coords))

    pos = nx.spring_layout(subgraph, pos=fixed_positions, fixed=nodes)
    overlay_node_artists = nx.draw_networkx_nodes(subgraph,
                                                  pos=pos,
                                                  alpha=0.65,
                                                  node_size=3000,
                                                  node_color='w',
                                                  ax=ax)
    overlay_edge_artists = nx.draw_networkx_edges(subgraph,
                                                  pos=pos,
                                                  alpha=0.65,
                                                  width=22,
                                                  edge_color='w',
                                                  ax=ax)

    return artists + [overlay_node_artists, overlay_edge_artists]
