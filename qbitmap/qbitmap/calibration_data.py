from typing import Dict, List, Tuple, Optional

import numpy as np
import cirq
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.colors as colors

from qbitmap import utils
from qbitmap import state_space

PROJECT_ID = None
PROCESSOR_ID = None
GATE_SET = None


def fermi_engine_sampler():
    """Construct a default `QuantumEngineSampler` using fermilab's token.

    Returns:
        QuantumEngineSampler object that behaves like a cirq.Simulator()
    """
    engine = cirq.google.Engine(project_id=PROJECT_ID)
    # proto_version=PROTO_VERSION)
    # exposes a `run_sweep` method and inherits `run`, `sample` from `Sampler`
    return engine.sampler(processor_id=PROCESSOR_ID, gate_set=GATE_SET)


def fermi_calibration_data():
    """Query current calibration data via fermilab's token.

    Returns:
        cirq.google.engine.Calibration object. For most purposes this object
            acts as a non-serializable dictionary.
    """
    engine = cirq.google.Engine(project_id=PROJECT_ID)
    processor = engine.get_processor(processor_id=PROCESSOR_ID)

    # Get the latest calibration metrics.
    return processor.get_current_calibration()


def truncate_colormap(cmap, minval=0.0, maxval=1.0, n=100):
    """'Renormalize' a color map to use only a subset of its range.

    The new color map will use colors only between (minval, maxval).
    """
    new_cmap = colors.LinearSegmentedColormap.from_list(
        'trunc({n},{a:.2f},{b:.2f})'.format(n=cmap.name, a=minval, b=maxval),
        cmap(np.linspace(minval, maxval, n)))
    return new_cmap


class CalibrationWrapper:
    """Container for graphs whose weights correspond to calibration metrics.

    Args:
        calibration_dict: A Dict or `cirq.Calibration` object containing the
            calibration data to incorporate into a graph
        qubits: The set of `cirq.GridQubit` objects that you intend to use. This
            is useful for testing consistency. Generally you should just pass
            in the result of device.qubits for proper behavior.
        metric_1q: What calibration key to weight vertices by.
        metric_2q: What calibration key to weight edges by.

    """

    def __init__(self, calibration_dct, qubits=None, metric_1q=None, metric_2q=None):

        if qubits is None:
            qubits = cirq.google.devices.Sycamore23.qubits
        self.qubits = qubits

        if metric_1q is None:
            metric_1q = "single_qubit_rb_average_error_per_gate"
        if metric_2q is None:
            metric_2q = "two_qubit_parallel_sqrt_iswap_gate_xeb_average_error_per_cycle"

        # Defer to calibration data as to whether a given edge exists.
        self.qubit_graph = self._trusted_connectivity_graph(
            qubits, calibration_dct)

        # Defer to `qubits` as to which nodes to keep.
        scrubbed = self.scrub_calibration_dct(calibration_dct)

        # Generate graphs
        self.noise_graph = self._make_noise_graph(scrubbed, metric_1q, metric_2q)

    def _trusted_connectivity_graph(self, qubits, dct):
        """Construct a connectivity based on input calibration data.

        This function assumes that the connectivity implied by the calibration
        data takes precedence over any hard-coded set of qubits, in case there is
        conflict between the implied nearest-neighbor topology of `qubits` and
        the implied connectivity of the calibration dict.

        Logically, we require set(`qubits`) ⊇ set(calibration qubits)

        Args:
            qubits: A set of cirq.GridQubits with implied nearest-neighbor
                connectivity
            dct: A nested calibration dictionary.

        Returns:
            nx.Graph containing edges and nodes implied by calibration data.
        """

        qubit_coord_dct = dict(zip(qubits, range(len(qubits))))

        # Add nodes with dict pointing to qubit (row, col)
        nodes = [(i, {
            "row": q.row,
            "col": q.col
        }) for i, q in enumerate(qubits)]
        edges = []
        for (q0, q1) in dct.get(
                'two_qubit_parallel_sqrt_iswap_gate_xeb_average_error_per_cycle'
        ).keys():
            node0 = qubit_coord_dct.get(q0)
            node1 = qubit_coord_dct.get(q1)
            if node0 is None or node1 is None:
                continue
            edges.append(tuple(sorted([node0, node1])))

        if not any(edges):
            raise ValueError("Did not find requested edges in calibration data "
                             " - Make sure you input the correct qubits.")
        G = nx.Graph()
        G.add_nodes_from(nodes)
        G.add_edges_from(edges)
        return G

    def scrub_calibration_dct(self, dct):
        """Strike any unrecognized qubits provided in calibration data.

        This will also do the following:
            1. flatten the calibration metrics
            2.
        Args:
            dct: A calibration data dictionary like:

                Dict(str, Dict(Iterable(qubits), Iterable(Float)))

        Returns:
            A calibration data dictionary where subdictionaries are either:

                Dict(Tuple[int], Float) - for 1-qubit metric
                Dict(Iterable(Tuple[int]), Float) - for 2-qubit metric
        """
        scrubbed = {}
        allowed_qubits = set(self.qubits)

        for metric_key, qubit_map in dct.items():
            temp = {}
            for qubit_iter, metric_iter in qubit_map.items():
                if not set(qubit_iter).issubset(allowed_qubits):
                    continue
                new_val = metric_iter[0]
                if len(qubit_iter) == 1:
                    qubit = qubit_iter[0]
                    new_key = (qubit.row, qubit.col)
                elif len(qubit_iter) == 2:
                    qubit0, qubit1 = qubit_iter
                    new_key = tuple([(q.row, q.col) for q in qubit_iter])
                temp[new_key] = new_val
            scrubbed[metric_key] = temp

        return scrubbed

    def _make_noise_graph(self, calibration_dict, metric_1q, metric_2q):
        """Construct a noise graph for specific choices of calibration metrics.

        This will combine the following two metrics into a single weighted
        undirected graph:

            - two_qubit_parallel_sqrt_iswap_gate_xeb_average_error_per_cycle
            - single_qubit_rb_average_error_per_gate

        returns:
            calibration_graphs: nx.Graph containing weights corresponding to
                the single- and two-qubit metrics and edges/nodes corresponding
                to the device topology implied by calibration data.

        """

        G = nx.Graph()
        noise_1q = calibration_dict.get(metric_1q)
        noise_2q = calibration_dict.get(metric_2q)
        readout_p10 = calibration_dict.get("single_qubit_p00_error")
        readout_p01 = calibration_dict.get("single_qubit_p11_error")
        # Assign the weights to edges and nodes
        for qubit_xy, metric in noise_1q.items():
            p10 = readout_p10.get(qubit_xy)
            p01 = readout_p01.get(qubit_xy)
            readout_error = 0.5 * (p01 + p10)
            G.add_node(qubit_xy, weight=metric, readout_error=readout_error)
        for qubit_xy_lst, metric in noise_2q.items():
            G.add_edge(*qubit_xy_lst, weight=metric)

        return G

    def plot_noise_graph(self, ax=None, with_labels=False):
        """Plot a graph of the qubit and entangler noise.

        Args:
            ax: (plt.Axes) matplotlib ax to be plotted on
        """
        if ax is None:
            _, ax = plt.subplots(figsize=(8, 8))

        # Set up a regular grid for plotting
        nodes = self.noise_graph.nodes()
        coords = [(node[1], node[0]) for node in nodes]
        fixed_positions = dict(zip(nodes, coords))
        pos = nx.spring_layout(self.noise_graph,
                               pos=fixed_positions,
                               fixed=nodes)

        # PLOTTING KWARGS
        node_font_size = 14
        edge_font_size = 14
        edge_width = 18
        node_size = 1000

        # Assign (custom) labels to nodes. If you want to plot a different choice of
        # 1-qubit fidelity change the key controlling this attribute getter.
        node_labels = nx.get_node_attributes(self.noise_graph, 'weight')
        temp = utils.minmax_scaler(np.asarray(list(node_labels.values())))
        node_labels = dict(zip(node_labels.keys(), temp))

        # node_labels_fmt = {k: f"{k}\n{v:3.2f}" for k, v in node_labels.items()}
        node_labels_fmt = {k: f"{k}" for k, v in node_labels.items()}

        node_cmap = truncate_colormap(plt.get_cmap('hot'),
                                      minval=0.0,
                                      maxval=0.7)
        node_colors = np.array(
            [node_cmap(node_labels.get(node)) for node in nodes])
        # Assign edge labels in a similar fashion.
        edge_labels = nx.get_edge_attributes(self.noise_graph, 'weight')
        temp = utils.minmax_scaler(np.asarray(list(edge_labels.values())))
        edge_labels = dict(zip(edge_labels.keys(), temp))

        edges = self.noise_graph.edges()
        edge_widths = [edge_width for edge in edges]
        edge_cmap = truncate_colormap(plt.get_cmap('hot'),
                                      minval=0.0,
                                      maxval=0.7)
        edge_colors = [edge_cmap(edge_labels.get(edge)) for edge in edges]

        if with_labels:
            edge_labels_fmt = {(x[0], x[1]): f"{edge_labels[x]:3.2f}"
                               for x in edge_labels}
            nx.draw_networkx_edge_labels(self.noise_graph,
                                         pos=pos,
                                         edge_labels=edge_labels_fmt,
                                         font_size=edge_font_size,
                                         ax=ax)

        node_artists = nx.draw_networkx_nodes(self.noise_graph,
                                              pos=pos,
                                              node_size=node_size,
                                              node_color=node_colors,
                                              ax=ax)
        node_label_artists = nx.draw_networkx_labels(self.noise_graph,
                                              pos=pos,
                                              labels=node_labels_fmt,
                                              font_color='w',
                                              font_size=node_font_size,
                                              ax=ax)
        edge_artists = nx.draw_networkx_edges(self.noise_graph,
                                              pos=pos,
                                              width=edge_widths,
                                              edge_color=edge_colors,
                                              ax=ax)
        # nx.draw(self.noise_graph,
        #         pos=pos,
        #         labels=node_labels_fmt,
        #         node_size=node_size,
        #         with_labels=True,
        #         font_color='w',
        #         font_size=node_font_size,
        #         node_color=node_colors,
        #         width=edge_widths,
        #         edge_color=edge_colors,
        #         cmap=node_cmap,
        #         ax=ax)

        xmin, xmax = ax.get_xlim()
        ymin, ymax = ax.get_ylim()
        buffer = .1  # buffer in axis size units
        xsz = xmax - xmin
        ysz = ymax - ymin
        ax.set_xlim(xmin - buffer * xsz, xmax + buffer * xsz)
        ax.set_ylim(ymin - buffer * ysz, ymax + buffer * ysz)
        return [node_artists, edge_artists, node_label_artists]

    def plot_circuit_overlay(self, circuit: cirq.Circuit, ax=None):
        """Overlay a circuit on the noise graph in some consistent way."""

        if ax is None:
            _, ax = plt.subplots(figsize=(8, 8))

        qubit_order = state_space.construct_qubit_order_line_circuit(
            circuit, self.noise_graph)
        nodes = [(q.row, q.col) for q in qubit_order]
        edges = [(nodes[i], nodes[i + 1]) for i in range(len(nodes) - 1)]
        subgraph = nx.Graph()
        subgraph.add_nodes_from(nodes)
        subgraph.add_edges_from(edges)
        # Overlay a shaded graph that _doesn't_ contain the edges and nodes
        # in `circuit`
        # temp = self.noise_graph.copy()
        # for node in coords:
        #     temp.remove_node(node)
        # for edge in edges:
        #     temp.remove_edge(*edge)

        artists = self.plot_noise_graph(ax=ax)
        coords = [(node[1], node[0]) for node in nodes]
        fixed_positions = dict(zip(nodes, coords))
        pos = nx.spring_layout(subgraph, pos=fixed_positions, fixed=nodes)

        # nx.draw(subgraph,
        #         pos=pos,
        #         alpha=0.65,
        #         node_size=1000,
        #         width=18,
        #         node_color='w',
        #         edge_color='w',
        #         ax=ax)
        overlay_node_artists = nx.draw_networkx_nodes(subgraph,
                                                      pos=pos,
                                                      alpha=0.65,
                                                      node_size=1000,
                                                      node_color='w',
                                                      ax=ax)
        overlay_edge_artists = nx.draw_networkx_edges(subgraph,
                                                      pos=pos,
                                                      alpha=0.65,
                                                      width=18,
                                                      edge_color='w',
                                                      ax=ax)

        return artists + [overlay_node_artists, overlay_edge_artists]
