"""state_space.py - Updating elements of the state space over circuits."""
from typing import Tuple, Sequence, Optional
import collections

import cirq
import networkx as nx
import numpy as np

from qbitmap import utils


def _validate_qubits_simple_path(qubit_order: Tuple[cirq.GridQubit],
                                 graph: nx.Graph) -> bool:
    """Validate that a frozen qubit order respects simple path topology."""

    coords = [(q.row, q.col) for q in qubit_order]
    return _validate_coords_simple_path(coords, graph)


def _validate_coords_simple_path(coords: Tuple[Tuple],
                                 graph: nx.Graph) -> bool:
    """Validate that a coordinate order respects simple path topology."""

    implied_edges = [(coords[i], coords[i + 1])
                     for i in range(len(coords) - 1)]
    subgraph = nx.Graph()
    subgraph.add_nodes_from(coords)
    subgraph.add_edges_from(implied_edges)
    return nx.is_simple_path(subgraph, coords)


def construct_qubit_order_line_circuit(
        circuit: cirq.Circuit, graph: nx.Graph) -> Tuple[Sequence[Tuple]]:
    """Given a line circuit and a topology graph, compute an ordering over qubits

    The promise is that `circuit` has line topology and is composed of
    `cirq.GridQubit` qubits.

    Warning: This function is not self-consistent over multiple calls. You can
    manipulate the ordering of qubits in `circuit` but you cannot ensure that
    the function applied to two different circuits sharing a subsequence of
    ordered qubits returns the same order for both circuits.

    As a result, this function is useful for updating a circuit but not for
    specifying the order of two `similar` circuits.

    """
    circuit_is_valid_subgraph = utils.validate_circuit_as_subgraph(
        circuit, graph)
    if not circuit_is_valid_subgraph:
        raise ValueError(
            "Input circuit contains qubits or entangling operations "
            "not supported by the corresponding graph.")

    # If we are not provided a qubit order, we need to calculate the 'endpoints'
    # of our line circuit to infer the ordering of the qubits along a line
    # A reliable way to 'reverse' the line of qubits is to defer to the
    # topology graph: Construct a subgraph over qubits

    # WARNING: this won't work if there are entanglers missing...
    entangling_operations = circuit.findall_operations(
        lambda x: x.gate.num_qubits() == 2)
    qubit_pairs = [t[1].qubits for t in entangling_operations]
    # The details of the outer sort are irrelevant as long as its consistently
    # defined over lists of tuples. The inner tuple MUST NOT BE SORTED to
    # maintain correspondence to a real qubit grid.
    undirected_edges = [
        tuple(sorted([(q.row, q.col) for q in qubits]))
        for qubits in qubit_pairs
    ]
    undirected_edges = list(set(undirected_edges))

    # This is ugly and inefficient, sorry.
    sources = [t[0] for t in undirected_edges]
    sinks = [t[1] for t in undirected_edges]
    all_hits = sources + sinks
    # there are precisely two elements that occur once in all_hits
    counter = collections.Counter(all_hits)
    extrema = [k for k, v in counter.items() if v == 1]
    # Construct one of two possible orderings on the qubits implied by
    # the connectivity of the graph and the given qubits (nodes)
    # FIXME: This whole process is O(n^2)
    candidate_node = extrema[0]
    qubit_order = [candidate_node]
    temp = [x for x in undirected_edges]
    while len(temp) > 0:
        candidate_edge = next(x for x in temp if candidate_node in x)
        temp.remove(candidate_edge)
        candidate_node = (set(candidate_edge) - set([candidate_node])).pop()
        qubit_order.append(candidate_node)

    return [cirq.GridQubit(*coords) for coords in qubit_order]


def update_linear_path(
        nodes: Tuple[Tuple],
        graph: nx.Graph,
        num_new_max: int = 2,
        swap_dist: int = 0,
        transition_probs: Sequence[int] = None,
        seed: Optional[int] = None,
) -> Tuple[Tuple]:
    """Update a line connectivity circuit according to neighborhood definition.

    Updates are defined according to the following rule:
        1. Sample x ~ `transition_probs`
        2.
            if x == 0: Perform a permutation on the qubits of `circuit`
                according to `swap_dist`.
            if x == 1: regenerate qubits of `circuit` with at most `num_new_max`
                new qubits introduced. Does not constrain differences in edges!
            if x == 2: NotImplemented

    Args:
        nodes: A nested tuple defined over a valid subset of coordinates from `graph`.
        graph: A graph containing circuit topology.
        num_new_max: The maximum number of new qubits to include in an update.
        swap_dist: The maximum number of (implemented) swaps to allow. This does
            not affect sampling a circuit whose qubits are strictly reversed.
        transition_probs: The pmf over which transitions are sampled

    Returns:
        An updated nested Tuple in the neighborhood of `path`

    """
    path_is_valid = _validate_coords_simple_path(nodes, graph)
    if not path_is_valid:
        raise ValueError("Input path is not a valid simple path on `graph`")

    if transition_probs is None:
        # This is a tunable behavior...
        transition_probs = [0.3, 0.7]

    # Be careful: Seed is for debugging mostly; fixing a seed during annealing
    # will ruin the state sampling!
    if seed:
        np.random.seed(seed)
    # Sample a state update
    x = np.random.choice(range(len(transition_probs)), p=transition_probs)
    n = len(nodes)

    if x == 0:
        return tuple(reversed(nodes))

    if x == 1:
        # Suggest a new circuit with at most `num_new_max` distinct qubits
        edges = [(nodes[i], nodes[i + 1]) for i in range(len(nodes) - 1)]

        # select a contiguous subsequence to keep
        n_keep = n - num_new_max
        keep_start = np.random.choice(num_new_max +
                                      1)  # off-by-one for `choice`
        keep_idx = np.arange(keep_start, keep_start + n_keep)
        new_nodes = [nodes[i] for i in keep_idx]
        new_edges = [edges[i] for i in keep_idx[:-1]]

        # Accumulate nearest neighbors on either end iteratively
        counter = 0
        while len(new_nodes) < n:
            pointer = np.random.choice(2) * -1
            node = new_nodes[pointer]
            edge = new_edges[pointer]
            # don't double back on yourself
            prohibited_direction = (edge[0][0] - edge[1][0],
                                    edge[0][1] - edge[1][1])
            allowed_directions = [(1, 0), (-1, 0), (0, -1), (0, 1)]
            allowed_directions.remove(prohibited_direction)
            # check for out of bounds
            for direction in allowed_directions:
                attempt = (node[0] + direction[0], node[1] + direction[1])
                if attempt not in graph.nodes:
                    allowed_directions.remove(direction)

            direction = allowed_directions[np.random.choice(
                len(allowed_directions))]
            suggested_node = (node[0] + direction[0], node[1] + direction[1])
            suggested_edge = (node, suggested_node)
            # Remember: nx Graphs index undirected edges independently of order!
            if suggested_node in graph.nodes and suggested_edge in graph.edges and suggested_node not in new_nodes:
                if pointer == 0:
                    new_nodes = [suggested_node] + new_nodes
                    new_edges = [(suggested_node, node)] + new_edges
                else:
                    new_nodes.append(suggested_node)
                    new_edges.append(suggested_edge)
            counter += 1
            if counter > 20:
                # It is possible to get stuck in a corner for this topology...
                print("Stuck in a corner!")
                keep_start = np.random.choice(num_new_max + 1)
                keep_idx = np.arange(keep_start, keep_start + n_keep)
                new_nodes = [nodes[i] for i in keep_idx]
                new_edges = [edges[i] for i in keep_idx[:-1]]
                # import pdb; pdb.set_trace()
                counter = 0
                # raise ValueError
        return tuple(map(tuple, new_nodes))


def update_linear_circuit(
        circuit: cirq.Circuit,
        graph: nx.Graph,
        num_new_max: int = 2,
        swap_dist: int = 0,
        transition_probs: Sequence[int] = None,
        qubit_order: Tuple[cirq.GridQubit] = None,
        seed: Optional[int] = None,
) -> cirq.Circuit:
    """Update a line connectivity circuit according to neighborhood definition.

    Updates are defined according to the following rule:
        1. Sample x ~ `transition_probs`
        2.
            if x == 0: Perform a permutation on the qubits of `circuit`
                according to `swap_dist`.
            if x == 1: regenerate qubits of `circuit` with at most `num_new_max`
                new qubits introduced. Does not constrain differences in edges!
            if x == 2: NotImplemented

    Args:
        circuit: A circuit defined over a valid subset of qubits from `graph`.
        graph: A graph containing circuit topology.
        num_new_max: The maximum number of new qubits to include in an update.
        swap_dist: The maximum number of (implemented) swaps to allow. This does
            not affect sampling a circuit whose qubits are strictly reversed.
        transition_probs: The pmf over which transitions are sampled
        qubit_order: If you know the ordering of the qubits in `circuit` that
            respects simple path topology then it can be used to more efficiently
            sample from the state space for neighborhood updates.

    Returns:
        An updated `cirq.Circuit` that is in the Neighborhood of `circuit`.

    """
    circuit_is_valid_subgraph = utils.validate_circuit_as_subgraph(
        circuit, graph)
    if not circuit_is_valid_subgraph:
        raise ValueError(
            "Input circuit contains qubits or entangling operations "
            "not supported by the corresponding graph.")

    if transition_probs is None:
        # This is a tunable behavior...
        transition_probs = [0.3, 0.7]

    if qubit_order and not _validate_qubits_simple_path(qubit_order, graph):
        raise ValueError(
            "Input `qubit_order` does not respect line connectivity")

    # Sample a state update
    if not qubit_order:
        qubit_order = construct_qubit_order_line_circuit(circuit, graph)

    nodes = [(q.row, q.col) for q in qubit_order]
    new_nodes = update_linear_path(nodes=nodes,
                                   graph=graph,
                                   num_new_max=num_new_max,
                                   swap_dist=swap_dist,
                                   transition_probs=transition_probs,
                                   seed=seed)
    new_qubits = [cirq.GridQubit(*coords) for coords in new_nodes]
    qubit_map = dict(zip(qubit_order, new_qubits))
    new_circuit = circuit.transform_qubits(qubit_map)
    return new_circuit


def snake_iterate_rect(start, ncols, nrows):
    """Iterate an (nrows, ncols) shape along a snaking path starting from `start`.

    This has the convention that nodes will be ordered starting from `start`
    first increasing in the direction that rowcount increases, then increasing
    in the direction that rowcount decreases on the next column in the direction
    that column count increases. For example, the ordering for a 3x2 grid is

                    0 | 5
                    --|--
                    1 | 4
                    --|--
                    2 | 3

    A 2x2 grid will be ordered in the counter-clockwise direction starting from the
    top left (in absolute grid coordinates) corner.

    A 1xm line will be ordered in increasing index from the start index.

    """
    out = []
    direction = 1
    for i in range(0, ncols):
        direction = -1 * direction
        # Alternate iterating top-to-bottom or bottom-to-top
        if direction > 0:
            steps = range(0, nrows, 1)
        else:
            steps = range(nrows - 1, -1, -1)
        for j in steps:
            out.append((start[0] + j, start[1] + i))
    return out


def all_rectangles(graph, nrows, ncols):
    """Return an iterator over specified rectangles on the input grid graph.

    This will return a generator over all (nrows, ncols) and (ncols, nrows)
    rectangular subgraphs on `graph`. Note that this returns unique _sets_ of
    qubits (since the `snake_iterate_rect` always assumes it starts from a
    "corner" of the qubit set it iterates over). There will generally be
    reflections and permutations of the returned paths that will also provide
    a different _assignment_ to the unique qubit set.
    """
    for node in graph.nodes:
        # Construct the (nrows, ncols) graph that we _want_
        attempt = snake_iterate_rect(node, ncols, nrows)
        x = graph.subgraph(attempt)

        # Checking that we got all the desired nodes is sufficient
        # to determine that we attempted a valid rectangle
        if len(x.nodes) == len(attempt):
            yield attempt
        # Repeat the above for a 90 degree rotation of the desired subgrid
        attempt = snake_iterate_rect(node, nrows, ncols)
        x = graph.subgraph(attempt)
        if len(x.nodes) == len(attempt):
            yield attempt
