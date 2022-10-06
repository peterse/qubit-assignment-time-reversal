import pytest

import matplotlib.pyplot as plt
import cirq
import networkx as nx

from qbitmap import state_space


DISPLAY = True
GRAPH = nx.generators.lattice.grid_2d_graph(5, 5)

INITIAL_COORDS = [
    (1, 1),
    (1, 2),
    (2, 2),
    (2, 1),
    (3, 1),
    (4, 1),
]
INITIAL_QUBITS = [cirq.GridQubit(*x) for x in INITIAL_COORDS]


def test_update_linear_circuit_reverse():
    """Test that the reverse/permutation linear circuit update works.

    FIXME: it would be nice if cirq supported random circuits with constrained
    topology, e.g.

        cirq.testing.random_circuit(..., device=FoxTrot)

    or whatever scheme they have that enforces linear topology.
    """

    ops = [cirq.H(INITIAL_QUBITS[0])] + [
        cirq.CNOT(INITIAL_QUBITS[i], INITIAL_QUBITS[i+1])
        for i in range(len(INITIAL_QUBITS) - 1)
    ]
    circuit = cirq.Circuit(*ops)

    # Only test for the "permute in place" update.
    transition_probs = [1, 0, 0]

    new_circuit = state_space.update_linear_circuit(circuit, GRAPH,
                              num_new_max=0, swap_dist=0,
                              transition_probs=transition_probs,
                              qubit_order=None
                              )
    reversed_qubits = list(reversed(INITIAL_QUBITS))
    expected_ops = [cirq.H(reversed_qubits[0])] + [
        cirq.CNOT(reversed_qubits[i], reversed_qubits[i+1])
        for i in range(len(reversed_qubits) - 1)
    ]

    assert list(new_circuit.all_operations()) == expected_ops


@pytest.mark.parametrize('num_new_max', [1, 2, 3])
def test_update_linear_circuit_new_path(num_new_max):
    """Test that fetching a new set of qubits for linear circuit update works.

    """

    ops = [cirq.H(INITIAL_QUBITS[0])] + [
        cirq.CNOT(INITIAL_QUBITS[i], INITIAL_QUBITS[i+1])
        for i in range(len(INITIAL_QUBITS) - 1)
    ]
    circuit = cirq.Circuit(*ops)

    # Only test for the "permute in place" update.
    transition_probs = [0, 1, 0]
    new_circuit = state_space.update_linear_circuit(circuit, GRAPH,
                              num_new_max=3, swap_dist=0,
                              transition_probs=transition_probs,
                              )

    assert new_circuit.all_qubits().symmetric_difference(circuit.all_qubits()) <= num_new_max


@pytest.mark.skipif(not DISPLAY, reason="Currently only a visual check.")
def test_visualize_linear_circuit_new_path():

    ops = [cirq.H(INITIAL_QUBITS[0])] + [
        cirq.CNOT(INITIAL_QUBITS[i], INITIAL_QUBITS[i+1])
        for i in range(len(INITIAL_QUBITS) - 1)
    ]
    old_circuit = cirq.Circuit(*ops)

    new_circuit = state_space.update_linear_circuit(old_circuit, GRAPH,
                              num_new_max=2, swap_dist=0,
                              transition_probs=[0, 1, 0],
                              seed=294592
                              )
    _, axes = plt.subplots(1, 2, figsize=(8, 4))

    for i, circuit in enumerate([old_circuit, new_circuit]):
        qubit_order = state_space.construct_qubit_order_line_circuit(circuit, GRAPH)
        coords = [(q.row, q.col) for q in qubit_order]
        # coords = utils.gridqubit_coords_to_nx_coords(coords, 5, 5)
        edges = [(coords[i], coords[i+1]) for i in range(len(coords) - 1)]
        print(circuit)
        print(coords)
        print(edges)
        # Funny layout default in nx
        graph_coords = [(t[1], t[0]) for t in GRAPH.nodes()]
        fixed_positions = dict(zip(GRAPH.nodes(), graph_coords))
        pos = nx.spring_layout(GRAPH, pos=fixed_positions, fixed=GRAPH.nodes())

        node_colors = ['r' if node in coords else 'k' for node in GRAPH.nodes()]
        edge_colors = []
        for edge in GRAPH.edges():
            if edge in edges or tuple(reversed(list(edge))) in edges:
                edge_colors.append('pink')
            else:
                edge_colors.append('k')
        # edge_colors = ['pink' if edge in edges else 'k' for edge in GRAPH.edges()]
        edge_widths = [3 for _ in edge_colors]
        # if i == 1:
        #     import pdb; pdb.set_trace()
        nx.draw(GRAPH,
                pos=pos,
                # labels=node_labels_fmt,
                # node_size=node_size,
                with_labels=True,
                font_color='g',
                # font_size=node_font_size,
                node_color=node_colors,
                width=edge_widths,
                edge_color=edge_colors,
                # cmap=node_cmap,
                ax=axes[i])
    plt.show()


if __name__ == "__main__":
    # test_update_linear_circuit_new_path()
    test_visualize_linear_circuit_new_path()