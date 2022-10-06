from typing import Sequence, Optional, Iterable, Callable
import cirq
import cirq_google as cg
from cirq.experiments.random_quantum_circuit_generation import _single_qubit_gates_arg_to_factory, _two_qubit_layer
from cirq.experiments import random_rotations_between_grid_interaction_layers_circuit
import numpy as np


def loschmidt_circuit(circuit_generator, *args, measure="m", **kwargs):
    """Automate construction of a circuit, given a circuit generator.

    All required args and kwargs for `circuit_generator` must be provided.

    Args:
        circuit_generator: Some function consuming *args, **kwargs and returning
            a `cirq.Circuit`
        measure: Optional string. If None is provided, the output loschmidt
            fidelity circuit will not have measurements
    Note: This potentially scrambles qubit order in the measurement compared to
    any enforced ordering in the qubit constructor, but this is perfectly fine
    for the purposes of loschmidt echo fidelity.
    """
    circuit = circuit_generator(*args, **kwargs, measure=None)
    circuit += cirq.inverse(circuit)
    if measure:
        circuit += cirq.measure(*circuit.all_qubits(), key=measure)
    return circuit


def hea_circuit_line_topology(ordered_qubits, n_layers=1, angles=None, twoq_gate=None, measure: Optional[str] = None):
    """Generate a Hardware Efficient Ansatz circuit on a line.

    Args:
        ordered_qubits: Qubits that are ordered according to their position on
            a line (such that the i-th and (i+1)-th qubit support entangling op)
        n_layers: How many repetitiosn of the HEA to run.
        angles: shape `(n_qubits, 2*(n_layers + 1))` array of 1-qubit gate angles
        twoq_gate: The entangling gate to use. This should be a function that
            takes in two qubits and returns a sequence or generator over
            `cirq.Operation` objects.
        measure: A string indicating the `measure_each` key if measurement should be done

    """
    if twoq_gate is None:
        twoq_gate = cirq.ISWAP ** 0.5

    n_qubits = len(ordered_qubits)
    if angles is None:
        angles = np.random.random(size=(n_qubits, 2 * (n_layers + 1)))

    circuit = cirq.Circuit()
    for i in range(n_layers):
        circuit.append([cirq.ry(angles[j, i*2]).on(qj) for j, qj in enumerate(ordered_qubits)], strategy=cirq.InsertStrategy.NEW_THEN_INLINE)
        circuit.append([cirq.rz(angles[j, i*2 + 1]).on(qj) for j, qj in enumerate(ordered_qubits)], strategy=cirq.InsertStrategy.NEW_THEN_INLINE)
        circuit.append([twoq_gate(ordered_qubits[j], ordered_qubits[j+1]) for j in range(0, n_qubits - 1, 2)], strategy=cirq.InsertStrategy.INLINE)
        circuit.append([twoq_gate(ordered_qubits[j], ordered_qubits[j+1]) for j in range(1, n_qubits - 1, 2)], strategy=cirq.InsertStrategy.INLINE)
    circuit.append([cirq.ry(angles[j, -2]).on(qj) for j, qj in enumerate(ordered_qubits)], strategy=cirq.InsertStrategy.NEW_THEN_INLINE)
    circuit.append([cirq.rz(angles[j, -1]).on(qj) for j, qj in enumerate(ordered_qubits)], strategy=cirq.InsertStrategy.NEW_THEN_INLINE)

    if measure:
        cirq.measure(*ordered_qubits, key=measure)
    return circuit


def ghz_circuit_line_topology(ordered_qubits, measure: Optional[str] = None, native=True):
    """Generate a Greenberger-Horne-Zeilinger (GHZ) circuit on a line.

    Note that this has depth linear in n for nearest-neighbor line connectivity
    but can be reduced to log(n) depth for some types of grids.

    Args:
        ordered_qubits: Qubits that are ordered according to their position on
            a line (such that the i-th and (i+1)-th qubit support entangling op)
        measure: A string indicating the `measure_each` key if measurement should be done

    """
    n_qubits = len(ordered_qubits)
    circuit = cirq.Circuit(cirq.H(ordered_qubits[0]))
    for i in range(n_qubits - 1):
        circuit += cirq.CNOT(ordered_qubits[i], ordered_qubits[i+1])
    if measure:
        cirq.measure(*ordered_qubits, key=measure)
    if native:
        circuit = cg.optimized_for_sycamore(circuit, optimizer_type='sqrt_iswap')
    return circuit


def ghz_circuit_line_topology_halfdepth(ordered_qubits, measure: Optional[str] = None, native=True):
    """Generate a Greenberger-Horne-Zeilinger (GHZ) circuit on a line.

    This generator improves on the depth of `ghz_circuit_line_topology` by
    roughly a factor of two, but is still O(n) depth.

    If n is odd, the forwards depth is ceil(n/2) + 1
    If n is even, the forwards depth is n//2 + 1

    Args:
        ordered_qubits: Qubits that are ordered according to their position on
            a line (such that the i-th and (i+1)-th qubit support entangling op)
        measure: A string indicating the `measure_each` key if measurement should be done

    """
    n_qubits = len(ordered_qubits)
    start = n_qubits // 2
    circuit = cirq.Circuit()
    circuit += cirq.H(ordered_qubits[start])
    pointer_top, pointer_bot = start, start - 1

    circuit += cirq.CNOT(ordered_qubits[pointer_top], ordered_qubits[pointer_bot])
    # there is an asymmetry for n odd, which marks the termination condition
    while pointer_top > 1:
        pointer_top -= 1
        circuit += cirq.CNOT(ordered_qubits[pointer_top], ordered_qubits[pointer_top - 1])

        pointer_bot += 1
        if pointer_bot < n_qubits - 1:
            circuit += cirq.CNOT(ordered_qubits[pointer_bot], ordered_qubits[pointer_bot + 1])

    if measure:
        cirq.measure(*ordered_qubits, key=measure)
    if native:
        circuit = cg.optimized_for_sycamore(circuit, optimizer_type='sqrt_iswap')
    return circuit


def ghz_ladder_2x4(qubits, measure: Optional[str] = None, native=True):
    """Generate a GHZ circuit on a 2x4 grid connectivity set of qubits.

    This type of layout allows construction of the GHZ state on grid hardware
    with depth that is logarithmic in `len(qubits)`. If we suppose that CNOT is
    executed in depth 1, then this scheme has depth 3+1 when a line circuit would
    require depth 5+1 with nearest neighbor connectivity. On hardware this
    translates to savings of ~depth 20.

    The ordering of `qubits` is expected to be a nested list like this:

                3,0 |  3,1
                ____|_____
                2,0 |  2,1
                    |
                1,0 |  1,1
                ____|_____
                0,0 |  0,1
                    |

    where 0,0 is the element at qubits[0][0]

    TODO: This algorithm generalizes for shape patterns like the following:

    1 -> 1x2 -> 1x4 -> 2x4 -> 4x4 -> ...
    1 -> 1x2 -> 2x2 -> 2x4 -> 4x4 -> ...

    TODO: I'd love to hear from someone on hardware what the best tiling pattern
    for crosstalk would be!
    """
    # This is where the hadamard will be applied; it must be interior to the
    # rectangle
    # seed = 1

    # Pattern 1: 1-1x2-2x2-4x2
    circuit = cirq.Circuit(cirq.H(qubits[2][0]))
    circuit += cirq.CNOT(qubits[2][0], qubits[1][0])

    circuit += cirq.CNOT(qubits[2][0], qubits[2][1])
    circuit += cirq.CNOT(qubits[1][0], qubits[1][1])

    circuit += cirq.CNOT(qubits[2][0], qubits[3][0])
    circuit += cirq.CNOT(qubits[2][1], qubits[3][1])
    circuit += cirq.CNOT(qubits[1][0], qubits[0][0])
    circuit += cirq.CNOT(qubits[1][1], qubits[0][1])

    if measure:
        flattened_qubits = [x for sub in qubits for x in sub]
        circuit += cirq.measure(*flattened_qubits, key=measure)
    if native:
        circuit = cg.optimized_for_sycamore(circuit, optimizer_type='sqrt_iswap')
    return circuit


def qft_line_topology_native(qubits, gate_type='sqrt_iswap', factor=-1.,
                             nearest_neighbor=True,
                             nearest_neighbor_further_swap=False, measure=None):
    c = qft_line_topology_gen(
            qubits,
            factor,
            nearest_neighbor,
            nearest_neighbor_further_swap,
            measure=measure
        )
    cc = cg.optimized_for_sycamore(c, optimizer_type=gate_type)
    return cc


def qft_line_topology_gen(qubits, factor=1., nearest_neighbor=False,
                          nearest_neighbor_further_swap=True,
                          measure=None):
    '''
    y_k = (1 / sqrt{n}) sum_{j=0}^{n - 1} e^{factor *  2 pi i k j / n} x_j
    Note that `factor` is a sign factor. For QFT convention, `factor` = +1,
    which corresponds to inverse DFT.

    Parameters
    ----------
    qubits                  : list
        list of qubits
    factor                          : float, optional
        default: 1.
    nearest_neighbor                : {True, False (default)}, optional
        whether to use nearest neighor algorithm
    nearest_neighbor_further_swap   : {True (default), False}, optional
        when using nearest neighor algorithm, whether to add additional swap
        gate to get back to the qubit ordering as in the non-nearest neighor
        algorithm
        with further swap: [nq-1, nq-2, ..., 1, 0] (note: reversed)
        without further swap: [nq-1, 0, 1, ..., nq-2]

    Note: the resulted qubit order is reversed.
    '''
    gate_list = []

    if nearest_neighbor:
        n_qubit = len(qubits)
        # table to keep track of qubit id before swap
        qid_preswap_table = [j for j in range(n_qubit)]
        # table to track whether H gate has been applied
        H_applied_table = [False] * n_qubit

        if nearest_neighbor_further_swap:
            swap_gate_list = []

        for j in range(1, n_qubit):
            for k in range(j, 0, -1):
                p = qubits[k]
                i, q = qid_preswap_table[k - 1], qubits[k - 1]

                # apply H gate to the first-time target
                if not H_applied_table[i]:
                    gate_list.append(cirq.H.on(q))
                    H_applied_table[i] = True

                angle = factor * 2. * np.pi / (2.**(j - i + 1))
                gate_list.append(cirq.CZPowGate(exponent=angle / np.pi).on(q, p))

                # swap except the 0-1 pair
                if k != 1:
                    swap_gate = cirq.SWAP.on(q, p)
                    qid_preswap_table[k], qid_preswap_table[k - 1] = \
                        qid_preswap_table[k - 1], qid_preswap_table[k]
                    if nearest_neighbor_further_swap:
                        swap_gate_list += [swap_gate]
                    gate_list.append(swap_gate)

        gate_list.append(cirq.H.on(qubits[qid_preswap_table[n_qubit - 1]]))
        H_applied_table[n_qubit - 1] = True

        # if needed, correct the qubit order by further swap
        if nearest_neighbor_further_swap:
            for gate in swap_gate_list[::-1]:
                gate_list.append(gate)

    else:
        for i, q in enumerate(qubits):
            gate_list.append(cirq.H.on(q))
            for j, p in enumerate(qubits[i + 1:], i + 1):
                angle = factor * 2. * np.pi / (2.**(j - i + 1))
                gate_list.append(cirq.CZPowGate(exponent=angle / np.pi).on(q, p))

    backend_circuit = cirq.Circuit()
    backend_circuit.append(gate_list, strategy=cirq.InsertStrategy.EARLIEST)

    if measure:
        cirq.measure(*qubits, key=measure)
    return backend_circuit


def qft_ladder_2by2_native(qubits, gate_type='sqrt_iswap', factor=-1., measure=None):
    c = qft_2x2_topology_gen(
            qubits,
            factor,
            measure=measure
        )
    cc = cg.optimized_for_sycamore(c, optimizer_type=gate_type)
    return cc


def qft_2x2_topology_gen(qubits, factor=1., measure=None):
    """
    Compute the QFT on a 2x2 grid with reversed output order.

    Input qubits reflect a 2x2 grid topology. They should be ordered so that
    qubits at indices 0 and 2 are _not_ nearest neighbors, and indices reflect
    clockwise positions on the grid:

                  0 |  1
                ____|_____
                  3 |  2
                    |
    """
    gate_list = []
    q1, q2, q3, q4 = qubits

    def _cz_k(q0, q1, k):
        yield cirq.CZ(q0, q1) ** (factor * 1/2**(k-1))

    gate_list.append(cirq.H.on(q1))
    gate_list.append(_cz_k(q2, q1, 2))
    gate_list.append(_cz_k(q3, q1, 3))
    gate_list.append(cirq.SWAP(q1, q3))
    gate_list.append(_cz_k(q4, q3, 4))

    gate_list.append(cirq.H.on(q2))
    gate_list.append(_cz_k(q1, q2, 2))
    gate_list.append(_cz_k(q4, q2, 3))
    gate_list.append(cirq.SWAP(q1, q3))

    gate_list.append(cirq.H.on(q3))
    gate_list.append(_cz_k(q4, q3, 2))

    gate_list.append(cirq.H.on(q4))

    backend_circuit = cirq.Circuit()
    backend_circuit.append(gate_list, strategy=cirq.InsertStrategy.EARLIEST)

    if measure:
        cirq.measure(*qubits, key=measure)
    return backend_circuit


def iswap_network_line_topology(ordered_qubits, measure: Optional[str] = None, native=True):
    """Construct an iSWAP network swapping the first and last qubits in a line.

    Note that if `len(ordered_qubits) % 4 = 1`, then the state at
    qubit 0 will be exactly swapped into the register n-1. Otherwise there will
    be some power of i in the phase between |0> and |1> of the swapped qubit.

    Args:
        ordered_qubits: Qubits that are ordered according to their position on
            a line (such that the i-th and (i+1)-th qubit support entangling op)
        measure: A string indicating the `measure_each` key if measurement should be done
    """
    n_qubits = len(ordered_qubits)
    circuit = cirq.Circuit()
    for i in range(n_qubits - 1):
        if native:
            circuit += (cirq.ISWAP ** 0.5).on(ordered_qubits[i], ordered_qubits[i+1])
            circuit += (cirq.ISWAP ** 0.5).on(ordered_qubits[i], ordered_qubits[i+1])
        else:
            circuit += cirq.ISWAP(ordered_qubits[i], ordered_qubits[i+1])
    if measure:
        cirq.measure(*ordered_qubits, key=measure)
    return circuit


def random_rotations_between_line_interaction_layers_circuit(
    qubits: Iterable['cirq.GridQubit'],
    depth: int,
    two_qubit_op_factory: Callable[
        ['cirq.GridQubit', 'cirq.GridQubit', 'np.random.RandomState'], 'cirq.OP_TREE'
    ] = lambda a, b, _: cirq.CZPowGate()(a, b),
    single_qubit_gates: Sequence['cirq.Gate'] = (
        cirq.X ** 0.5,
        cirq.Y ** 0.5,
        cirq.PhasedXPowGate(phase_exponent=0.25, exponent=0.5),
    ),
    add_final_single_qubit_layer: bool = True,
    seed: cirq.RANDOM_STATE_OR_SEED_LIKE = None,
) -> cirq.Circuit:
    """Generate a random quantum circuit of a particular form.

    This is mostly adapted from cirq source code. I take no credit for existing
    code that I have copied; see:

    https://github.com/quantumlib/Cirq/blob/bc4123ef2a85c6a61667dc268554429690d2838d/cirq-core/cirq/experiments/random_quantum_circuit_generation.py

    This construction is based on the circuits used in the paper
    https://www.nature.com/articles/s41586-019-1666-5.
    The generated circuit consists of a number of "cycles", this number being
    specified by `depth`. Each cycle is actually composed of two sub-layers:
    a layer of single-qubit gates followed by a layer of two-qubit gates,
    controlled by their respective arguments, see below. The pairs of qubits
    in a given entangling layer is controlled by the `pattern` argument,
    see below.
    Args:
        qubits: The qubits to use.
        depth: The number of cycles.
        two_qubit_op_factory: A callable that returns a two-qubit operation.
            These operations will be generated with calls of the form
            `two_qubit_op_factory(q0, q1, prng)`, where `prng` is the
            pseudorandom number generator.
        single_qubit_gates: Single-qubit gates are selected randomly from this
            sequence. No qubit is acted upon by the same single-qubit gate in
            consecutive cycles. If only one choice of single-qubit gate is
            given, then this constraint is not enforced.
        add_final_single_qubit_layer: Whether to include a final layer of
            single-qubit gates after the last cycle.
        seed: A seed or random state to use for the pseudorandom number
            generator.
    """
    prng = cirq.value.parse_random_state(seed)
    qubits = list(qubits)
    coupled_qubit_pairs = [(qubits[i], qubits[i+1]) for i in range(len(qubits) - 1)]
    pattern = [
        [(qubits[j], qubits[j+1]) for j in range(1, len(qubits) - 1, 2)],
        [(qubits[j], qubits[j+1]) for j in range(0, len(qubits) - 1, 2)],
    ]

    circuit = cirq.Circuit()
    previous_single_qubit_layer = cirq.Moment()
    single_qubit_layer_factory = _single_qubit_gates_arg_to_factory(
        single_qubit_gates=single_qubit_gates, qubits=qubits, prng=prng
    )

    for i in range(depth):
        single_qubit_layer = single_qubit_layer_factory.new_layer(previous_single_qubit_layer)
        circuit += single_qubit_layer

        two_qubit_layer = _two_qubit_layer(
            coupled_qubit_pairs, two_qubit_op_factory, pattern[i % len(pattern)], prng
        )
        circuit += two_qubit_layer
        previous_single_qubit_layer = single_qubit_layer

    if add_final_single_qubit_layer:
        circuit += single_qubit_layer_factory.new_layer(previous_single_qubit_layer)

    return circuit


def create_random_line_circuit(
    qubits: Sequence[cirq.GridQubit],
    depth: int,
    twoq_gate: cirq.Gate = cirq.FSimGate(np.pi / 4, 0.0),
    measure: Optional[str] = None,
    seed: Optional[int] = None,
) -> cirq.Circuit:
    """Returns a Loschmidt echo circuit using a random unitary U.

    DISCLAIMER: This code is modified from Google's API docs. I take no credit

        https://quantumai.google/cirq/tutorials/google/echoes

    Args:
        qubits: Qubits to use.
        cycles: Depth of random rotations in the forward & reverse unitary.
        twoq_gate: Two-qubit gate to use.
        pause: Optional duration to pause for between U and U^\dagger.
        seed: Seed for circuit generation.
    """
    # Forward (U) operations.

    # Generate an extra-long random circuit before truncateing to desired depth
    tot_cycles = depth
    out = random_rotations_between_line_interaction_layers_circuit(
        qubits,
        depth=tot_cycles,
        two_qubit_op_factory=lambda a, b, _: twoq_gate.on(a, b),
        single_qubit_gates=[cirq.PhasedXPowGate(phase_exponent=p, exponent=0.5)
                            for p in np.arange(-1.0, 1.0, 0.25)],
        seed=seed
    )
    out = out[:depth]
    if measure:
        out += cirq.measure(*qubits, key=measure)

    return out


def create_random_2d_circuit(
    qubits: Sequence[cirq.GridQubit],
    depth: int,
    twoq_gate: cirq.Gate = cirq.FSimGate(np.pi / 4, 0.0),
    seed: Optional[int] = None,
    measure: Optional[bool] = None,
) -> cirq.Circuit:
    """Returns a Loschmidt echo circuit using a random unitary U.

    Args:
        qubits: Qubits to use.
        cycles: Depth of random rotations in the forward & reverse unitary.
        twoq_gate: Two-qubit gate to use.
        pause: Optional duration to pause for between U and U^\dagger.
        seed: Seed for circuit generation.
    """
    # Forward (U) operations.
    out = random_rotations_between_grid_interaction_layers_circuit(
        qubits,
        depth=depth,
        two_qubit_op_factory=lambda a, b, _: twoq_gate.on(a, b),
        pattern=cirq.experiments.GRID_STAGGERED_PATTERN,
        single_qubit_gates=[cirq.PhasedXPowGate(phase_exponent=p, exponent=0.5)
                            for p in np.arange(-1.0, 1.0, 0.25)],
        seed=seed
    )
    if measure:
        out += cirq.measure(*qubits, key=measure)
    return out
