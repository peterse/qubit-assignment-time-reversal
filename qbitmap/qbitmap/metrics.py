"""metrics.py - Performance metrics for simulated and hardware circuits."""
import cirq
import networkx as nx
import numpy as np

from qbitmap import utils


def compute_true_fidelity(circuit, noisy_circuit, simulator=None, atol=1e-6):
    """ """
    if simulator is None:
        simulator = cirq.DensityMatrixSimulator()

    wf = circuit.final_state_vector()
    dm = simulator.simulate(noisy_circuit).final_density_matrix
    temp = wf.conj() @ dm @ wf
    if abs(temp.imag) > atol:
        raise ValueError("Fidelity error exceed tolerance")
    return temp.real


def compute_calibration_fidelity(circuit: cirq.Circuit,
                                 graph: nx.Graph,
                                 readout_error: bool = False,
                                 metric_1q="weight",
                                 metric_2q="weight",
                                 c1q=1,
                                 c2q=1) -> float:
    """Compute a simple 'fidelity' for a circuit based on calibration data.

    The simple metric we compute is essentially that of Finnigan, 2018:

        F = \prod_{i\in V_p*} (1-ϵ_i)^{n_i} \prod_{i,j \in E_p*} (1 - η_{ij})^{n_ij}

    where  (V_p*, E_p*) are the set of edges and vertices used in the input circuit,
    n_i and n_{ij} count the number of times a gate occurs on any specific vertix `i`
    or edge `ij`, and ϵ,η  are performance metrix for 1- and 2-qubit gates respectively

    If `readout_error=True`, then an additional product over

        (1 - average(p01, p10))

    for all qubits will also be included.

    Args:
        circuit: A circuit defined over a valid subset of qubits from `graph`
        graph: A graph containing calibration metrics to score the circuit on
        readout_error: Whether to account for readout error in the metric
        metric_1q, metric_2q: String identifiers for the nx.Graph property to
            pull up and compute F0 with.
        c1q, c2q: Rescaling constants to be applied to the 1- and 2-qubit
            properties pulled up from the noise Graph. This is an important
            modification! For example, `single_qubit_rb_average_error_per_gate`
            is off from the corresponding depolarizing probability for the
            twirled version of that gate - see Table 1 in the supplementary
            information of the Google quantum supremacy paper.
    Returns:
        the quantity `F` computed for this circuit.
    """

    circuit_is_valid_subgraph = utils.validate_circuit_as_subgraph(
        circuit, graph)
    if not circuit_is_valid_subgraph:
        raise ValueError(
            "Input circuit contains qubits or entangling operations "
            "not supported by the corresponding graph.")

    f = 1
    for moment in circuit:
        for op in moment.operations:
            qubits = op.qubits
            n_qubits = op.gate.num_qubits()
            if n_qubits == 1:
                node = (qubits[0].row, qubits[0].col)
                temp = 1 - c1q * graph.nodes[node].get(metric_1q)
            elif n_qubits == 2:
                edge = tuple([(q.row, q.col) for q in qubits])
                temp = 1 - c2q * graph.edges[edge].get(metric_2q)
            f *= temp

    if readout_error:
        for qubit in circuit.all_qubits():
            node = (qubit.row, qubit.col)
            f *= 1 - graph.nodes[node].get("readout_error")
    return f


def recompute_F0(circuit, template_qubits, noise_graph, paths, c1q=1, c2q=1):
    """Recompute F0 for every path in an experiment.

    This allows one to experiment with different scaling constants applied to
    the 1- and 2-qubit error metrics pulled down from the processor.

    As always, you cannot trust the ordering of a qubit to survive a round trip
    of `all_qubits` called from `cirq.Circuit`
    """

    F0_out = []
    F0_raw_out = []
    for i, v in enumerate(paths):
        # Construct the circuit on this path
        targets = [cirq.GridQubit(*x) for x in v]
        qubit_map = dict(zip(template_qubits, targets))
        mapped_forward_circuit = circuit.transform_qubits(qubit_map)

        # So just ignore the metric kwargs in this function; theres an internal
        # conversion in my calibration wrapper for the weight names on an
        # nx.Graph that I'm schlepping around for backwards compatibility reasons
        F0 = compute_calibration_fidelity(circuit=mapped_forward_circuit,
                                          graph=noise_graph,
                                          readout_error=False,
                                          c1q=c1q,
                                          c2q=c2q)
        F0_readout_err = compute_calibration_fidelity(
            circuit=mapped_forward_circuit,
            graph=noise_graph,
            readout_error=True,
            c1q=c1q,
            c2q=c2q)

        F0_out.append(F0)
        F0_raw_out.append(F0_readout_err)

    return np.asarray(F0_out), np.asarray(F0_raw_out)


def bootstrap_metric(metric, X, Y, *args, ntrials=1000):
    """Compute confidence intervals for a metric using bootstrap.

    Args:
        metric: Function with inputs (X, Y, *args) and outputs real number
        X, Y: Variables to compare metric for
        *args: Other arguments for the metric
        ntrials: Number of bootstrapping trials

    Returns:
        metrics: bootstrapped sample of `metric` on X and Y.

    """
    assert len(X) == len(Y)

    metrics = []
    for i in range(ntrials):
        idx_rand_sample = np.random.choice(np.arange(len(X)),
                                           size=len(X),
                                           replace=True)
        x_sample = X[idx_rand_sample]
        y_sample = Y[idx_rand_sample]
        metrics.append(metric(x_sample, y_sample, *args))

    return metrics
