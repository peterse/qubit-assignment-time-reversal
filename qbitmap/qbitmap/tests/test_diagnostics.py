import pytest
import numpy as np

from qbitmap import circuits
import cirq
from qbitmap import utils, diagnostics, metrics
@pytest.mark.parametrize("n", [2, 3, 4])
@pytest.mark.parametrize("factor", [1, -1])
@pytest.mark.parametrize("nearest_neighbor_further_swap", [True, False])
def test_qft_sep_dfe(n, factor, nearest_neighbor_further_swap):
    """Test that noiseless DFE works for a variety of states, qubits."""
    qubits = cirq.LineQubit.range(n)
    ntrials = 3
    states = np.random.randint(2, size=(ntrials, n))

    for input_state in states:
        qft_circuit = circuits.qft_line_topology_gen(
            qubits,
            factor=factor,
            nearest_neighbor=True,
            nearest_neighbor_further_swap=nearest_neighbor_further_swap,
            measure=None)
        dfe = diagnostics.DirectFidelityEstimator(
            mode="qft_sep",
            circuit=qft_circuit,
            qubits=qubits,
            repetitions=5000,
            debug=True,
            simulate_readout_error=False,
            input_state=input_state,
            factor=factor,
            nearest_neighbor_further_swap=nearest_neighbor_further_swap)
        raw, _, raw_var, _ = dfe.run()

        np.testing.assert_allclose(raw, 1)


@pytest.mark.parametrize("n", [4])
@pytest.mark.parametrize("factor", [1, -1])
@pytest.mark.parametrize("nearest_neighbor_further_swap", [True, False])
def test_qft_sep_dfe_noisy(n, factor, nearest_neighbor_further_swap):
    """Test that noisy DFE works for a variety of states, qubits."""
    qubits = cirq.LineQubit.range(n)
    ntrials = 3
    states = np.random.randint(2, size=(ntrials, n))

    for input_state in states:
        qft_circuit = circuits.qft_line_topology_gen(
            qubits,
            factor=factor,
            nearest_neighbor=True,
            nearest_neighbor_further_swap=nearest_neighbor_further_swap,
            measure=None)

        clean_dfe = diagnostics.DirectFidelityEstimator(
            mode="qft_sep",
            circuit=qft_circuit,
            qubits=qubits,
            repetitions=50000,
            debug=True,
            simulate_readout_error=False,
            input_state=input_state,
            factor=factor,
            nearest_neighbor_further_swap=nearest_neighbor_further_swap)
        noisy_dfe = diagnostics.DirectFidelityEstimator(
            mode="qft_sep",
            circuit=qft_circuit,
            qubits=qubits,
            repetitions=50000,
            debug=True,
            simulate_readout_error=True,
            input_state=input_state,
            factor=factor,
            nearest_neighbor_further_swap=nearest_neighbor_further_swap)
        raw, _, raw_var, _ = noisy_dfe.run()

        clean_circuit = clean_dfe._qft_sep_circuit(measure=False)
        noisy_circuit = noisy_dfe._qft_sep_circuit(measure=False)
        expected = metrics.compute_true_fidelity(clean_circuit, noisy_circuit)
        np.testing.assert_allclose(raw, expected, atol=1e-2, rtol=1e-2)

@pytest.mark.parametrize("n", [4])
@pytest.mark.parametrize("factor", [1, -1])
@pytest.mark.parametrize("nearest_neighbor_further_swap", [True, False])
def test_qft_sep_dfe_noisy(n, factor, nearest_neighbor_further_swap):
    """Test that noisy DFE works for a variety of states, qubits."""
    qubits = cirq.LineQubit.range(n)
    ntrials = 3
    states = np.random.randint(2, size=(ntrials, n))
    for input_state in states:
        qft_circuit = circuits.qft_line_topology_gen(
                    qubits,
                    factor=factor,
                    nearest_neighbor=True,
                    nearest_neighbor_further_swap=nearest_neighbor_further_swap,
                    measure=None
                )

        clean_dfe = diagnostics.DirectFidelityEstimator(
            mode="qft_sep",
            circuit=qft_circuit,
            qubits=qubits,
            repetitions=50000,
            debug=True,
            simulate_readout_error=False,
            input_state=input_state,
            factor=factor,
            nearest_neighbor_further_swap=nearest_neighbor_further_swap
        )
        # Add some non-bitflip noise to the _input_ of the DFE
        # Remember that the DFE will prepend state prep. pre-circuits, so we
        # can't directly use the `qft_circuit` from above.
        # This comes with a bit of a price, as there are additional measurement
        # subcircuit rotations incorporated into the fidelity estimation, but
        # in this case noise is applied in a gatecount-independent fashion.
        noisy_input = [
            cirq.amplitude_damp(np.random.random() / 5).on(q) for q in qubits
        ] + qft_circuit + [
            cirq.depolarize(np.random.random() / 5).on(q) for q in qubits
        ]
        noisy_dfe = diagnostics.DirectFidelityEstimator(
            mode="qft_sep",
            circuit=noisy_input,
            qubits=qubits,
            repetitions=50000,
            debug=True,
            simulate_readout_error=False,
            input_state=input_state,
            factor=factor,
            nearest_neighbor_further_swap=nearest_neighbor_further_swap
        )

        raw, _, raw_var, _ = noisy_dfe.run()
        clean_circuit = clean_dfe._qft_sep_circuit(measure=False)
        noisy_circuit = noisy_dfe._qft_sep_circuit(measure=False)
        expected = metrics.compute_true_fidelity(clean_circuit, noisy_circuit)
        np.testing.assert_allclose(raw, expected, atol=1e-2, rtol=1e-2)
