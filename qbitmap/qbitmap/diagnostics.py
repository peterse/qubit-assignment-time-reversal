import os
from typing import Sequence, Optional

import matplotlib.pyplot as plt
import numpy as np
import sympy
import cirq
import cirq_google as cg

from qbitmap import hw
from qbitmap import utils


def debug_run_sweep(engine, program, repetitions, params, debug):
    if debug:
        return engine.run_sweep(
            program=program,
            repetitions=repetitions,
            params=params,
        )
    return engine.run_sweep(
        program=program,
        repetitions=repetitions,
        params=params,
        processor_ids=[hw.PROCESSOR_ID],
        gate_set=hw.GATESET,
    )


def readout_error_debug_ops(qubits):
    return [cirq.bit_flip(.01 * (j + 1)).on(q) for j, q in enumerate(qubits)]


def readout_diagnostic_circuit(symbols, qubits, debug):
    """A parameterized circuit for preparing computational basis states."""
    ops = [(cirq.X**x).on(q) for (x, q) in zip(symbols, qubits)]

    # For debugging, give each qubit a different readout error probabilitc
    if debug:
        ops += readout_error_debug_ops(qubits)
    circuit = cirq.Circuit(*ops)
    circuit += cirq.measure(*qubits, key='m')
    return circuit


def plot_response_matrix(R,
                         n_qubits,
                         targets,
                         cutoff=None,
                         fig=None,
                         sort_by_weight=True,
                         **kwargs):
    """Plot the response matrix from this diagnostic."""
    if fig is None:
        fig, ax = plt.subplots(figsize=(8, 8))
    else:
        ax = fig.add_subplot()
    # Rearrange labels on the space of U according to bitstring weight
    if not sort_by_weight:
        raise NotImplementedError

    if not any(kwargs):
        kwargs = {'cmap': 'seismic', 'vmin': -1, 'vmax': 1}
    im = ax.imshow(R, **kwargs)
    # ticklabels get shuffled according to weight
    ticklabs = [utils.int2bin_str(i, n_qubits) for i in targets]
    ax.set_xticks(range(len(targets)))
    ax.set_xticklabels(ticklabs, rotation=90, size=12)
    ax.set_yticks(range(len(targets)))
    ax.set_yticklabels(ticklabs, size=12)

    if cutoff:
        # Put boundaries between fixed-particle number regions
        k = 0
        for i in range(cutoff):
            k += utils.ncr(n_qubits, i)
            ax.axhline(k - 0.5, ls=':', lw=1, c='k', alpha=1)
            ax.axvline(k - 0.5, ls=':', lw=1, c='k', alpha=1)

    fig.subplots_adjust(right=0.85)
    cbar_ax = fig.add_axes([0.88, 0.15, 0.04, 0.7])
    fig.colorbar(im, cax=cbar_ax)

    return fig, ax


class ReadoutErrorDiagnostic():
    def __init__(self,
                 timestamp: str,
                 qubits: Sequence[cirq.Qid],
                 repetitions: int,
                 cutoff: Optional[int] = None,
                 path: Optional[str] = "./",
                 debug: Optional[bool] = False):
        """Implement a readout error diagnostic assuming correlated bitflips.

        This will require an exponential number of experiments in the number of
        qubits, and so should be reserved for small validation runs.

        Args:
            timestamp: A string specifying date/time of diagnostic.
            qubits: The set of qubits to perform simultaneous readout on during
            repetitions: Number of experiments per bitstring to run.
            cutoff: The weight cutoff for determining transition probability between
                bitstrings x and y. If provided, this diagnostic will compute
                P(x|0) only if |x| <= w.
            path: Optional directory to log results to.
            engine: Backend to submit to. Use engine=cirq.Simulator() for debugging.
        """

        self.timestamp = timestamp

        self.qubits = qubits
        self.n_qubits = len(qubits)

        self.repetitions = repetitions
        self.path = path
        self.debug = debug

        self.engine = hw.ENGINE
        if debug:
            self.engine = cirq.DensityMatrixSimulator()

        # Generate a persistent hash from bin(x) to index in weight-sorted array
        self.symbols = [sympy.Symbol(str(q)) for q in self.qubits]
        self.R = None

        self.cutoff = cutoff
        if cutoff:
            self.targets = utils.idxsort_by_weight(len(qubits), cutoff)
        else:
            self.targets = np.arange(1 << len(qubits))

    def _make_output_fname(self):
        if self.cutoff:
            return os.path.join(self.path,
                                f"R_{self.timestamp}_cutoff{self.cutoff}.npy")
        return os.path.join(self.path, f"R_{self.timestamp}_full.npy")

    def _response_matrix_diagnostic_sweep(self):
        """Generate a sweep over binary masks specifying bitstrings."""
        out = []
        for mask in self.targets:
            vec = utils.int2bin_array(mask, self.n_qubits)
            out.append(dict(zip(self.symbols, vec)))
        return out

    def _make_response_matrix(self, data):
        """Compute the response matrix for this experiment.

        Args:
            data: job returned by calling `run_sweep` method of a cirq engine.
            dim: Dimensions of the output response matrix
        """
        dim = len(self.targets)
        R = np.zeros((dim, dim))
        # The results of the experiment are ordered by weight sector, but the
        # bitstrings in each counter are unordered.
        for j, result in enumerate(data):
            counter = result.histogram(key="m")
            for i, idx in enumerate(self.targets):
                counts = counter[idx]
                R[i, j] = counts / self.repetitions
        self.R = R
        return R

    def run(self):
        """Run the diagnostic on hardware.

        If `cutoff` was provided, it is used to generate response matrix
        characterizing transitions of order `w` or less. The resulting matrix
        will have dimensions t(w) x t(w) where

            t(w) = sum_{i=0}^w C(n, w)

        and C(n, k) is the n-choose-k function. This matrix can be produced with
        t(w) diagnostic experiments that prepare the state. Otherwise, this will
        naively run 2^n experiments to determine P(i||j⟩) for all bitstrings i,j.

        """
        if not self.debug:
            try:
                self.load()
            except FileNotFoundError:
                pass
            else:
                raise ValueError(
                    "Found previous data at {} - delete or move this "
                    "data before running diagnostic."
                    "".format(self._make_output_fname()))

        # These parameters define masks for all bitstrings of weight
        sweep_params = self._response_matrix_diagnostic_sweep()

        circuit = readout_diagnostic_circuit(self.symbols, self.qubits,
                                             self.debug)
        job = debug_run_sweep(engine=self.engine,
                              program=circuit,
                              repetitions=self.repetitions,
                              params=sweep_params,
                              debug=self.debug)

        # Postprocess and save results
        R = self._make_response_matrix(job)
        np.save(self._make_output_fname(), R)

        return R

    def load(self):
        """Construct a response matrix from previous data runs.

        This is only to be called by subclasses responsible for actually
        generating response matrices.

        Returns:
            R: Response matrix (indices ordered by weight sector targets)
            Rvar: Beroulli variance estimators for each element of R.
        """
        R = np.load(self._make_output_fname())
        self.R = R
        return R

    def plot(self):
        return plot_response_matrix(self.R, self.n_qubits, self.targets,
                                    self.cutoff)


class SeparableReadoutErrorDiagnostic():
    """Implement a readout error diagnostic assuming uncorrelated bitflips.

    Args:
        timestamp: A string specifying date/time of diagnostic.
        qubits: The set of qubits to perform simultaneous readout on during
        repetitions: Number of experiments per bitstring to run.
        path: Optional directory to log results to.
        engine: Backend to submit to. Use engine=cirq.Simulator() for debugging.
    """
    def __init__(self,
                 timestamp: str,
                 qubits: Sequence[cirq.Qid],
                 repetitions: int,
                 path: Optional[str] = "./",
                 debug: bool = False):
        self.timestamp = timestamp

        self.qubits = qubits
        self.n_qubits = len(qubits)

        self.repetitions = repetitions
        self.path = path
        self.debug = debug

        self.engine = hw.ENGINE
        if debug:
            self.engine = cirq.DensityMatrixSimulator()

        # Generate a persistent hash from bin(x) to index in weight-sorted array
        self.symbols = [sympy.Symbol(str(q)) for q in self.qubits]
        self.qvals = None

    def _make_output_fname(self):
        return os.path.join(self.path, f"qvals_sep_{self.timestamp}.npy")

    def _get_conditionals(self, x, xhat, y, yhat):
        """Compute bitwise conditional bitflip probabilities.

        Note
        I assume independent readout _errors_ (even if the
        readout probailities are interdependent) and therefore compute q_k using
        bitwise mean over all experiments, instead of some statistic on
        the observed set of length-n bitstrings.

        Args:
            x, y: A test bitstring and its compelemtn
            xhat, yhat: mean observed bitstrings for x, y respectively.
        """
        n = len(x)
        # first row is q(1|0), second row is q(0|1)
        q_mat = np.zeros((2, n))

        qx = abs(x - xhat)
        for i, (xi, qi) in enumerate(zip(x, qx)):
            q_mat[xi, i] = qi

        qy = abs(y - yhat)
        for j, (yj, qj) in enumerate(zip(y, qy)):
            q_mat[yj, j] = qj

        return q_mat[1, :], q_mat[0, :]

    def run(self, ntrials):
        """Run the diagnostic assuming a separable response matrix.

        This will populate a (2, n, ntrials) matrix where the (0, k, :) slice
        stores the trials for single bit conditionals q_k(0|1) for observing `0`
        given `1` on qubit `k`, while the `(1, k, :)` slice stores q_k(1|0).

        To remove some possible bias, this diagnostic generates a random string
        s ∈ {0,1}^n and then runs X_k on each qubit with power given by the bit
        value s_k, and then repeats the process for the complement ̄s = s ⊕ 1...1

        Args:
            ntrials: How many different configurations of `s` to attempt. Total
                number of experiments is `2 * ntrials * repetitions`

        Returns:
            qmat: `(2, n_qubits, ntrials)` array storing all trials for q_k(i|j)
        """
        if not self.debug:
            try:
                self.load()
            except FileNotFoundError:
                pass
            else:
                raise ValueError(
                    "Found previous data at {} - delete or move this "
                    "data before running diagnostic."
                    "".format(self._make_output_fname()))

        out = np.zeros((2, self.n_qubits, ntrials))
        circuit = readout_diagnostic_circuit(self.symbols, self.qubits,
                                             self.debug)
        svals = [
            np.random.randint(2, size=self.n_qubits) for _ in range(ntrials)
        ]
        sbar_vals = [(~s.astype(bool)).astype(int) for s in svals]
        params = []
        for trial in range(ntrials):
            # Sweep over a column of X masked by s, then a column masked by ̄s
            params += [
                dict(zip(self.symbols, svals[trial])),
                dict(zip(self.symbols, sbar_vals[trial]))
            ]

        job = debug_run_sweep(engine=self.engine,
                              program=circuit,
                              repetitions=self.repetitions,
                              params=params,
                              debug=self.debug)
        for i in range(ntrials):
            s = svals[i]
            sbar = sbar_vals[i]
            shat = job[2 * i].measurements['m'].mean(axis=0)
            sbarhat = job[2 * i + 1].measurements['m'].mean(axis=0)
            q01, q10 = self._get_conditionals(s, shat, sbar, sbarhat)
            out[0, :, i] = q01
            out[1, :, i] = q10

        np.save(self._make_output_fname(), out)
        self.qvals = out

        return out

    def load(self):
        qvals = np.load(self._make_output_fname())
        self.qvals = qvals
        return qvals

    def make_response_matrices(self):
        """Construct a set of single-qubit responses from experimental data.

        Args:
            qvals: shape `(2, n, ntrials)` array containing results of computing
                each single-qubit bitflip conditional.
        """
        if self.qvals is None:
            raise ValueError
        out = []
        qmeans = self.qvals.mean(axis=2)
        for k in range(self.n_qubits):
            q01 = qmeans[0, k]
            q10 = qmeans[1, k]
            temp = np.asarray([[1 - q10, q01], [q10, 1 - q01]])
            out.append(temp)
        return out

    def invert_and_correct(self, arr):
        """Apply readout error correction to a job using the sampled readout err.

        """
        if self.qvals is None:
            raise ValueError(
                "Diagnostic does not have `qvals` set - load or run the experiment."
            )
        Qmats = self.make_response_matrices()
        Qmats_inv = [np.linalg.inv(Q) for Q in Qmats]
        R_inv = Qmats_inv[0]
        for i in range(1, len(Qmats)):
            R_inv = np.kron(R_inv, Qmats_inv[i])
        return R_inv.dot(arr)


class DirectFidelityEstimator:
    """Utility for direct fidelity estimation for a specific circuit."""
    supported_modes = ['ghz', 'qft_sep']

    def __init__(self,
                 mode,
                 circuit,
                 qubits,
                 repetitions,
                 input_state: Optional[Sequence] = None,
                 debug=False,
                 simulate_readout_error=False,
                 readout_calibration=None,
                 factor=None,
                 nearest_neighbor_further_swap=None):
        """

        This relies on prior knowledge of an efficient Pauli decomposition for
        the state we wish to estimate fidelity on. Currently supported modes are

            'ghz' - Fidelity estimation for |0^n⟩ + |1^n⟩. This requires (n+1)
                experiments.
            `qft_sep` - Fidelity estimation for QFT|k> for a fixed input state
                [k_0, k_1, ...]. This requires 1 experiment. The input circuit
                MUST HAVE THE INPUT STATE PREPARATION CIRCUIT PREPENDED. This
                will not be done by the DFE estimation routine.

        If you want to simulate readout error, you must do so through the
        `simulate_readout_error` interface - this is because this diagnostic
        modifies the input circuit by adding gates, and non-terminal bitflip
        errors are not reliably corrected by postprocessing. Therefore setting
        this option will apply a fixed, terminal bitflip error to the qubits.

        Args:
            mode: Fixed identifier for the state fidelity to estimate.
            circuit: Circuit that generates desired state on hardware. Should
                not contain any measurements.
            qubits: A sequence of ordered qubits.
            repetitions: How many repetitions to estimate each circuit
            input_state: A sequence describing the input computational basis
                state for a separable QFT diagnostic. This sequence is assumed
                to describe an integer with least signficant bit to the right.
            debug: Whether to invoke a simulator or query hardware directly
            simulate_readout_error: Insert simulated readout error to test
                readout error correction.
            readout_calibration: A diagnostic that exposes `invert_and_correct`
                that can correct the fidelity estimates using readout error
                correction.

        """

        if mode not in self.supported_modes:
            raise NotImplementedError(f"Currently don't support mode: {mode}")
        self.mode = mode

        if mode != 'qft_sep' and any(
            [input_state, factor, nearest_neighbor_further_swap]):
            raise ValueError(f"Received inputs not supported for mode: {mode}")
        if mode == 'qft_sep':
            if ((input_state is None) or (factor is None)
                    or (nearest_neighbor_further_swap is None)):
                raise ValueError(
                    f"Must provide the following kwargs for `qft_sep`:"
                    "\n\t input_state\n\t factor\n\t nearest_neighbor_further_swap"
                )
            else:
                assert len(input_state) == len(qubits)
                self.input_state = input_state
                assert np.allclose(abs(factor), 1)
                self.factor = factor
                self.nearest_neighbor_further_swap = nearest_neighbor_further_swap

        self.circuit = circuit
        self.qubits = qubits
        self.n = len(self.qubits)

        self.debug = debug
        self.simulate_readout_error = simulate_readout_error
        self.repetitions = repetitions
        self.engine = hw.ENGINE
        if debug:
            self.engine = cirq.DensityMatrixSimulator()
        self.readout_calibration = readout_calibration

    def dump_estimation_circuit(self):
        """Print out circuits used for DFE in the provided mode."""

        if self.mode == 'ghz':
            return self._dump_ghz_circuits()
        if self.mode == 'qft_sep':
            return self._dump_qft_circuits()

    def run(self, factor=None, nearest_neighbor_further_swap=None):
        """Run the direct fidelity estimation.

        BE CAREFUL: The function signature varies depending on the mode. Namely,
        the non-commutativity of the observables for ghz DFE make estimating
        Variance impractical, while the QFT mode supports this trivially.
        Args:
            All kwargs are for `qft_sep`, provide as many from the circuit
            generators as you see fit. Currently supported generators:
                `circuits.qft_line_topology_gen`
                `circuits.qft_2x2_topology_gen`
            These will be used for either compatibility with the provided circuit
            generator or for throwing NotImplementedError when the DFE doesn't
            support a given generator argument.
        """
        if self.mode == 'ghz':
            # FIXME: I want to be able to get both error-corrected and vanilla results simultaneously
            # Right now I have to call the below method again with `readout_correction` set to None
            if any([factor, nearest_neighbor_further_swap]):
                raise ValueError(
                    "GHZ DFE does not support additional options.")
            fz_job, fxy_job = self._run_ghz()
            raw = self._process_ghz(fz_job, fxy_job, readout_calibration=None)
            corr = self._process_ghz(
                fz_job, fxy_job, readout_calibration=self.readout_calibration)
            return raw, corr
        if self.mode == 'qft_sep':
            fm_job = self._run_qft_sep()
            raw, raw_var = self._process_qft_sep(fm_job,
                                                 readout_calibration=None)
            corr, corr_var = self._process_qft_sep(
                fm_job, readout_calibration=self.readout_calibration)
            return raw, corr, raw_var, corr_var

    def _run_ghz(self):
        """Run the direct fidelity estimation for a GHZ state.

        This is a straightforward implementation of the technique of

        Otfried Gühne, Chao-Yang Lu, Wei-Bo Gao, and Jian-Wei Pan
            Phys. Rev. A 76, 030305(R) – 18 September 2007
        """

        # First we need to measure ⟨Z^n⟩, which has no measurement subcircuit.
        fz_circuit = self.circuit.copy()
        if self.simulate_readout_error:
            fz_circuit += readout_error_debug_ops(self.qubits)
        fz_circuit += cirq.measure(*self.qubits, key='m')
        fz_job = debug_run_sweep(engine=self.engine,
                                 program=fz_circuit,
                                 repetitions=self.repetitions,
                                 params=None,
                                 debug=self.debug)

        # Then we need to sweep over n local Z rotations
        symbols = [sympy.Symbol(f"{str(q)}_local") for q in self.qubits]
        fxy_measure = cirq.Circuit(
            [cirq.Rz(rads=x).on(q) for (x, q) in zip(symbols, self.qubits)] +
            [cirq.H.on(q) for q in self.qubits])
        if not self.debug:
            fxy_measure = cg.optimized_for_sycamore(
                fxy_measure, optimizer_type="sqrt_iswap")
        params = []
        for k in range(1, self.n + 1):
            angle = -1. * k * np.pi / self.n
            params.append(dict(zip(symbols, [angle for _ in symbols])))

        fxy_circuit = self.circuit.copy() + fxy_measure
        if self.simulate_readout_error:
            fxy_circuit += readout_error_debug_ops(self.qubits)
        fxy_circuit += cirq.measure(*self.qubits, key='m')
        fxy_job = debug_run_sweep(engine=self.engine,
                                  program=fxy_circuit,
                                  repetitions=self.repetitions,
                                  params=params,
                                  debug=self.debug)
        return fz_job, fxy_job

    def _dump_ghz_circuits(self):
        circuits_out = []
        fz_circuit = self.circuit.copy()
        fz_circuit += cirq.measure(*self.qubits, key='m')
        circuits_out.append(fz_circuit)

        # Then we need to sweep over n local Z rotations
        symbols = [sympy.Symbol(f"{str(q)}_local") for q in self.qubits]
        fxy_measure = cirq.Circuit(
            [cirq.Rz(rads=x).on(q) for (x, q) in zip(symbols, self.qubits)] +
            [cirq.H.on(q) for q in self.qubits])
        params = []
        for k in range(1, self.n + 1):
            angle = -1. * k * np.pi / self.n
            params.append(dict(zip(symbols, [angle for _ in symbols])))

        fxy_circuit = self.circuit.copy() + fxy_measure
        fxy_circuit += cirq.measure(*self.qubits, key='m')
        circuits_out.append(fxy_circuit)
        return circuits_out, params

    def _process_ghz(self, fz_data, fxy_data, readout_calibration=None):
        """Postprocess measurement results to compute fidelity."""

        z_observable = cirq.I(self.qubits[0]) * 0
        # TODO: more efficient subroutine for this?
        for vec in utils.binarr(self.n):
            if sum(vec) % 2 != 0:
                continue
            paulistring = []
            for k, val in enumerate(vec):
                if val == 0:
                    paulistring.append(cirq.I(self.qubits[k]))
                else:
                    paulistring.append(cirq.Z(self.qubits[k]))
            z_observable += 2**(-self.n) * cirq.PauliString(*paulistring)

        # Up to preprocessing this is the observable we're interested in
        xy_observable = cirq.PauliString(*[cirq.Z(q) for q in self.qubits])

        # Offload expectation value calculations to cirq
        # well technically not offloading - I wrote that cirq code too :p
        fz_wf = np.zeros(1 << self.n, dtype=np.complex64)
        for k, v in fz_data[0].histogram(key='m').items():
            fz_wf[k] = v
        if readout_calibration:
            fz_wf = readout_calibration.invert_and_correct(fz_wf)
        fz_wf = np.sqrt(fz_wf / self.repetitions)
        fz_wf = fz_wf / np.linalg.norm(fz_wf, ord=2)
        f_z = z_observable.expectation_from_state_vector(
            state_vector=fz_wf,
            qubit_map={q: i
                       for i, q in enumerate(self.qubits)})

        f_xy = 0
        for j, trial in enumerate(fxy_data):
            fxy_wf = np.zeros(1 << self.n, dtype=np.complex64)
            for k, v in trial.histogram(key='m').items():
                fxy_wf[k] = v
            # import pdb; pdb.set_trace()
            if readout_calibration:
                fxy_wf = readout_calibration.invert_and_correct(fxy_wf)
            fxy_wf = np.sqrt(fxy_wf / self.repetitions)
            fxy_wf = fxy_wf / np.linalg.norm(fxy_wf, ord=2)

            fxy_j = xy_observable.expectation_from_state_vector(
                state_vector=fxy_wf,
                qubit_map={q: i
                           for i, q in enumerate(self.qubits)})
            f_xy += (-1)**(j + 1) * fxy_j / (self.n * 2)

        return (f_xy + f_z).real

    def _qft_sep_circuit(self, measure=True):
        """Construct just the circuit for QFT DFE.

        This is mostly for debugging purposes to get exact wavefunction
        simulation to check fidelity of the state directly.
        """
        # Cast the input state to int
        jint = 0
        for b in self.input_state:
            jint = (jint << 1) | int(b)

        # Thetas need to be in reverse order compared to significance of input
        angles = list(
            reversed([
                self.factor * np.pi * jint * 2**(1 - p)
                for p in range(1, self.n + 1, 1)
            ]))
        if self.nearest_neighbor_further_swap is False:
            # When using `nearest_neighbor_further_swap`, the qubits in
            # positions [1:n-1] are in ascending order.
            # To match this, we must permute our angles from [n-1, n-2, ..., 0] to
            # [n-1, 0, 1, 2]
            angles = [angles[0]] + list(reversed(angles[1:]))

        fm_measure = cirq.Circuit(
            [cirq.Rz(rads=-x).on(q) for (x, q) in zip(angles, self.qubits)] +
            [cirq.H.on(q) for q in self.qubits])
        if not self.debug:
            fm_measure = cg.optimized_for_sycamore(fm_measure,
                                                   optimizer_type="sqrt_iswap")
        # Compose the measurement subcircuit with potentially readout error
        # debugging noise. Also prepend the state prep circuit for |j>

        fm_circuit = self.circuit.copy() + fm_measure
        if self.simulate_readout_error:
            fm_circuit += readout_error_debug_ops(self.qubits)
        if measure:
            fm_circuit += cirq.measure(*self.qubits, key='m')
        return fm_circuit

    def _run_qft_sep(self):
        """Run the direct fidelity estimation for QFT|k> for some basis state.

        Here we use the convention that n-qubit QFT is defined elementwise as

            [QFT]_{jk} = exp(-i2πjk/2^n)

        Also note that we do NOT expect the input circuit to implement any
        SWAP network - the qubits at the circuit output are expected to be
        ordered with the least significant bit at the "left" of the ket.

        Returns:
            job: a (hardware) job containing measurements for the single DFE
                circuit.
        """
        fm_circuit = self._qft_sep_circuit(measure=True)
        fm_job = debug_run_sweep(engine=self.engine,
                                 program=fm_circuit,
                                 repetitions=self.repetitions,
                                 params=None,
                                 debug=self.debug)

        return fm_job

    def _dump_qft_circuits(self):
        jint = 0
        for b in self.input_state:
            jint = (jint << 1) | int(b)
        angles = list(
            reversed([
                self.factor * np.pi * jint * 2**(1 - p)
                for p in range(1, self.n + 1, 1)
            ]))
        if self.nearest_neighbor_further_swap is False:
            angles = [angles[0]] + list(reversed(angles[1:]))
        fm_measure = cirq.Circuit(
            [cirq.Rz(rads=-x).on(q) for (x, q) in zip(angles, self.qubits)] +
            [cirq.H.on(q) for q in self.qubits])
        fm_circuit = self.circuit.copy() + fm_measure
        fm_circuit += cirq.measure(*self.qubits, key='m')
        return fm_circuit

    def _process_qft_sep(self, fm_data, readout_calibration=None):
        """Process the results of DFE on a separable QFT circuit."""

        # Construct the sum of all possible length-n pauli Z strings.
        z_observable = cirq.I(self.qubits[0]) * 0
        for mask in utils.binarr(self.n):
            z_observable += 2**(-self.n) * cirq.PauliString(*[
                cirq.I(q) if b == 0 else cirq.Z(q)
                for b, q in zip(mask, self.qubits)
            ])

        # Once again ugly hack to do expectation values from job results
        fm_wf = np.zeros(1 << self.n, dtype=np.complex64)
        for k, v in fm_data[0].histogram(key='m').items():
            fm_wf[k] = v
        if readout_calibration:
            fm_wf = readout_calibration.invert_and_correct(fm_wf)
        fm_wf = np.sqrt(fm_wf / self.repetitions)
        fm_wf = fm_wf / np.linalg.norm(fm_wf, ord=2)

        #
        fm = z_observable.expectation_from_state_vector(
            state_vector=fm_wf,
            qubit_map={q: i
                       for i, q in enumerate(self.qubits)})
        Z_sq = (z_observable * z_observable).expectation_from_state_vector(
            state_vector=fm_wf,
            qubit_map={q: i
                       for i, q in enumerate(self.qubits)})
        var_fm = Z_sq - fm.real**2

        return fm.real, var_fm.real
