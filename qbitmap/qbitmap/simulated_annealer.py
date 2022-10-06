from abc import ABC, abstractmethod
from typing import Sequence, Tuple, Optional
import time

import numpy as np
import cirq

from qbitmap import metrics
from qbitmap import state_space


class SimulatedAnnealer(ABC):
    """Template class for annealers.

    The user needs to specify a cost function that acts on a `cirq.Circuit` and
    some other set of objects given in the child definition.
    """
    def __init__(self, initial_state: cirq.Circuit, verbose=False):
        self.state = initial_state
        self.verbose = verbose

    @abstractmethod
    def compute_score(self, state: cirq.Circuit) -> float:
        pass

    @abstractmethod
    def update_state(self, state: cirq.Circuit) -> cirq.Circuit:
        pass

    def anneal(self,
               nsteps: int,
               temperatures: Sequence[float],
               max_num_queries: Optional[int] = None):
        """Run the annealer for nsteps using a predfined temperature schedule.

        Args:
            max_num_queries: If provided, this will override nsteps by terminating
                the annealer after `max_num_queries` distinct states are
                queried for a score. Note that the output arrays may be
                shorter than `nsteps` in this case.
        """
        successful_transitions = 0
        improved_transitions = 0
        if max_num_queries is None:
            max_num_queries = nsteps
        self.start = time.time()
        current_score = self.compute_score(self.state)

        state_log = set([self.state])
        history_dct = {
            'assignment_history': [self.state],
            'score_history': [current_score],
            'temp_history': [temperatures[0]],
            'unique_states_counter': [len(state_log)]
        }

        if self.verbose:
            elapsed = time.time() - self.start
            print(
                '\n Step    Temperature      Score    Accept   Improve  Elapsed'
            )
            print(
                f'\r       {temperatures[0]:12.5f}{current_score:12.2f}                      {elapsed:4.4f}            '
            )

        for k, temperature in enumerate(temperatures):

            # Perform "rerouting" in the annealer state space: We allow for
            # removing certain states from the space; if the SA update finds
            # one of these removed states, it is forced to try again.
            new_score = None
            while new_score is None:
                new_state = self.update_state(self.state)
                new_score = self.compute_score(new_state)
            # Transition based on whether the new score is better or worse, as
            # well as the annealer's disposition towards exploring lower score
            # configurations.
            # This comparison is good for avoiding overflows in exp
            if new_score > current_score:
                current_score = new_score
                successful_transitions += 1
                improved_transitions += 1
                self.state = new_state
            elif np.random.random() < np.exp(
                (new_score - current_score) / temperature):
                current_score = new_score
                successful_transitions += 1
                self.state = new_state
            state_log.add(self.state)
            if self.verbose:
                remain = nsteps - k
                print(
                    f'\r {k:04d}  {temperature:12.5f}{current_score:12.2f}   {successful_transitions / (k+1):7.2%}   {improved_transitions / (k+1):7.2%}  {elapsed:4.4f}'
                )

            history_dct['assignment_history'].append(self.state)
            history_dct['score_history'].append(current_score)
            history_dct['temp_history'].append(temperature)
            history_dct['unique_states_counter'].append(len(state_log))

            if len(state_log) == max_num_queries:
                break
        out = {
            'qubit_map': self.state,
            'score': current_score,
            'successful_transitions': successful_transitions,
            'improved_transitions': successful_transitions
        }

        return out, history_dct


class SimplePathCalibrationAnnealer(SimulatedAnnealer):
    """Anneal a circuit selection based purely on a noise graph."""
    def __init__(self,
                 initial_state,
                 noise_graph,
                 num_new_max=3,
                 transition_probs=None,
                 verbose=False):
        """
        Args:
            initial_state: A circuit defined over a valid subset of qubits from `graph`.
            noise_graph: A graph containing circuit topology and calibration strengths
            num_new_max: The maximum number of new qubits to include in an update.
            transition_probs: The pmf over which transitions are sampled
        """
        super().__init__(initial_state, verbose=verbose)
        self.noise_graph = noise_graph
        print(initial_state)
        self.state = initial_state
        print(self.state)
        self.num_new_max = num_new_max
        self.transition_probs = transition_probs

        # Tracking annealer steps
        self.history = {}
        self.step_i = 0

    def compute_score(self, state: cirq.Circuit) -> float:
        return metrics.compute_calibration_fidelity(state, self.noise_graph)

    def update_state(self, state: cirq.Circuit) -> cirq.Circuit:
        return state_space.update_linear_circuit(
            circuit=state,
            graph=self.noise_graph,
            num_new_max=self.num_new_max,
            swap_dist=0,
            transition_probs=self.transition_probs,
            qubit_order=None,
            seed=None)


class SimplePathFixedResultsAnnealer(SimulatedAnnealer):
    """Anneal a circuit selection based on a set of experimental results."""
    def __init__(self,
                 initial_state,
                 paths,
                 fidelities,
                 graph,
                 num_new_max=1,
                 transition_probs=None,
                 verbose=False):
        """
        Args:
            initial_state: A tuple defined over a valid subset of qubits from `graph`.
            paths: An array with each row being a valid choice of state, e.g.

                    ((row_0, col_0), ..., (row_n, col_n))

                where `row_j`, `col_j` are the coordinates of the `j-th` qubit.
            fidelities: An ordered array of `len(paths)` entries corresponding
                to each path.
            graph: A graph over which the paths are constrained. This necessary
                to inform the state update tool of the valid paths it can pick.
            num_new_max: The maximum number of new qubits to include in an update.
            transition_probs: The pmf over which transitions are sampled
        """
        super().__init__(initial_state, verbose=verbose)

        # Construct a hash from paths and fidelities
        # Must be cast as a hashable, i.e. nested tuple\
        paths = [tuple(map(tuple, path)) for path in paths]
        self.lookup = dict(zip(paths, fidelities))

        self.state = tuple(map(tuple, initial_state))

        self.graph = graph
        self.num_new_max = num_new_max
        self.transition_probs = transition_probs

        # Tracking annealer steps
        self.history = {}
        self.step_i = 0

    def compute_score(self, state: tuple) -> float:
        return self.lookup.get(state)

    def update_state(self, state: Tuple[Tuple]) -> Tuple[Tuple]:
        return state_space.update_linear_path(
            nodes=state,
            graph=self.graph,
            num_new_max=self.num_new_max,
            swap_dist=0,
            transition_probs=self.transition_probs,
        )
