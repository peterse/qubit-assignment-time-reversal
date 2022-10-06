"""io.py - input and output for mostly simulated SA experiments."""
import os
from typing import Optional


def make_experiment_fout(connectivity: str, circuit_name: str, n_qubits: int,
                         tag: str, path: Optional[str] = "."):
    """Make a filename for the simulated SA data.

    Args:
        connectivity: Circuit connectivity, choose one of
            `line`, `grid`, `all`
        circuit_name: Family of circuits diagnosed, e.g. `ghz`, `rand`, `qft`
        n_qubits: number of qubits in experiment
        tag: Encode additional metadata. For instance, you probably want to
            keep track of which noise map you used in this experiment.
        path: Directory to save/load file to.

    Retursn:
        path filename, results filename
    """
    template = os.path.join(
        path, f"{connectivity}_{circuit_name}_{n_qubits}_{tag}")
    path_fname = template + "_paths.npy"
    res_fname = template + "_results.npy"
    return path_fname, res_fname
