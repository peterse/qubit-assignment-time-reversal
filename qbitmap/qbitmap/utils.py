"""utils.py - unorganized utilities."""
from typing import Sequence
from functools import reduce
import operator as op

import cirq
import numpy as np
import networkx as nx


def get_tuple_up_to_ordering(t: tuple(), dct: dict):
    """Attempt `dct.get(t)` up to ambiguity in the ordering of `t`.

    If a dictionary has tuples as keys that correspond to the edges in an
    undirected graph, the tuple will be accessible up to a convention of ordering.

    Example:
        dct = {(1, 2): "A", (1, 3): "B"}

    Then `get_tuple_up_to_ordering((2, 1), dct` returns "A".

    Args:
        t: A length-2 tuple
        dct: A dictionary whose keys are tuples

    Returns:
        dct[t] or dct[t[::-1]]

    Raises:
        AttributeError: If both `t` and its reverse are in dct. This indicates
            that the user did not construct `dct` properly for an undirected graph.
    """
    rev = t[::-1]
    out = dct[t]
    if out is None:
        return out[rev]
    elif out[rev]:
        raise AttributeError("Input dict does not contain keys corresponding to a "
                             "valid undirected graph.")


def validate_circuit_as_subgraph(circuit: cirq.Circuit, graph: nx.Graph) -> bool:
    """Verfiy that the operations in `circuit` form a valid subgraph of `graph`.

    Every qubit and and entangling operation in `circuit` should correspond to a
    vertex or edge in `graph`. The opposite is not necessarily true.

    Args:
        circuit: A circuit defined over a valid subset of qubits from `graph`
        graph: A graph containing calibration metrics to score the circuit on

    Returns:
        Truth value for whether this circuit implies a valid subgraph.
    """
    for moment in circuit:
        for opp in moment.operations:
            qubits = opp.qubits
            coords = [(q.row, q.col) for q in qubits]
            # nx graphs are undirected by default.
            if len(coords) == 1 and not graph.has_node(*coords):
                return False
            if len(coords) == 2 and not graph.has_edge(*coords):
                return False

    return True


def gridqubit_coords_to_nx_coords(coords, width, height):
    """Cast cirq.GridQubit coordinates as networkx coordinates.

    This is for plotting only; a guaranteed correspondence for other graph-based
    routines means we do not have to worry about coordinate conversions.
    """
    # coords = [(t[0] + 1, t[1] + 1) for t in coords]
    return [(t[1], t[0]) for t in coords]


def minmax_scaler(x):
    """Normalize values of `x` to fall in [0, 1]"""
    return (x - np.min(x)) / (np.max(x) - np.min(x))


def int2bin_str(x, pad):
    """Return a little endian string for bin(x) padded to `pad` bits."""
    s = str(bin(x)[2:])
    temp = "0" * (pad - len(s)) + s
    return temp


def int2bin_array(x: int, pad: int) -> Sequence[int]:
    """Return a little endian list for bin(x) padded to `pad` bits.

    This is incredibly inefficient but is capable of handling large `pad` with
    no storage problems.
    """
    s = str(bin(x)[2:])
    temp = "0" * (pad - len(s)) + s
    return np.asarray([int(c) for c in temp], dtype=np.int8)


def kbits(n: int, k: int) -> Sequence[int]:
    """Generate integer form for all length-n bitstrings of weight k.

    Output indices are ordered consistently but arbitrarily.

    DISCLAIMER: ripped from StackOverflow, I don't take credit for this code.

    Args:
        n, k: integers

    Returns:
        Generator for indices that are ordered by their binary weight.
    """
    limit = 1 << n
    val = (1 << k) - 1
    while val < limit:
        yield val
        minbit = val & -val  #rightmost 1 bit
        fillbit = (val + minbit) & ~val  #rightmost 0 to the left of that bit
        val = val + minbit | (fillbit // (minbit << 1)) - 1


def idxsort_by_weight(n: int, w: int) -> Sequence[int]:
    """Construct a sorted list of all length-`n` bitstrings with weight <=`w`.

    Within each weight class strings are sorted arbitrariy (based on the
    implementation of `kbits`).

    Args:
        n: Number of bits

    Returns:
        List[Int] with length 2**n containing integers that are sorted by
            binary weight
    """
    out = [0]
    for k in range(1, w + 1):
        out += list(kbits(n, k))
    return out


def ncr(n: int, r: int) -> int:
    """Efficient computation of n-choose-r.

    DISCLAIMER: ripped from StackOverflow, I don't take credit for this code.

    Args:
        n, r: integers

    Returns:
        n-choose-r
    """
    r = min(r, n - r)
    numer = reduce(op.mul, range(n, n - r, -1), 1)
    denom = reduce(op.mul, range(1, r + 1), 1)
    return numer // denom


def binarr(m: int):
    """Produce an ordered column of all binary vectors length m.

    Example for m=3:
        array([[0, 0, 0],
               [0, 0, 1],
               [0, 1, 0],
               [0, 1, 1],
               [1, 0, 0],
               [1, 0, 1],
               [1, 1, 0],
               [1, 1, 1]])
    """
    d = np.arange(2**m)
    return (((d[:, None] & (1 << np.arange(m)))) > 0).astype(int)[:, ::-1]


def hist_as_np(counter, n_qubits, repetitions):
    """Cast a cirq Job histogram to a numpy array.

    Args:
        counter: A `collections.Counter` resulting from calling
            `job.results()[0].histogram(key="m")`
        n_qubits: The number of (measured) qubits the circuit ran on.
        repetitions: The number of repetitions the circuit ran for.

    Returns:
        shape (2**n_qubits) `np.ndarray` where j-th entry is the probability
        of measuring bitstring whose decimal value is `j`.
    """

    out = np.zeros(1 << n_qubits)
    for k, v in counter.items():
        out[k] = v / repetitions
    return out


def std_absX(x):
    """Basic computation of Var(|X|) ** 0.5"""
    v = np.var(x)
    e_x = np.mean(x) ** 2
    e_abs_x = np.mean(abs(x)) ** 2
    return np.sqrt(v + e_x - e_abs_x)