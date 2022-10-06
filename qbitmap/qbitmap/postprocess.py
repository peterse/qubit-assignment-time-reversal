"""postprocess.py - utilities for data analysis after experiments have run."""
import os
import numpy as np


def load_qvals_arr(timestamp, n_experiments, path='./readout_ec', postpend=""):
    """Load in a `(n_experiments, 2, n_qubits)` array of single qubit bitpflips.

    This assumes a standard directory structure, but `path` can be modified
    if the readout error diagnostics escaped to better lands.

    Args:
        timestamp: An identifier matching this calibration to google calibration
        n_experiments: Size of the job; more of a check than a requirement
        path: Where the readout diagnostics are stored, absolute path.
        postpend: a final string to postpend to the target diagnostic data.
            Sorry, naming conventions happen on the fly.

    """

    stamp = f"qvals_sep_{timestamp}_" + "{}" + f"{postpend}.npy"
    test = np.load(os.path.join(path, stamp.format(0)))
    n_qubits = test.shape[1]
    out = np.zeros((n_experiments, 2, n_qubits))

    for i in range(n_experiments):
        out[i, :, :] = np.load(os.path.join(path, stamp.format(i))).reshape(
            2, n_qubits)
    return out


def accept_by_bitflip(threshold, qvals_arr):
    """Return a set of indices satisfying max(q01, q10)<threshold.

    This relies on an absolute path access to the readout error results.
    """
    max_by_run = qvals_arr.max(axis=(1, 2))
    out = np.arange(len(qvals_arr))[max_by_run < threshold]
    diff = len(qvals_arr) - len(out)
    print("rejected {} events for qmax>{}".format(diff, threshold))
    return out


def Y_gt_y_X_gt_X_conditional(x, y, p):
    """Compute the conditional probability for exceeding percentile p.

    This is the probability

        P(Y > percentile(y, p) | X > percentile(x, p))

    Args:
        x, y: variables to compare
        p: percentile to compute conditional probability of exceeding
    """
    xy = np.vstack((x, y)).T
    xthresh = np.percentile(x, p)
    ythresh = np.percentile(y, p)

    # p(X > xk)
    p_x_gt_xk = len(np.where(x > xthresh)[0]) / len(x)

    # p(X > xk, Y > yk)
    joint_p = 0
    for (xi, yi) in xy:
        if xi > xthresh and yi > ythresh:
            joint_p += 1
    joint_p = joint_p / len(x)

    # Definition of joint probability
    return joint_p / p_x_gt_xk
