#!/usr/bin/env python
"""Train an LSTM network to recognize the embedded Reber grammar."""

import random
import numpy as np
import ocrolib


# State transition table
TRANSITIONS = [
    [('T', 1), ('P', 2)],  # 0=B
    [('X', 3), ('S', 1)],  # 1=BT
    [('V', 4), ('T', 2)],  # 2=BP
    [('X', 2), ('S', 5)],  # 3=BTX
    [('P', 3), ('V', 5)],  # 4=BPV
    [('E', -1)],  # 5=BTXS
]

# Symbol encoding
SYMS = {'T': 0, 'P': 1, 'X': 2, 'S': 3, 'V': 4, 'B': 5, 'E': 6}

# See http://www.willamette.edu/~gorr/classes/cs449/reber.html
def make_reber():
    idx = 0
    out = 'B'
    while idx != -1:
        ts = TRANSITIONS[idx]
        symbol, idx = random.choice(ts)
        out += symbol
    return out


def make_embedded_reber():
    c = random.choice(['T', 'P'])
    return 'B%s%s%sE' % (c, make_reber(), c)


def str_to_vec(s):
    """Convert a Reber string to a sequence of unit vectors."""
    a = np.zeros((len(s), len(SYMS)))
    for i, c in enumerate(s):
        a[i][SYMS[c]] = 1
    return a


def str_to_next(s):
    """Given a Reber string, return a vectorized sequence of next chars.

    This is the target output of the Neural Net."""
    out = np.zeros((len(s), len(SYMS)))
    idx = 0
    for i, c in enumerate(s[1:]):
        ts = TRANSITIONS[idx]
        for next_c, _ in ts:
            out[i, SYMS[next_c]] = 1

        next_idx = [j for next_c, j in ts if next_c == c]
        assert len(next_idx) == 1
        idx = next_idx[0]

    return out


def vec_to_str(xs):
    """Given a matrix, return a Reber string (with choices)."""
    idx_to_sym = dict((v,k) for k,v in SYMS.iteritems())
    out = ''
    for i in range(0, xs.shape[0]):
        vs = np.nonzero(xs[i,:])[0]
        chars = [idx_to_sym[v] for v in vs]
        if len(chars) == 1:
            out += chars[0]
        else:
            out += '{%s}' % ','.join(chars)
    return out


def str_to_next_embedded(s):
    """Like str_to_next, but for the Embedded Reber grammar."""


if __name__ == '__main__':
    network = ocrolib.lstm.LSTM(len(SYMS), len(SYMS))  # 7 --> 7
    network.setLearningRate(0.1)

    for i in range(0, 5000):
        seq = make_reber()
        xs = str_to_vec(seq)
        ys = str_to_next(seq)
        network.train(xs, ys)
        if i % 1000 == 1:
            print '%5d iterations' % i

    for i in range(0, 20):
        seq = make_reber()
        xs = str_to_vec(seq)
        ys = str_to_next(seq)
        outs = network.predict(xs) > 0.5
        errs = np.abs(outs - ys).sum()
        print '%s: %d errors' % (seq, errs)
        if errs > 0:
            print vec_to_str(ys)
            print vec_to_str(outs)
