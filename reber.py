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
IDX_TO_SYM = ['T', 'P', 'X', 'S', 'V', 'B', 'E']
SYMS = dict((c, i) for i, c in enumerate(IDX_TO_SYM))

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
    out = ''
    for i in range(0, xs.shape[0]):
        vs = np.nonzero(xs[i,:])[0]
        chars = [IDX_TO_SYM[v] for v in vs]
        if len(chars) == 1:
            out += chars[0]
        else:
            out += '{%s}' % ','.join(chars)
    return out


def str_to_next_embedded(s):
    """Like str_to_next, but for the Embedded Reber grammar."""
    xs = np.zeros((len(s), len(SYMS)))
    assert s[0] == 'B'
    assert s[-1] == 'E'
    assert s[1] in ['T', 'P']
    assert s[-2] == s[1]

    xs[2:-2,:] = str_to_next(s[2:-2])
    xs[ 0, SYMS['T']] = 1
    xs[ 0, SYMS['P']] = 1
    xs[ 1, SYMS['B']] = 1
    xs[-3, SYMS[s[1]]] = 1
    xs[-2, SYMS['E']] = 1
    return xs


def error_rate(network, xss, yss):
    errs = 0
    for xs, ys in zip(xss, yss):
        outs = network.predict(xs) > 0.5
        errs += np.abs(outs - ys).sum()
    return errs


if __name__ == '__main__':
    network = ocrolib.lstm.LSTM(len(SYMS), len(SYMS))  # 7 --> 7
    network.setLearningRate(0.01)

    test_seqs = [make_embedded_reber() for i in range(0, 20)]
    test_xs = [str_to_vec(seq) for seq in test_seqs]
    test_ys = [str_to_next_embedded(seq) for seq in test_seqs]

    for i in range(0, 20000):
        seq = make_embedded_reber()
        xs = str_to_vec(seq)
        ys = str_to_next_embedded(seq)
        network.train(xs, ys)
        if i % 1000 == 1:
            print '%5d iterations, %d errors' % (i, error_rate(network, test_xs, test_ys))

    for seq, xs, ys in zip(test_seqs, test_xs, test_ys):
        outs = network.predict(xs) > 0.5
        errs = np.abs(outs - ys).sum()
        print '%s: %d errors' % (seq, errs)
        if errs > 0:
            print '  %s' % vec_to_str(ys)
            print '  %s' % vec_to_str(outs)
