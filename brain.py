#!/usr/bin/env python
"""Train an LSTM network on the embedded Reber Grammar with pybrain.
"""

import reber

from scipy import sin, rand, arange
from pybrain.datasets            import SequentialDataSet
from pybrain.structure.modules   import LSTMLayer, SigmoidLayer
from pybrain.supervised          import BackpropTrainer
from pybrain.tools.validation    import ModuleValidator, SequenceHelper
from pybrain.tools.shortcuts     import buildNetwork

import numpy as np

NUM_SYMS = len(reber.SYMS)


rnn = buildNetwork(NUM_SYMS, 4, NUM_SYMS,
                   hiddenclass=LSTMLayer,
                   outclass=SigmoidLayer,
                   recurrent=True)

# Staggered initialization
lstm = rnn.modulesSorted[2]
lstm.forgetgate = np.linspace(1, 3, lstm.forgetgate.shape[1])
lstm.ingate = -lstm.forgetgate
lstm.outgate = -lstm.forgetgate

def make_data_set(N):
    data = SequentialDataSet(NUM_SYMS, NUM_SYMS)
    count = 0
    for i in range(0, N):
        seq = reber.make_embedded_reber()
        xs = reber.str_to_vec(seq)
        ys = reber.str_to_next_embedded(seq)
        data.newSequence()
        for x, y in zip(xs, ys):
            data.addSample(x, y)
            count += 1
    return data


NUM_TRAIN = 250

train_data = make_data_set(NUM_TRAIN)
test_data = make_data_set(50)


def num_errors(module, dataset):
    target = dataset.getField('target')
    output = ModuleValidator.calculateModuleOutput(module, dataset)
    target = np.array(target)
    output = np.array(output)
    ends = SequenceHelper.getSequenceEnds(dataset)
    # target and output are Nx7 arrays, the concatenation of all sequences.
    # print 'ends: %s' % str(ends)
    # print 'target: %s' % str(output.shape)
    errs = 0
    for start_idx, limit_idx in zip(np.concatenate(([0], ends[:-1])), ends):
        ys = target[start_idx:limit_idx+1,:]
        outs = output[start_idx:limit_idx+1,:] > 0.5
        new_errs = np.abs(outs - ys).sum()
        if errs == 0 and new_errs != 0:
            print 'ys:   %s' % reber.vec_to_str(ys)
            print 'outs: %s' % reber.vec_to_str(outs)
        errs += new_errs
    return errs


trainer = BackpropTrainer(rnn,
                          dataset=train_data,
                          verbose=True,
                          #momentum=0.9,
                          momentum=0.0,
                          learningrate=0.1 )

n_epochs = 0
for i in range(100):
    trainer.trainEpochs(2)
    n_epochs += 2
    test_result = ModuleValidator.MSE(rnn, test_data)
    test_errs = num_errors(rnn, test_data)
    print "%03d / %05d test MSE: %f (%d errs)" % (
            n_epochs, NUM_TRAIN * n_epochs, test_result, test_errs)
