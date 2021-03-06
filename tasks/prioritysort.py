"""Priory Sort NTM model"""
import random

from attr import attrs, attrib, Factory
import torch
from torch import nn
from torch import optim
import numpy as np

from ntm.aio import EncapsulatedNTM

def dataloader(num_batches,
               batch_size,
               seq_len,
               seq_width):
    """
    Data loader for the Priority Sort task.

    It generates set of binary sequences with
    attached a priority drawn from a given
    distribution.

    :param num_batches:
    :param batch_size:
    :param seq_width:
    :return:
    """
    for batch_num in range(num_batches):
        seq = np.random.binomial(1, 0.5, (seq_len, batch_size, seq_width))
        seq = torch.from_numpy(seq)

        # The input includes an additional channel used for the delimiter
        inp = torch.zeros(seq_len + 1, batch_size, seq_width + 1)

        # Add priority number (just a single one drawn from the uniform distribution)
        priority = np.random.uniform(-1, 1, (seq_len, batch_size, 1))
        priority = torch.from_numpy(priority)

        # Construct the input vectors
        inp[:seq_len, :, :seq_width] = seq
        inp[:seq_len, :, (seq_width):] = priority  # priority

        # Add the delimiter
        inp_delim = torch.zeros(seq_len + 1, batch_size, seq_width + 2)
        inp_delim[:(seq_len+1), :, :(seq_width+1)] = inp
        inp_delim[seq_len, :, seq_width+1] = 1.0  # delimiter in our control channel

        # Construct the output which will be a sorterd version of
        # the sequences given by looking at the priority
        outp = inp_delim.numpy()

        # Strip all the binary vectors into a list
        # and sort the list by looking at the last column
        # (which will contain the priority)
        temp = []
        for i in range(len(outp)):
            temp.append(outp[i][0])
        del temp[-1] # Remove the delimiter
        temp.sort(key=lambda x: x[seq_width], reverse=True) # Sort elements descending order

        # Keep only the highest entries as specified in the paper.
        # This means that for 20 entries we want to predict only the highest 16.
        # This will be done only if a sequence is larger than 4 elements.
        if len(temp) > 4:
            del temp[-4:]

        # FIXME
        # Ugly hack to present the tensor structure as the one
        # required by the framework
        layer = []
        for i in range(len(temp)):
            tmp_layer = []
            tmp_layer.append(np.array(temp[i]))
            layer.append(tmp_layer)

        # Convert everything to numpy and to a tensor
        outp = torch.from_numpy(np.array(layer))

        yield batch_num + 1, inp_delim.float(), outp.float()

@attrs
class PrioritySortTaskParams(object):
    name = attrib(default="priority-sort-task")
    cuda = attrib(default=False)
    controller_size = attrib(default=100, converter=int)
    controller_layers = attrib(default=2,converter=int)
    num_heads = attrib(default=5, converter=int)
    sequence_width = attrib(default=8, converter=int)
    sequence_min_len = attrib(default=1,converter=int)
    sequence_max_len = attrib(default=20, converter=int)
    memory_n = attrib(default=128, converter=int)
    memory_m = attrib(default=20, converter=int)
    num_batches = attrib(default=50000, converter=int)
    batch_size = attrib(default=1, converter=int)
    rmsprop_lr = attrib(default=3e-5, converter=float)
    rmsprop_momentum = attrib(default=0.9, converter=float)
    rmsprop_alpha = attrib(default=0.95, converter=float)

@attrs
class PrioritySortTaskModelTraining(object):
    params = attrib(default=Factory(PrioritySortTaskParams))
    cuda = attrib(default=False)
    net = attrib()
    dataloader = attrib()
    criterion = attrib()
    optimizer = attrib()

    @net.default
    def default_net(self):

        # We have an input/output increase of the size since we want to return also
        # the priority of that specific sequence
        net = EncapsulatedNTM(self.params.sequence_width + 2, self.params.sequence_width+2,
                              self.params.controller_size, self.params.controller_layers,
                              self.params.num_heads,
                              self.params.memory_n, self.params.memory_m,
                              self.cuda)
        if self.cuda:
            net = net.cuda()

        return net

    @dataloader.default
    def default_dataloader(self):
        return dataloader(self.params.num_batches, self.params.batch_size,
                          self.params.sequence_max_len, self.params.sequence_width)

    @criterion.default
    def default_criterion(self):
        criterion = nn.BCELoss()
        if self.cuda:
            criterion = criterion.cuda()

        return criterion

    @optimizer.default
    def default_optimizer(self):
        optimizer = optim.RMSprop(self.net.parameters(),
                                  momentum=self.params.rmsprop_momentum,
                                  alpha=self.params.rmsprop_alpha,
                                  lr=self.params.rmsprop_lr)

        return optimizer
