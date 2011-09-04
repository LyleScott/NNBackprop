"""
Author : Lyle Scott III    lyle@digitalfoo.net
Date   : 2011-03-26
"""
import cPickle
import datetime
import math
import random
import sys
import time
import NNUtils


class NeuronLink():
    def __init__(self, in_index, out_index, layer_type):
        self.layer_type = layer_type
        self.in_index   = in_index               # index of input node
        self.out_index  = out_index              # index of output node
        self.weight     = random.uniform(-1, 1)  # floating point [-1, 1]
        
    def __str__(self):
        weight = str(round(self.weight,12))
        if self.layer_type == "hidden":
            return ''.join(['Input', str(self.in_index),
                          ' <---------- \t' + weight + '\t --------> ',  
                          'Hidden', str(self.out_index)])
        elif self.layer_type == "output":
            return ''.join(['Hidden', str(self.in_index),
                            ' <--------- \t', weight, '\t --------> ',
                            'Output', str(self.out_index)])  
        elif self.layer_type == "hidden_bias":
            return ''.join(['HiddenBias', 
                            ' <------\t', weight, '\t --------> ',
                             'Hidden', str(self.out_index)])
        elif self.layer_type == "output_bias":
            return ''.join(['OutputBias',
                            ' <------\t' + weight + '\t --------> ',
                            'Output', str(self.out_index)])
