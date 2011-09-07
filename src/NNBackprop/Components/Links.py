"""
Author : Lyle Scott III    lyle@digitalfoo.net
Date   : 2011-03-26
"""
import math
import random


class NeuronLink():
    def __init__(self, in_index, out_index, layer_type, n_outputs=-1):
        self.layer_type = layer_type
        self.in_index   = in_index
        self.out_index  = out_index
        
        if n_outputs == -1:
            self.weight = random.uniform(-.5, .5)
        else:
            n = 1.0 / math.sqrt(n_outputs)
            self.weight = random.uniform(-n, n)
        
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
