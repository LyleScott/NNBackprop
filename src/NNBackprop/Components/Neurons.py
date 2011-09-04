"""
Author : Lyle Scott III    lyle@digitalfoo.net
Date   : 2011-03-26
"""

#class Neuron():
#    def __init__(self, index, value=None):
#        pass


class InputNeuron():
    def __init__(self, index, value=None):
        self.index = index
        self.name = 'Input%s' % index
        self.value = value
        self.layer_type = 'input'
        
    def __str__(self):
        return ''.join([str(self.name).ljust(10), 
                        '     value=' + str(self.value).ljust(15)])


class HiddenNeuron():
    def __init__(self, index, value=0.0, error_signal=None):
        self.index = index
        self.name = 'Hidden%s' % index
        self.value = value
        self.error_signal = error_signal
        self.layer_type = 'hidden'
        self.input_links  = []
        self.output_links = []
            
    def __str__(self):
        return ''.join([str(self.name).ljust(10),
                        '    value=', str(self.value).ljust(15),
                        '    error_signal=', str(self.error_signal).ljust(15)])


class OutputNeuron():
    def __init__(self, index, value=0.0, error_signal=0.0, expected_value=None):
        self.index          = index
        self.name           = 'Output%s' % index
        self.value          = value
        self.expected_value = expected_value
        self.error_signal   = error_signal
        self.layer_type     = 'output'
                
    def __str__(self):
        return ''.join([str(self.name).ljust(10),
                       '    value=', str(round(self.value, 12)).ljust(15),
                       '    error_signal=', 
                       str(round(self.error_signal, 12)).ljust(15),
                       '    expected_value=', 
                       str(round(self.expected_value, 12))]) 


class BiasNeuron():
    def __init__(self, layer_type, value=1.0):
        self.value      = value
        self.layer_type = layer_type
    
    def __str__(self):
        return ''.join([str("BiasNeuron").ljust(10),
                       '     layer=', self.layer_type,
                       '     value=', str(self.value).ljust(15)])
