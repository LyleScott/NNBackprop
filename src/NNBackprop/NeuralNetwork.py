"""
Author : Lyle Scott III    lyle@digitalfoo.net
Date   : 2011-03-26
"""
import datetime
import math
import sys
import time

from NNBackprop.Components import Neurons
from NNBackprop.Components import Links
from NNUtils import NNUtils


class NeuralNetwork():
    def __init__(self, name=None, n_inputs=None, n_hiddens=0, n_outputs=0,
                 learn_rate=.35, momentum=1.0, training_set=None, 
                 training_file=None, network_state_file=None, 
                 print_flags=('all'), 
                 moving_window_size=None, moving_window_step_size=None,
                 save_network=False, save_network_state_flags=('all')):
        self.set_name(name)
        self.set_learning_rate(learn_rate)
        self.set_momentum(momentum)
        if n_inputs and n_hiddens and n_outputs: 
            self.create_network_architecture(n_inputs, n_hiddens, n_outputs)
        self.set_printable_flags(print_flags)
        self.save_network = save_network
        self.save_network_state_flags = save_network_state_flags
        self.scale_min = None
        self.scale_max = None
        self.tsse = 0.0
        self.mse = 0.0
        self.rmse = 0.0
        if network_state_file:
            self.load_network_state(network_state_file)
        if training_file:
            self.import_training_file(training_file)

    def create_network_architecture(self, n_inputs=None, n_hiddens=None, 
                                    n_outputs=None):  
        self.inputs             = []
        self.hiddens            = []
        self.outputs            = [] 
        self.hidden_bias_links  = [] 
        self.output_bias_links  = []
    
        # initialize a bias neuron for each layer
        self.hidden_bias_neuron = Neurons.BiasNeuron('hidden')
        self.output_bias_neuron = Neurons.BiasNeuron('output')
        # create neurons at each layer
        self.inputs  = [Neurons.InputNeuron(i)  for i in xrange(int(n_inputs))]
        self.hiddens = [Neurons.HiddenNeuron(i) for i in xrange(int(n_hiddens))]
        self.outputs = [Neurons.OutputNeuron(i) for i in xrange(int(n_outputs))]
          
        # create input<---->hidden neuron links
        for hidden_neuron in self.hiddens:
            for input_neuron in self.inputs:
                self.hiddens[hidden_neuron.index].input_links.append(
                    Links.NeuronLink(input_neuron.index, hidden_neuron.index, 
                               'hidden')
                )
            # create hidden_bias<---->hidden neuron links
            self.hidden_bias_links.append(
               Links.NeuronLink(None, hidden_neuron.index, 'hidden_bias')
            )
        
        # create hidden<---->output neuron links
        for hidden_neuron in self.hiddens:    
            for output_neuron in self.outputs:
                self.hiddens[hidden_neuron.index].output_links.append(
                    Links.NeuronLink(hidden_neuron.index, 
                               output_neuron.index, 'output')
                )

        # create output_bias<---->output neuron links
        for output_neuron in self.outputs:
            self.output_bias_links.append(
                Links.NeuronLink(None, output_neuron.index, 'output_bias')
            )
            
    def import_training_file(self, filepath, delimiter='\t', col_num=0, 
                             moving_window_input_size=None,
                             moving_window_output_size=None,
                             moving_window_step_size=1):

        self.training_file_path = filepath
        if not moving_window_input_size or not moving_window_output_size:
            moving_window_input_size = len(self.inputs)
            moving_window_output_size = len(self.outputs)
        (self.training_set, self.scale_min, self.scale_max) = (
            NNUtils.parse_windowed_training_file(
                filepath, input_win_size=moving_window_input_size, 
                output_win_size=moving_window_output_size, 
                step_size=moving_window_step_size, 
                column_number=col_num))

    def set_training_set(self, data):
        self.training_set = data

    def get_training_set(self):
        return self.training_set
    
    def get_architecture(self):
        return (len(self.inputs), len(self.hiddens), len(self.outputs))
            
    # set the input node(s) value to the vector elements provided
    def set_inputs(self, input_vector):
        for i in xrange(len(input_vector)):
            self.inputs[i].value = float(input_vector[i])
    
    # set the output node(s) value to the vector elements provided
    def set_outputs(self, output_vector):
        for i in xrange(len(output_vector)):
            self.outputs[i].value = float(output_vector[i])
    
    # set the output node(s) expected value to the vector elements provided
    def set_expected_outputs(self, output_vector):
        for i in xrange(len(output_vector)):
            self.outputs[i].expected_value = float(output_vector[i])
        
    def set_name(self, name):
        self.name = name
    
    def set_learning_rate(self, learn_rate):
        self.learn_rate = learn_rate    
    
    def set_momentum(self, momentum):
        self.momentum = momentum
        
    def set_scale_minmax(self, scale_min, scale_max):
        self.set_scale_min(scale_min)
        self.set_scale_max(scale_max)
        
    def set_scale_min(self, scale_min):
        self.scale_min = scale_min
        
    def set_scale_max(self, scale_max):
        self.scale_max = scale_max
     
    # present correct patterns until network error is below threshold training
    def training_loop(self, max_epochs=None, mse_threshold=None, 
                      rmse_threshold=None, tsse_threshold=None, 
                      print_network_state=False):
        self.start_time = time.time()
        self.epoch_i = 0
        print '\n###\n### Training NN with training set...\n###'
        while True:
            if max_epochs:
                if not hasattr(self, 'max_epochs'):
                    self.max_epochs = max_epochs
                else:
                    if self.epoch_i >= self.max_epochs:
                        break 
            if mse_threshold:
                mse = self.get_mse()
                if mse and mse < mse_threshold: 
                    break
            if tsse_threshold:
                tsse = self.get_tsse() 
                if tsse and tsse < tsse_threshold: 
                    break
            if rmse_threshold:
                rmse = self.get_rmse()
                if rmse and rmse < rmse_threshold: 
                    break
            for training_vector in self.training_set:
                # set network's inputs to inputs provides by a training vector
                self.set_inputs(training_vector[0])
                # set network's expected outputs to train against
                self.set_expected_outputs(training_vector[1])    
                # feed the inputs through to produce values at output neurons  
                self.feed_forward()            
                # propagate errors and weights (connection strengths) backwards
                self.backpropagate_errors()
            self.epoch_i += 1
            #NNUtils.save_obj_to_file(self, '.', self.epoch_i)
            #sys.exit()
            if print_network_state is True:
                self.print_network_state()
        #if self.save_network is True:
        #    self.save_network_state('network_state.csv')
        
    def set_printable_flags(self, *flags):
       self.print_flags = flags

    def _in_print_flags(self, key):
        if key in self.print_flags or 'all' in self.print_flags:
            return True
        return False
        
    def print_network_state(self):
        if self.print_flags:
            print "-" * 80
        if self._in_print_flags('name'):
            print 'name         : %s' % self.name
        if self._in_print_flags('architecture'):
            print 'architecture :', self.get_architecture()
        if self._in_print_flags('learn_rate'):
            print 'learn rate   : %s' % self.learn_rate
        if self._in_print_flags('tsse'):
            tsse = self.get_tsse()
            if tsse is not None:
                print 'TSSE         : %.12f' % tsse
            else:
                print 'TSSE         : None'
        if self._in_print_flags('mse'):
            mse = self.get_mse()
            if mse is not None:
                print 'MSE          : %.12f' % mse
            else:
                print 'MSE          : None'
        if self._in_print_flags('rmse'):
            rmse = self.get_rmse()
            if rmse is not None:
                print 'RMSE         : %.12f' % rmse
            else:
                print 'RMSE         : None'
        if self._in_print_flags('runtime'):
            runtime = self.parse_runtime()
            if runtime is not None:
                print 'runtime      : %sh %sm %ss' % runtime
            else:
                print 'runtime      : None'
        if self._in_print_flags('epochs'):
            if hasattr(self, 'max_epochs') and self.max_epochs:
                print 'epochs       : %s/%s' % (self.epoch_i, self.max_epochs)
            else:
                print 'epochs       : %s' % (self.epoch_i)
                
        if self._in_print_flags('progress'):
            if hasattr(self, 'max_epochs') and self.max_epochs:
                percent = float(self.epoch_i) / float(self.max_epochs) * 100.
                print 'progress     : %s%%' % percent
    
    def parse_runtime(self):
        try:
            runtime  = (time.time() - self.start_time)
            hrs      = int(runtime / (60*60))
            runtime -= hrs * 60 * 60
            mins     = int(runtime / 60)
            runtime -= mins * 60
            secs     = int(runtime)
            return (hrs, mins, secs)
        except:
            return None
        
    # produce outputs of hidden and output layer neurons
    def feed_forward(self):
        # The output values for a neuron is calculated by summing up all of the 
        # pre-synaptic neuron's values multiplied by that link's weight.
        #
        # This summation is then ran through the activation function inorder to
        # get value ... 
        #
        # sum = i0*w0 + i1*w1 + ... + iN*wN + LayerBias*wB
        # sum = activation_function(sum)
        
        # update hidden neuron values 
        for hidden_neuron in self.hiddens:
            sum = (self.hidden_bias_neuron.value 
                   * self.hidden_bias_links[hidden_neuron.index].weight)
            for input_neuron in self.inputs:
                sum += (input_neuron.value 
                        * hidden_neuron.input_links[input_neuron.index].weight)
            self.hiddens[hidden_neuron.index].value = self._activation_function(sum)

        # update output neuron values
        for output_neuron in self.outputs:     
            sum = (self.output_bias_neuron.value 
                   * self.output_bias_links[output_neuron.index].weight)
            for hidden_neuron in self.hiddens:
                sum += (hidden_neuron.value 
                        * hidden_neuron.output_links[output_neuron.index].weight)
            self.outputs[output_neuron.index].value = self._activation_function(sum)
            
    # send error signals from output neurons back to input links,
    # and adjust all weights according to new error signals / values
    def backpropagate_errors(self):       
        # update output neuron error signals
        # SIGMA = (ExpectedOutput - OutputValue) * OutputValue * (1 - OutputValue)
        for output_neuron in self.outputs:
            self.outputs[output_neuron.index].error_signal = (
                (output_neuron.expected_value - output_neuron.value) 
                * output_neuron.value 
                * (1.0 - output_neuron.value))

        # update hidden neuron error signals
        # SIGMA = HiddenValue * (1 - HiddenValue) * 
        #    sum {all post-synaptic neurons error signal * weight of link} 
        for hidden_neuron in self.hiddens:
            sum = 0.0
            for output_neuron in self.outputs:
                sum += (output_neuron.error_signal 
                       * hidden_neuron.output_links[output_neuron.index].weight)
            self.hiddens[hidden_neuron.index].error_signal = (
                hidden_neuron.value * (1.0 - hidden_neuron.value) * sum)

        # !!! NOTE !!!
        # weights are adjusted ONLY after all error values have been calculated
        # !!! NOTE !!!
        #
        # w += learn_rate * error signal of post-syntaptic neuron 
        #      * value of current neuron)
        
        # update weights between output and hidden layer
        for output_neuron in self.outputs:
            for hidden_neuron in self.hiddens:
                # update output <---> hidden weights
                self.hiddens[hidden_neuron.index].output_links[output_neuron.index].weight += (
                    (self.learn_rate * output_neuron.error_signal
                    * hidden_neuron.value) * self.momentum)
            # update output <---> output_layer_bias weights
            self.output_bias_links[output_neuron.index].weight += (
                self.learn_rate * output_neuron.error_signal 
                * self.output_bias_neuron.value)
        
        # update weights between input and hidden layer
        for hidden_neuron in self.hiddens:
            for input_neuron in self.inputs:
                # update hidden <---> input weights
                self.hiddens[hidden_neuron.index].input_links[input_neuron.index].weight += (
                    self.learn_rate * hidden_neuron.error_signal 
                    * input_neuron.value)
            # update hidden <---> hidden_layer_bias weights
            self.hidden_bias_links[hidden_neuron.index].weight += (
                self.learn_rate * hidden_neuron.error_signal 
                * self.hidden_bias_neuron.value)
    
    # Total Sum Squared Error
    def get_tsse(self):
        tsse = 0.0
        for output_neuron in self.outputs:
            if not output_neuron.expected_value and not output_neuron.value:
                return None
            difference = output_neuron.expected_value - output_neuron.value
            tsse += math.pow(difference, 2)  
        return tsse * 0.5
        
    # Mean Squared Error
    def get_mse(self):
        mse = 0.0
        for output_neuron in self.outputs:
            if not output_neuron.expected_value and not output_neuron.value:
                return None
            difference = output_neuron.expected_value - output_neuron.value
            mse += difference ** 2
        return mse / len(self.outputs)
    
    # Root Mean Squared Error
    def get_rmse(self):
        try:
            rmse = ((2 * self.get_tsse()) / 
                    (len(self.training_set) * len(self.outputs)))
        except TypeError:
            # tsse returned None...
            return None
        return math.sqrt(rmse)
        
    # sigmoid function; squashes a value between 0..1
    def _activation_function(self, net):
        return 1.0 / (1.0 + math.exp(-net))

    # return list of input values    
    def get_inputs(self):
        return [input_neuron.value for input_neuron in self.inputs]

    # return list of output values    
    def get_outputs(self):
        return [output_neuron.value for output_neuron in self.outputs]
    
    def set_name(self, name):
        self.name = name
    
    def save_network_state(self, filepath):
        now = str(datetime.datetime.now())
        file = open(filepath, "w")
        file.write(''.join(['date,', str(now), '\n',
                            'name,', self.name, '\n',
                            'learning_rate,',  str(self.learn_rate), '\n',
                            'momentum,', str(self.momentum), '\n',
                            'architecture,', str(len(self.inputs)), ',', 
                                str(len(self.hiddens)),  ',', 
                                str(len(self.outputs)), '\n',
                            'scale_min,', str(self.scale_min), '\n',
                            'scale_max,', str(self.scale_max), '\n',
                            ]))
        lines = []
        if ('tsse' in self.save_network_state_flags 
            or 'all' in self.save_network_state_flags):
            lines.append('tsse,%.12f' % self.get_tsse())
        if ('mse' in self.save_network_state_flags 
            or 'all' in self.save_network_state_flags):
            lines.append('mse,%.12f' % self.get_mse())
        if ('rmse' in self.save_network_state_flags 
            or 'all' in self.save_network_state_flags):
            lines.append('rmse,%.12f' % self.get_rmse())
        if ('runtime' in self.save_network_state_flags 
            or 'all' in self.save_network_state_flags):
            lines.append('runtime,%sh,%sm,%ss' % self.parse_runtime())
        if ('epochs' in self.save_network_state_flags 
            or 'all' in self.save_network_state_flags):
            lines.append('epochs,%s,%s' % (self.epoch_i, self.max_epochs))
        if ('progress' in self.save_network_state_flags 
            or 'all' in self.save_network_state_flags):
            percent = float(self.epoch_i) / float(self.max_epochs) * 100.
            lines.append('progress,%s%%' % percent)
        if ('input_values' in self.save_network_state_flags 
            or 'all' in self.save_network_state_flags):
            lines.append('input_values,' + 
                         ','.join(['%s'%i.value for i in self.inputs]))
        if ('output_values' in self.save_network_state_flags 
            or 'all' in self.save_network_state_flags):
            lines.append('outputs_values,' + 
                         ','.join(['%s'%i.value for i in self.outputs]))
        if ('output_expected_values' in self.save_network_state_flags
            or 'all' in self.save_network_state_flags):
            lines.append('output_expected_values,' +
                         ','.join(['%s'%i.expected_value for i in self.outputs]))
        if ('input2hidden_links' in self.save_network_state_flags 
            or 'all' in self.save_network_state_flags):
            for hidden_neuron in self.hiddens:
                for hidden_link in hidden_neuron.input_links:
                    lines.append(','.join(['IH', str(hidden_link.in_index), 
                                           str(hidden_link.out_index),
                                           str(hidden_link.weight)]))
        if ('hidden_bias_links' in self.save_network_state_flags 
            or 'all' in self.save_network_state_flags):
            for hidden_bias_link in self.hidden_bias_links:
                lines.append(','.join(['HB', str(hidden_bias_link.out_index),
                                       str(hidden_bias_link.weight) ]))
        if ('hidden2output_links' in self.save_network_state_flags 
            or 'all' in self.save_network_state_flags):
            for hidden_neuron in self.hiddens:
                for output_link in hidden_neuron.output_links:
                    lines.append(','.join(['HO', str(output_link.in_index),
                                           str(output_link.out_index),
                                           str(output_link.weight)]))
        if ('output_bias_links' in self.save_network_state_flags 
            or 'all' in self.save_network_state_flags):
            for output_bias_link in self.output_bias_links:
                lines.append(','.join(['OB', str(output_bias_link.out_index),
                                       str(output_bias_link.weight)]))
        if ('training_file_path' in self.save_network_state_flags 
            or 'all' in self.save_network_state_flags):
            if hasattr(self, 'training_file_path'):
                lines.append('training_file_path,%s' % self.training_file_path)
        file.write('\n'.join(lines))
        file.close()
    
    def load_network_state(self, filepath):
        fp = open(filepath)        
        for line in fp:
            if not line:         break     # EOF
            if not line.strip(): continue  # strip \n, \r, \t, spaces, etc
            if line[0] == "#":   continue  # skip a comment
            line = line.strip()
            if line.startswith('name,'):
                self.name = line.split(',')[1]
            elif line.startswith('architecture'):
                 ninputs, nhiddens, noutputs = line.split(',')[1:4]
                 self.create_network_architecture(ninputs, nhiddens, noutputs)
            elif line.startswith('learning_rate,'):
                self.set_learning_rate(line.split(',')[1])
            elif line.startswith('momentum,'):
                self.set_momentum(line.split(',')[1])
            elif line.startswith('scale_min,'):
                self.set_scale_min(line.split(',')[1])
            elif line.startswith('scale_max,'):
                self.set_scale_max(line.split(',')[1])
            elif line.startswith('runtime,'):
                hr,min,sec = line.split(',')[1:]
                self.runtime = (hr[:-1], min[:-1], sec[:-1])
            elif line.startswith('epochs,'):
                self.epoch_i = int(line.split(',')[1])
                self.max_epochs = int(line.split(',')[2])
            elif line.startswith('input_values,'):
                inputs = []
                for input in line.split(',')[1:]:
                    inputs.append(input)
                self.set_inputs(inputs)
            elif line.startswith('output_values,'):
                self.set_outputs(
                    [line.split(',')[i] for i in xrange(1, len(line.split(',')))])
            elif line.startswith('output_expected_values,'):
                self.set_expected_outputs(
                    [line.split(',')[i] for i in xrange(1, len(line.split(',')))])    
            elif line.startswith('IH,'):
                name,input_index,hidden_index,weight = line.split(',')
                input_index  = int(input_index)
                hidden_index = int(hidden_index)
                weight       = float(weight)
                self.hiddens[hidden_index].input_links[input_index].weight = weight
            elif line.startswith('HB,'):
                name,hidden_index,weight = line.split(",")
                hidden_index = int(hidden_index)
                weight       = float(weight)
                self.hidden_bias_links[hidden_index].weight = weight
            elif line.startswith('HO,'):
                name,hidden_index,output_index,weight = line.split(",")
                hidden_index = int(hidden_index)
                output_index = int(output_index)
                weight       = float(weight)
                self.hiddens[hidden_index].output_links[output_index].weight = weight
            elif line.startswith('OB,'):
                name,output_index,weight = line.split(",")
                output_index = int(output_index)
                weight       = float(weight)
                self.output_bias_links[output_index].weight = weight
            elif line.startswith('training_file_path,'):
                self.import_training_file(line.split(',')[1])
        fp.close()
        
