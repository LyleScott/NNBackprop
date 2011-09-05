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
    def __init__(self, name=None, n_inputs=0, n_hiddens=0, n_outputs=0,
                 learn_rate=.35, momentum=1.0, 
                 network_state_file=None, print_flags=('all'), 
                 save_network=False):
        """initialization routine for the NeuralNetwork object"""
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

    def create_network_architecture(self, n_inputs, n_hiddens, n_outputs):  
        """create a NeuralNetwork object with a specific architecture"""

        # lists of stuff we will need
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

    """
    def import_training_file(self, filepath, delimiter='\t', col_num=0, 
                             moving_window_input_size=None,
                             moving_window_output_size=None,
                             moving_window_step_size=1):

        self.training_file_path = filepath
        if not moving_window_input_size or not moving_window_output_size:
            moving_window_input_size = len(self.inputs)
            moving_window_output_size = len(self.outputs)
        (self.training_vectors, self.scale_min, self.scale_max) = (
            NNUtils.parse_windowed_training_file(
                filepath, input_win_size=moving_window_input_size, 
                output_win_size=moving_window_output_size, 
                step_size=moving_window_step_size, 
                column_number=col_num))
    """

    def set_training_vectors(self, vectors, scale_min, scale_max):
        """set the training vectors for the network"""
        self.training_vectors = vectors
        self.set_scale_minmax(scale_min, scale_max)

    def get_training_vectors(self):
        """get all of the training vectors for the network"""
        return self.training_vectors
    
    def get_architecture(self):
        """return a simple representation of the network architecture"""
        return (len(self.inputs), len(self.hiddens), len(self.outputs))
            
    def set_inputs(self, input_vector):
        """set input node values"""
        for i in xrange(len(input_vector)):
            self.inputs[i].value = float(input_vector[i])
    
    def set_outputs(self, output_vector):
        """set output node values"""
        for i in xrange(len(output_vector)):
            self.outputs[i].value = float(output_vector[i])
    
    def set_expected_outputs(self, output_vector):
        """set output node expected values"""
        for i in xrange(len(output_vector)):
            self.outputs[i].expected_value = float(output_vector[i])
        
    def set_learning_rate(self, learn_rate):
        """set the learning rate of the network"""
        self.learn_rate = learn_rate    
    
    def set_momentum(self, momentum):
        """set the momentum of the network"""
        self.momentum = momentum
        
    def set_scale_minmax(self, scale_min, scale_max):
        """set the min/max scale values"""
        self.set_scale_min(scale_min)
        self.set_scale_max(scale_max)
        
    def set_scale_min(self, scale_min):
        """set the min scale value"""
        self.scale_min = scale_min
        
    def set_scale_max(self, scale_max):
        """set the max scale value"""
        self.scale_max = scale_max
     
    def training_loop(self, max_epochs=None, mse_threshold=None, 
                      rmse_threshold=None, tsse_threshold=None, 
                      print_network_state=False):
        """
        present correct patterns until network error is below threshold training
        """
        self.start_time = time.time()
        self.epoch_i = 0

        # loop forever until a threshold has been met
        while True:

            # check if max epochs threshold has been reached
            if max_epochs:
                if not hasattr(self, 'max_epochs'):
                    self.max_epochs = max_epochs
                else:
                    if self.epoch_i >= self.max_epochs:
                        break 

            # check if an error threshold has been reached
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

            # present each training vector to the network and back propagate
            # weights and error signals
            for training_vector in self.training_vectors:
                # set network's inputs to inputs provided by a training vector
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
        """convenience function for checking if a key is in the set of 
        printable flags
        """
        if key in self.print_flags or 'all' in self.print_flags:
            return True
        return False
        
    def print_network_state(self):
        """print the current network state in Human readable form"""
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
        """parse the time delta for run time into something meaningful"""
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
        
    def feed_forward(self):
        """
        produce outputs of hidden and output layer neurons

        The output values for a neuron are calculated by summing up all of the 
        pre-synaptic neuron's values multiplied by that link's weight.

        This summation is then ran through the activation function inorder to
        get value ... 

        sum = i0*w0 + i1*w1 + ... + iN*wN + LayerBias*wB
        sum = activation_function(sum)
        """
        self._ff_calc_hidden_layer()
        self._ff_calc_output_layer()

    def _ff_calc_hidden_layer(self):
        """
        update all hidden neuron values with all of the input node/link 
        contributions
        """
        for hidden_neuron in self.hiddens:
            # value from Bias
            sum = (self.hidden_bias_neuron.value 
                   * self.hidden_bias_links[hidden_neuron.index].weight)
            # summation of input node values and their corresponding weights
            for input_neuron in self.inputs:
                sum += (input_neuron.value 
                        * hidden_neuron.input_links[input_neuron.index].weight)
            # set the hidden neuron's value
            self.hiddens[hidden_neuron.index].value = self._activation_function(sum)

    def _ff_calc_output_layer(self):
        """
        update all output neuron values with all of the hidden node/link
        contributions
        """
        for output_neuron in self.outputs:     
            # value from Bias
            sum = (self.output_bias_neuron.value 
                   * self.output_bias_links[output_neuron.index].weight)
            # summation of hidden node values and their corresponding weights
            for hidden_neuron in self.hiddens:
                sum += (hidden_neuron.value 
                        * hidden_neuron.output_links[output_neuron.index].weight)
            # set the output neuron's value
            self.outputs[output_neuron.index].value = self._activation_function(sum)
            
    def backpropagate_errors(self):       
        """
        Send error signals from output neurons back to input links.
        Adjust all weights according to new error signals / values.
        """
        self._bp_calc_output_error_signals()
        self._bp_calc_hidden_error_signals()
        self._bp_calc_output_to_hidden_weights()
        self._bp_calc_hidden_to_input_weights()

    def _bp_calc_output_error_signals(self):
        """
        update output neuron error signals
        
        SIGMA = (ExpectedOutput - OutputValue) * 
                 OutputValue * 
                 (1 - OutputValue)
        """
        for output_neuron in self.outputs:
            self.outputs[output_neuron.index].error_signal = (
                (output_neuron.expected_value - output_neuron.value) 
                 * output_neuron.value 
                 * (1.0 - output_neuron.value))

    def _bp_calc_hidden_error_signals(self):
        """
        update hidden neuron error signals

        FORMULA:
        sigma = HiddenValue * 
                (1 - HiddenValue) * 
                summation {all post-synaptic neurons error signals * 
                           weight of each corresponding link} 
        """
        for hidden_neuron in self.hiddens:
            sum = 0.0
            for output_neuron in self.outputs:
                sum += (output_neuron.error_signal 
                        * hidden_neuron.output_links[output_neuron.index].weight)
            self.hiddens[hidden_neuron.index].error_signal = (
                hidden_neuron.value * (1.0 - hidden_neuron.value) * sum)
        
    def _bp_calc_output_to_hidden_weights(self):
        """
        update weights on links between output and hidden layer
    
        NOTE: must be done ONLY after all error values have been calculated

        FORMULA:
        weight += learn_rate * 
                  error signal of post-syntaptic neuron * 
                  value of current neuron)
        """
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
        
    def _bp_calc_hidden_to_input_weights(self):
        """
        update weights between input and hidden layer

        NOTE: must be done ONLY after all error values have been calculated

        FORMULA:
        weight += learn_rate * 
                  error signal of post-syntaptic neuron * 
                  value of current neuron)
        """
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
    
    def get_tsse(self):
        """Total Sum Squared Error"""
        tsse = 0.0
        for output_neuron in self.outputs:
            if not output_neuron.expected_value and not output_neuron.value:
                return None
            difference = output_neuron.expected_value - output_neuron.value
            tsse += math.pow(difference, 2)  
        return tsse * 0.5
        
    def get_mse(self):
        """Mean Squared Error"""
        mse = 0.0
        for output_neuron in self.outputs:
            if not output_neuron.expected_value and not output_neuron.value:
                return None
            difference = output_neuron.expected_value - output_neuron.value
            mse += difference ** 2
        return mse / len(self.outputs)
    
    def get_rmse(self):
        """Root Mean Squared Error"""
        try:
            rmse = ((2 * self.get_tsse()) / 
                    (len(self.training_vectors) * len(self.outputs)))
        except TypeError:
            # tsse returned None...
            return None
        return math.sqrt(rmse)
        
    def _activation_function(self, net):
        """sigmoid function; squashes a value between 0..1"""
        return 1.0 / (1.0 + math.exp(-net))

    def get_inputs(self):
        """return list of input values"""
        return [input_neuron.value for input_neuron in self.inputs]

    def get_outputs(self):
        """return list of output values"""
        return [output_neuron.value for output_neuron in self.outputs]
    
    def set_name(self, name):
        """set name for the NeuralNetwork object"""
        if name == None:
            name = 'Unnamed Neural Network' 
        self.name = name

    def _in_network_state_flags(self, key):
        if (key in self.save_network_state_flags or 
            'all' in  self.save_network_state_flags):
            return True
        return False

    def save_network_state(self, filepath):
        """output the network state to a file"""
        now = str(datetime.datetime.now())
        progress = float(self.epoch_i) / float(self.max_epochs) * 100.0
        input_values = ','.join(['%s'%i.value for i in self.inputs])
        output_values = ','.join(['%s'%i.value for i in self.outputs])
        expected_values = ','.join(['%s'%i.expected_value 
                                    for i in self.outputs])
        fp = open(filepath, "w")
        fp.write('\n'.join(['date,%s\n' % now,
                            'name,%s\n' % self.name,
                            'learning_rate,%s\n' % self.learn_rate,
                            'momentum,', str(self.momentum), '\n',
                            'architecture,%s,%s,%s\n' % (len(self.inputs), 
                                                         len(self.hiddens), 
                                                         len(self.outputs)),
                            'scale_min,%s\n' % self.scale_min,
                            'scale_max,%s\n' % self.scale_max,
                            'tsse,%.12f\n' % self.get_tsse(),
                            'mse,%.12f\n' % self.get_mse(),
                            'rmse,%.12f\n' % self.get_rmse(),
                            'runtime,%sh,%sm,%ss' % self.parse_runtime(),
                            'epochs,%s,%s' % (self.epoch_i, self.max_epochs),
                            'progress,%s%%' % progress,
                            'input_values,%s\n' % input_values,             
                            'outputs_values,%s\n' % output_values,
                            'output_expected_values,%s\n' % expected_values,
                          ]))
        lines = []
        if self._in_network_state_flags('input2hidden_links'): 
            for hidden_neuron in self.hiddens:
                for hidden_link in hidden_neuron.input_links:
                    lines.append(','.join(['IH', str(hidden_link.in_index), 
                                           str(hidden_link.out_index),
                                           str(hidden_link.weight)]))
        if self._in_network_state_flags('hidden_bias_links'): 
            for hidden_bias_link in self.hidden_bias_links:
                lines.append(','.join(['HB', str(hidden_bias_link.out_index),
                                       str(hidden_bias_link.weight) ]))
        if self._in_network_state_flags('hidden2output_links'):
            for hidden_neuron in self.hiddens:
                for output_link in hidden_neuron.output_links:
                    lines.append(','.join(['HO', str(output_link.in_index),
                                           str(output_link.out_index),
                                           str(output_link.weight)]))
        if self._in_network_state_flags('output_bias_links'): 
            for output_bias_link in self.output_bias_links:
                lines.append(','.join(['OB', str(output_bias_link.out_index),
                                       str(output_bias_link.weight)]))
        fp.write('\n'.join(lines))
        fp.close()
    
    def load_network_state(self, filepath):
        """load an existing network state from a file"""
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
        fp.close()
