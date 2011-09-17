"""
Lyle Scott III
lyle@digitalfoo.net
"""
from NNBackprop import NeuralNetwork as NN
from NNUtils import NNUtils 

def main():
    #trainingpath = 'input/days1-training.dat'
    trainingpath = 'input/months-raw-training.dat'

    #N_INPUTS = 365
    #N_HIDDENS = 365
    N_INPUTS = 12
    N_HIDDENS = 12
    N_OUTPUTS = 1

    nnet = NN.NeuralNetwork(name='Example Sunspot Prediction Network')
    nnet.set_learning_rate(0.15)
    (vectors, scale_min, scale_max) = NNUtils.parse_windowed_training_file(
                                                               trainingpath, 
                                                               N_INPUTS,  
                                                               N_OUTPUTS,
                                                               1)
    nnet.set_training_vectors(vectors, scale_min, scale_max)
    nnet.create_network_architecture(n_inputs=N_INPUTS, n_hiddens=N_HIDDENS, 
                                     n_outputs=N_OUTPUTS)
    nnet.training_loop(epochs_threshold=5, print_network_state=True)
    NNUtils.save_nn_obj_to_file(nnet, 'network.pkl')

    #nnet = NNUtils.load_nn_obj_from_file('network.pkl')
    print '-'*80
    print nnet

if __name__ == '__main__':
    main()