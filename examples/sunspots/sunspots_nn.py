"""
Lyle Scott III
lyle@digitalfoo.net
"""
import sys
from NNBackprop import NeuralNetwork as NN
from NNUtils import NNUtils 

TRAININGPATH = 'input/months-raw-training.dat'
MAX_EPOCHS = 4000
TEST_SAMPLE_N = 100

PLOTTING_ENABLED = True


def main():
    #N_INPUTS = 365
    #N_HIDDENS = 365
    N_INPUTS = 12
    N_HIDDENS = 12
    N_OUTPUTS = 1
    LEARNING_RATE = 0.15
    NAME = 'Example Sunspot Prediction Network'

    nnet = NN.NeuralNetwork(name=NAME)
    nnet.set_learning_rate(LEARNING_RATE)
    (all_vectors, scale_min, scale_max) = NNUtils.parse_windowed_training_file(
                                                               TRAININGPATH, 
                                                               N_INPUTS,  
                                                               N_OUTPUTS,
                                                               1)
    cut_index = len(all_vectors) - TEST_SAMPLE_N
    training_vectors = all_vectors[:cut_index]
    validation_vectors = all_vectors[cut_index:]
    nnet.set_validation_vectors(validation_vectors)
    nnet.set_training_vectors(training_vectors, scale_min, scale_max)
    nnet.create_network_architecture(n_inputs=N_INPUTS, n_hiddens=N_HIDDENS, 
                                     n_outputs=N_OUTPUTS)
    
    train(nnet, MAX_EPOCHS, plot=PLOTTING_ENABLED)

def train(nnet, max_epochs, test_epoch_modulo=10, plot=False):
    """step through the training process manually (since we want to pull some
    stats between epochs"""   
    i = 0
    for epoch in xrange(max_epochs):
        nnet.train()
        nnet.print_network_state()        
        NNUtils.save_nn_obj_to_file(nnet, '/tmp/network.pkl')
        
        if i % test_epoch_modulo == 0:
            (x1, x2,) = test()
            
            if plot != False:
                filepath = 'plots/%s.png' % nnet.epoch_i
                NNUtils.xy_prediction_plot(x1, x2, filepath=filepath)

def test():
    nnet = NNUtils.load_nn_obj_from_file('/tmp/network.pkl')
    
    scale_min, scale_max = nnet.get_scale_minmax()

    x1 = []
    x2 = []

    for sample_n in xrange(TEST_SAMPLE_N):
        inputs = nnet.get_inputs()
        outputs = nnet.get_outputs()

        x1.extend(outputs)
        x2.extend(nnet.get_validation_vectors()[sample_n][1])
        
        predicted_value = NNUtils.denormalize(outputs, scale_min, scale_max)[0]
        actual_value = NNUtils.denormalize(
                                nnet.get_validation_vectors()[sample_n][1],
                                scale_min, scale_max)[0]
        #print '%s,%s' % (predicted_value, actual_value,)

        inputs = inputs[len(outputs):]
        inputs.extend(nnet.get_outputs())
        nnet.set_inputs(inputs)
        nnet.feed_forward()

    return (x1, x2)


if __name__ == '__main__':
    main()