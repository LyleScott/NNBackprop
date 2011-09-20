"""
Lyle Scott III
lyle@digitalfoo.net
"""
from datetime import datetime
import os
import sys
from NNBackprop import NeuralNetwork as NN
from NNUtils import NNUtils 

#TRAININGPATH = 'input/months-raw-training.dat'
TRAININGPATH = 'input/months-smoothed-training.dat'
MAX_EPOCHS = 3000
TEST_SAMPLE_N = 60 

# to view the series of PNG plots that gen generated as a video, install 
# ImageMagic and do 'animate -delay 20 /path/to/pngs/*.png'
PLOTTING_ENABLED = True

suffix = datetime.now().strftime('%Y-%m-%d_%H:%m:%S')

def main():
    #N_INPUTS = 365
    #N_HIDDENS = 365
    N_INPUTS = 12
    N_HIDDENS = 12
    N_OUTPUTS = 1
    LEARNING_RATE = 0.1
    NAME = 'Example Sunspot Prediction Network'

    nnet = NN.NeuralNetwork(name=NAME)
    nnet.set_learning_rate(LEARNING_RATE)
    (all_vectors, scale_min, scale_max) = NNUtils.parse_windowed_training_file(
                                                               TRAININGPATH, 
                                                               N_INPUTS,  
                                                               N_OUTPUTS,
                                                               1)
    # take TEST_SAMPLE_N samples from the training set and use this set to 
    # measure the neural network's predictions against factual values.
    cut_index = len(all_vectors) - TEST_SAMPLE_N
    training_vectors = all_vectors[:cut_index]
    validation_vectors = all_vectors[cut_index:]
    nnet.set_validation_vectors(validation_vectors)
    nnet.set_training_vectors(training_vectors, scale_min, scale_max)
    nnet.create_network_architecture(n_inputs=N_INPUTS, n_hiddens=N_HIDDENS, 
                                     n_outputs=N_OUTPUTS)
    
    train(nnet, MAX_EPOCHS, plot=PLOTTING_ENABLED, test_epoch_modulo=24)

def train(nnet, max_epochs, test_epoch_modulo=10, plot=False):
    """step through the training process manually (since we want to pull some
    stats between epochs"""   
    i = 0
    for epoch in xrange(max_epochs):
        nnet.train()   
        NNUtils.save_nn_obj_to_file(nnet, './tmpnetwork.pkl')
        
        if i % test_epoch_modulo == 0:
            nnet.print_network_state()
            (estimated, actual,) = test()
            
            if plot != False:
                directory = 'plots.%s' % suffix
                
                if not os.path.exists(directory):
                    os.mkdir(directory)
                
                filepath = '%s/%s.png' % (directory, str(nnet.epoch_i).zfill(6))
                title = '%s\nepoch %s' % (nnet.name, i)
                
                NNUtils.xy_prediction_plot(actual, estimated, filepath=filepath, 
                                           title=title,
                                           x1_label='Actual Values',
                                           x2_label='Predicted Values',
                                           x1_axis_label='months into the future', 
                                           x2_axis_label='monthly sunspot count',
                                           y_minmax=(0,90))
        i += 1

def test():
    nnet = NNUtils.load_nn_obj_from_file('./tmpnetwork.pkl')
    
    scale_min, scale_max = nnet.get_scale_minmax()

    x1 = []
    x2 = []

    for sample_n in xrange(TEST_SAMPLE_N):
        inputs = nnet.get_inputs()
        outputs = nnet.get_outputs()

        x1.extend(NNUtils.denormalize(outputs, scale_min, scale_max))
        validations = nnet.get_validation_vectors()[sample_n][1]
        x2.extend(NNUtils.denormalize(validations, scale_min, scale_max))
        
        actual_value = NNUtils.denormalize(
                                nnet.get_validation_vectors()[sample_n][1],
                                scale_min, scale_max)[0]
                                
        #predicted_value = NNUtils.denormalize(outputs, scale_min, scale_max)[0]
        #print '%s,%s' % (predicted_value, actual_value,)

        inputs = inputs[len(outputs):]
        inputs.extend(nnet.get_outputs())
        nnet.set_inputs(inputs)
        nnet.feed_forward()

    return (x1, x2)


if __name__ == '__main__':
    main()