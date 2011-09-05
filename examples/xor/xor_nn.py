"""
Lyle Scott III
lyle@digitalfoo.net
"""

from NNBackprop import NeuralNetwork as NN
from NNUtils import NNUtils 


def main():
    path = 'input/training.txt'

    nnet = NN.NeuralNetwork(name='Example XOR Network')
    nnet.create_network_architecture(n_inputs=2, n_hiddens=2, n_outputs=1)

    # train the network
    (vectors, scale_min, scale_max) = NNUtils.parse_training_file(path, 2, 1)
    nnet.set_training_vectors(vectors, scale_min, scale_max)
    nnet.training_loop(tsse_threshold=.005)

    # test the network
    # - input values that are NOT xor'ed should put the output value 'low'
    # - input values that ARE xor'ed should put the output value 'high'
    for vector in [[0,0], [0,1], [1,0], [1,1]]:
        nnet.set_inputs([vector[0], vector[1]])
        nnet.feed_forward()
        print 'NN output:', nnet.get_outputs()[0]



if __name__ == '__main__':
    main()
