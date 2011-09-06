"""
Lyle Scott III
lyle@digitalfoo.net
"""
from NNBackprop import NeuralNetwork as NN
from NNUtils import NNUtils 


def main():
    path = 'input/training.txt'

    # create a simple 2x2x1 network.
    nnet = NN.NeuralNetwork(name='Example XOR Network')
    nnet.set_learning_rate(0.4)
    nnet.create_network_architecture(n_inputs=2, n_hiddens=2, n_outputs=1)

    # train the network
    (vectors, scale_min, scale_max) = NNUtils.parse_training_file(path, 2, 1)
    nnet.set_training_vectors(vectors, scale_min, scale_max)
    nnet.training_loop(tsse_threshold=.005)
    #nnet.training_loop(epochs_threshold=9999)

    print '-'*80

    # test the network
    # - input values that are NOT xor'ed should put the output value 'low'
    # - input values that ARE xor'ed should put the output value 'high'
    for vector in [[0,0], [0,1], [1,0], [1,1]]:
        nnet.set_inputs([vector[0], vector[1]])
        nnet.feed_forward()
        print 'NN test:', vector,  '==', nnet.get_outputs()[0]

    nnet.print_network_state()

    # plot the activation due to a bunch of numbers 0-1
    # 1) uncomment save_network_state and save_network_state_iteration_modulo 
    #    in the above constructor.
    # 2) uncomment the following 5 lines of code    
    """
    fp = open('activation.csv')
    for x in NNUtils.frange(0, 1, 100):
        for y in NNUtils.frange(0, 1, 100):
            nnet.set_inputs([x, y])
            nnet.feed_forward()
            fp.write('%s,%s,%s\n' % (x, y, nnet.get_outputs()[0],))
    fp.close()
    """

if __name__ == '__main__':
    main()
