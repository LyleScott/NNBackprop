from NNBackprop import NeuralNetwork as NN
from NNUtils import NNUtils 


def main():
    nnet = NN.NeuralNetwork(name='Example XOR Network', 
                            n_inputs=2, n_hiddens=2, n_outputs=1)

    path = 'input.txt'
    (vectors, min_n, max_n) = NNUtils.parse_training_file(path, 2, 1)
    nnet.set_training_set(vectors)
    nnet.training_loop(tsse_threshold=.005)
    nnet.print_network_state()

if __name__ == '__main__':
    main()
