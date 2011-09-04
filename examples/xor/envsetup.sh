# Lyle Scott III
# lyle@digitalfoo.net
#
# Setup a PYTHONPATH that will be aware of the NeuralNetwork's modules.
# (this is not needed if it was installed properly)

curdir="`pwd`"
export PYTHONPATH="$curdir/../../src:$curdir/../../src/NNBackprop"
