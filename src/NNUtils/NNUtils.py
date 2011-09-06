"""
Author : Lyle Scott III    lyle@digitalfoo.net
Date   : 2011-03-26
 
Copyright (c) 2011, Lyle Scott III
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:
    * Redistributions of source code must retain the above copyright
      notice, this list of conditions and the following disclaimer.
    * Redistributions in binary form must reproduce the above copyright
      notice, this list of conditions and the following disclaimer in the
      documentation and/or other materials provided with the distribution.
    * Neither the name of the <organization> nor the
      names of its contributors may be used to endorse or promote products
      derived from this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL <COPYRIGHT HOLDER> BE LIABLE FOR ANY
DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
(INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
"""
import cPickle
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import re
import sys
import time


# generate a range of floating point values
def frange(start, stop, n):
    L = [0.0] * n
    nm1 = n - 1
    nm1inv = 1.0 / nm1
    for i in xrange(n):
        L[i] = nm1inv * (start*(nm1 - i) + stop*i)
    return L      

def parse_training_file(filepath, n_inputs, n_outputs, delimiter=r'\s+'):
    vectors = []
    datum = []
    fp = open(filepath)
    for line in fp.readlines():
        # EOF
        if not line: 
            break     
        # strip \n, \r, \t, spaces, etc or skip comment
        if not line.strip() or line[0] == '#': 
            continue
        datapoints = map(int, re.split(delimiter, line.strip()))
        datum.extend(datapoints)

        vectors.append([datapoints[0:n_inputs], 
                        datapoints[n_inputs:]])
    fp.close()

    min_n, max_n = normalize(datum)

    return (vectors, min_n, max_n,)
              
def parse_windowed_training_file(path, input_win_size, 
                                 output_win_size, step_size, column_number=0):
    lines = []
    columns = []
    fp = open(path)
    for line in fp:
        if not line:         break     # EOF
        if not line.strip(): continue  # strip \n, \r, \t, spaces, etc
        if line[0] == "#":   continue  # skip a comment
        datapoint = float(line.rstrip().split()[column_number]) 
        lines.append(datapoint)
        
    if input_win_size > len(lines):
        print ''.join(['ERROR, windows_size (', str(input_win_size),
                       ') > number of data points (', str(len(lines)), ')'])
        sys.exit(-1)
    
    minn, maxn = normalize(lines)
    
    # put into an array of input/output vector pairs
    # ie  [ [i0, i1, ..., iN] , [o0, o1, ..., iN] ]
    vectors = []
    for line_i in xrange(0, len(lines)-input_win_size-output_win_size+1, step_size):
        input_vector  = [lines[line_i+i] for i in xrange(input_win_size)]
        output_vector = [lines[line_i+input_win_size+i] for i in xrange(output_win_size)]
        vectors.append([input_vector, output_vector])
    print '###\n### Parsed %s vectors from %s\n###' % (len(vectors), path)
    return (vectors, minn, maxn)
        
# put all values between [0, 1]
def normalize(vectors):
    minn = min(vectors)
    maxn = max(vectors)
    for i in xrange(len(vectors)):
        vectors[i] = (vectors[i] - minn) / (maxn - minn)
    return (minn, maxn)    
        
# take values out of [0, 1] range and restore original value
def denormalize(vectors, minn, maxn): 
    for i in xrange(len(vectors)):
        vectors[i] = vectors[i] * (maxn - minn) + minn
    return vectors
        
def load_testing_file(path, column_number=0):
    # put all lines into an array (lines) for easy access
    datapoints = []
    fp = open(path)
    for line in fp:
        if not line:         break     # EOF
        if not line.strip(): continue  # strip \n, \r, \t, spaces, etc
        if line[0] == "#":   continue  # skip a comment
        datapoints.append(float(line.rstrip().split()[column_number]))
    fp.close()
    return datapoints

def line_mse(observed_points, actual_points):
    if len(observed_points) != len(actual_points):
        print "ERROR, can't take MSE if observed != actual number of points"
        sys.exit()
    mse = 0.0
    for i in xrange(len(observed_points)):
        mse += (actual_points[i] - observed_points[i][0]) ** 2
    return mse / len(observed_points)

def prediction_analysis(dir):
    DIR_NAME = "../saved_nn/" + dir
    
    n_predictions   = 100
    testing_data    = load_testing_file("input/monthssn.PAPER.testing.txt", 
                                        n_predictions)

    for subdir,dirs,files in os.walk(DIR_NAME):
        for file in sorted(files):
            if str(file).endswith(".txt") == False:
                continue
            
            # where the NN left off after training
            NN      = NeuralNetwork(load_path=DIR_NAME+"/"+file)
            try:
                inputs  = NN.get_inputs()
                outputs = NN.get_outputs()
            except:
                print "exiting..."
                continue
                        
            predicted_values = []
            
            ep = str(NN.epochs).zfill(4)
            
            predicted_file = open(DIR_NAME + "/predictions_" + ep + ".txt", "w")
            
            for prediction in xrange(n_predictions):
                for i in xrange(len(inputs)):
                    try:
                        inputs[i] = inputs[i+1]
                        
                    except:
                        inputs[i] = NN.get_outputs()[0]
                NN.set_inputs(inputs)
                NN.feed_forward()
                denormalized = denormalize(NN.get_outputs())
                predicted_values.append(denormalized)
                
                for value in denormalized:
                    predicted_file.write(str(value) + "\n")

            predicted_file.close()
        
            
            img_name = DIR_NAME+ "/" + ep + "epochs__" + str(Decimal(str(round(NN.mse, 9)))) + "mse__" + str(Decimal(str(round(NN.mse, 9)))) + "tsse__" + str(NN.window_size) + "ws.png"
        
            if os.path.exists(img_name):
                print "exists (skipping): " + img_name
                continue
        
            print "saving figure " + str(NN.epochs)
            plt.figure(NN.epochs)
            plt.plot(range(len(testing_data[1])), testing_data[1], label="Real (Smooth)", linewidth=4, color="blue")
            plt.plot(range(len(predicted_values)), predicted_values, label="Prediction")
            plt.title('Real vs Prediction of Monthly Sunspots  (lr=' + str(NN.learning_rate) + '\n' \
                'epoch=' + str(NN.epochs) + ', mse=' + str(round(NN.mse, 8)) + ", tsse=" + str(round(NN.tsse, 8)))    
            plt.xlabel('Months Ahead From Now')
            plt.ylabel('Monthly Sunspot Count')
            plt.ylim((0, 200))
            plt.legend()
            plt.savefig(img_name)
            
        print "DONE!"
        print os.path.abspath(DIR_NAME)    

def save_nn_obj_to_file(self, obj, path, epoch='default'):
    fp = open('%s/%s.pkl' % (path, epoch) , 'wb')
    cPickle.dump(self, fp)
    fp.close()
    
def load_nn_obj_from_file(self, path, epoch='default'):
    fp = open('%s/%s.pkl' % (path, epoch) , 'rb')
    obj = cPickle.load(fp)
    fp.close()
    return obj 
