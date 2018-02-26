##
# @file   main_imdb.py
# @author Yibo Lin
# @date   Jan 2018
# @brief  modified from tflearn
#

# -*- coding: utf-8 -*-
"""
Simple example using LSTM recurrent neural network to classify IMDB
sentiment dataset.

References:
    - Long Short Term Memory, Sepp Hochreiter & Jurgen Schmidhuber, Neural
    Computation 9(8): 1735-1780, 1997.
    - Andrew L. Maas, Raymond E. Daly, Peter T. Pham, Dan Huang, Andrew Y. Ng,
    and Christopher Potts. (2011). Learning Word Vectors for Sentiment
    Analysis. The 49th Annual Meeting of the Association for Computational
    Linguistics (ACL 2011).

Links:
    - http://deeplearning.cs.cmu.edu/pdfs/Hochreiter97_lstm.pdf
    - http://ai.stanford.edu/~amaas/data/sentiment/

"""
from __future__ import division, print_function

import tflearn
from tflearn.data_utils import to_categorical, pad_sequences
from tflearn.datasets import imdb
import tensorflow as tf 

import Params
import os 
import sys 
import time 
import numpy as np 
import datetime
import glob 

class AdamDecay(tflearn.optimizers.Adam):
    """
    Adam with learning rate decay 
    """
    def __init__(self, learning_rate=0.001, beta1=0.9, beta2=0.999,
                 epsilon=1e-8, lr_decay=0., decay_step=1000,
                 use_locking=False, name="Adam"):
        super(AdamDecay, self).__init__(learning_rate, beta1, beta2, epsilon, use_locking, name)
        self.lr_decay = lr_decay
        if self.lr_decay > 0.:
            self.has_decay = True
        self.decay_step = decay_step

    def build(self, step_tensor=None):
        self.built = True
        if self.has_decay:
            if not step_tensor:
                raise Exception("Learning rate decay but no step_tensor "
                                "provided.")
            self.learning_rate = tf.train.exponential_decay(
                self.learning_rate, step_tensor,
                self.decay_step, self.lr_decay,
                staircase=True)
            tf.add_to_collection(tf.GraphKeys.LR_VARIABLES, self.learning_rate)
        self.tensor = tf.train.AdamOptimizer(
            learning_rate=self.learning_rate, beta1=self.beta1,
            beta2=self.beta2, epsilon=self.epsilon,
            use_locking=self.use_locking, name=self.name)

def train(params): 

    # fix random seed 
    np.random.seed(params.random_seed)
    # set random seed before build the graph 
    tf.set_random_seed(params.random_seed)

    # IMDB Dataset loading
    os.system("mkdir -p datasets")
    train, valid, test = imdb.load_data(path='datasets/imdb.pkl', n_words=10000,
                                    valid_portion=0)
    trainX, trainY = train
    validX, validY = valid
    testX, testY = test

    total_len = 0 
    max_len = 0
    for x in trainX: 
        total_len += len(x)
        max_len = max(max_len, len(x))
    print("average sequence length = ", total_len/len(trainX))
    print("max sequence length = ", max_len)

    maxlen = int(params.dataset[len("imdb."):])
    # Data preprocessing
    # Sequence padding
    trainX = pad_sequences(trainX, maxlen=maxlen, value=0.)
    validX = pad_sequences(validX, maxlen=maxlen, value=0.)
    testX = pad_sequences(testX, maxlen=maxlen, value=0.)
    # Converting labels to binary vectors
    trainY = to_categorical(trainY)
    validY = to_categorical(validY)
    testY = to_categorical(testY)

    print("trainX.shape = ", len(trainX))
    print("trainY.shape = ", len(trainY))
    print("testX.shape = ", len(testX))
    print("testY.shape = ", len(testY))

    params.time_steps = maxlen
    params.input_size = 1
    params.output_size = 2
    params.regression_flag = False 
    #params.freqs = np.logspace(np.log2(0.25), np.log2(params.time_steps/3), 5-1, base=2).tolist()
    #params.freqs.append(0.0)
    #params.freqs.sort()
    #params.freqs = np.linspace(0, maxlen, 40).tolist()

    # Network building
    net = tflearn.input_data([None, maxlen])
    net = tflearn.embedding(net, input_dim=10000, output_dim=params.num_units)

    if params.cell == "RNN": 
        net = tflearn.simple_rnn(net, n_units=params.num_units, dropout=params.dropout_keep_rate)
    elif params.cell == "LSTM": 
        net = tflearn.lstm(net, n_units=params.num_units, dropout=params.dropout_keep_rate)
    elif params.cell == "SRU": 
        params.phi_size = 200 # only for SRU 
        net = tflearn.sru(net, 
                num_stats=params.phi_size, 
                mavg_alphas=tf.get_variable('alphas', initializer=tf.constant(params.alphas), trainable=False), 
                output_dims=params.num_units, 
                recur_dims=params.r_size, 
                dropout=params.dropout_keep_rate
                )
    elif params.cell == "FRU": 
        params.phi_size = 10 # only for FRU 
        net = tflearn.fru(net, 
                num_stats=params.phi_size, 
                freqs=params.freqs, 
                freqs_mask=params.freqs_mask, 
                seq_len=params.time_steps, 
                output_dims=params.num_units, 
                recur_dims=params.r_size, 
                dropout=params.dropout_keep_rate
                )
    else:
        assert 0, "unsupported cell %s" % (params.cell)
    net = tflearn.fully_connected(net, 2, activation='softmax')
    net = tflearn.regression(net, optimizer=AdamDecay(learning_rate=params.initial_learning_rate, lr_decay=params.lr_decay, decay_step=1000), learning_rate=params.initial_learning_rate,
                             loss='categorical_crossentropy')

    print("parameters = ", params)

    print("trainable_variables = ", '\n'.join([str(v) for v in tf.trainable_variables()]))
    # summarize #vars 
    num_vars = 0 
    for var in tf.trainable_variables(): 
        num = 1
        for dim in var.get_shape(): 
            num *= dim.value 
        num_vars += num 
    print("# trainable_variables = ", num_vars)

    # Training
    best_checkpoint_path = None
    if validX: 
        assert 0 
        dt = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
        best_checkpoint_path = "results/%s/" % (dt)
        os.mkdir(best_checkpoint_path)
        validation_set = (validX, validY)
    else:
        validation_set = (testX, testY)
    model = tflearn.DNN(net, tensorboard_verbose=0, best_checkpoint_path=best_checkpoint_path)
    model.fit(trainX, trainY, validation_set=validation_set, n_epoch=params.num_epochs, show_metric=True,
              batch_size=params.batch_size, snapshot_step=100, validation_batch_size=params.batch_size*4)
    if validX: 
        # load best checkpoint 
        best_checkpoint = None
        with open(best_checkpoint_path+"/checkpoint", "r") as f: 
            for line in f: 
                line = line.strip()
                if line.startswith("model_checkpoint_path:"):
                    best_checkpoint = line.split()[1].strip()[1:-1]
                    break 
        print("best_checkpoint = ", best_checkpoint)
        model.load(best_checkpoint)
        print("best validation = ", model.evaluate(validX, validY))
    print("final evaluation = ", model.evaluate(testX, testY))

if __name__=='__main__':
    if len(sys.argv) < 2:
        print("input parameters in json format in required")
        exit() 
    paramsArray = []
    for i in range(1, len(sys.argv)):
        params = Params.Params()
        params.load(sys.argv[i])
        paramsArray.append(params)
    print("parameters[%d] = %s" % (len(paramsArray), paramsArray))

    tt = time.time()
    for params in paramsArray: 
        train(params)
    print("program takes %.3f seconds" % (time.time()-tt))
