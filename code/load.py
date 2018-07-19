# -*- coding: utf-8 -*-
##
# @file   util.py
# @author Yibo Lin
# @date   Dec 2017
#

import os
import re 
import sys
import time
import math
import numpy as np
import csv
import pickle
import sklearn
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from sklearn.datasets import fetch_mldata
import urllib2
import Params

''' prepare dataset '''

def load_mnist(params):
    mnist = fetch_mldata('MNIST original')
    mnist_X, mnist_y = shuffle(mnist.data, mnist.target, random_state=params.random_seed)
    mnist_X = mnist_X / 255.0

    print("MNIST data prepared")

    mnist_X, mnist_y = mnist_X.astype('float32'), mnist_y.astype('int64')

    def flatten_img(images):
        '''
        images: shape => (n, rows, columns)
        output: shape => (n, rows*columns)
        '''
        n_rows    = images.shape[1]
        n_columns = images.shape[2]
        for num in range(n_rows):
            if num % 2 != 0:
                images[:, num, :] = images[:, num, :][:, ::-1]
        output = images.reshape(-1, n_rows*n_columns)
        return output

    time_steps = 28*28
    if params.dataset.startswith("mnist.permute"):
        print "permuate MNIST"
        mnist_X = mnist_X.reshape((-1, time_steps))
        perm = np.random.permutation(time_steps)
        for i in xrange(len(mnist_X)):
            mnist_X[i] = mnist_X[i][perm]
        if len(params.dataset) > len("mnist.permute."):
            time_steps = int(params.dataset[len("mnist.permute."):])
    else:
        if len(params.dataset) > len("mnist."): # mnist.xx
            time_steps = int(params.dataset[len("mnist."):])
    print "time_steps = ", time_steps
    mnist_X = mnist_X.reshape((-1, time_steps, 28*28/time_steps))
    #mnist_X = flatten_img(mnist_X) # X.shape => (n_samples, seq_len)
    print "mnist_X.shape = ", mnist_X.shape
    #mnist_X = mnist_X[:, :, np.newaxis] # X.shape => (n_samples, seq_len, n_features)
    mnist_y_one_hot = np.zeros((mnist_y.shape[0], 10))
    for i in xrange(len(mnist_y)):
        mnist_y_one_hot[i][mnist_y[i]] = 1
    print "mnist_y.shape = ", mnist_y_one_hot.shape

    # split to training and testing set 
    train_X, test_X, train_y, test_y = train_test_split(mnist_X, mnist_y_one_hot,
                                                        test_size=0.2,
                                                        random_state=params.random_seed)
    # need to set parameters according to dataset
    params.time_steps = train_X.shape[1]
    params.input_size = train_X.shape[2]
    params.output_size = 10
    params.regression_flag = False
    return train_X, test_X, train_y, test_y

# synthetic sine curves
def load_sine_synthetic(params):
    np.random.seed(params.random_seed)
    seq_len = 176; 
    x_axis = 2 * (np.arange(seq_len) - 0.5*seq_len) / seq_len
    dataX = []
    n_poly = 5
    nfreqs = 15
    freqs = np.random.randint(1, 30, size=nfreqs)*0.1 # frequency
    coefficients = np.random.uniform(-1, 1, size=(n_poly, nfreqs))
    phases = np.random.uniform(-1, 1, size=(n_poly, nfreqs))

    curves = np.zeros((n_poly, seq_len))
    for j in range(n_poly):
        for d in range(nfreqs):
            curves[j] += np.sin( 2 * np.pi * x_axis * freqs[d] + 2 * np.pi * phases[j, d]) * coefficients[j,d] 

    for i in range(4000):
        amps_noise = np.random.normal(0, 0.1, size=(n_poly,))
        bias_noise = np.random.normal(0, 0.1)
        x = bias_noise + np.dot(amps_noise, curves)
        # plot sequence
        if i < 5:
            try: 
                plt.plot(np.arange(seq_len), x)
                if i == 1:
                    plt.xlabel('time step')
                    plt.ylabel('sequence value')
                    plt.savefig("log/sin_nfreq%d_npoly%d.png" % (nfreqs, n_poly))
            except:
                print "failed to plot"
        dataX.append(x)
    dataX = np.array(dataX).reshape((-1, seq_len, 1))
    dataY = dataX[:, 1:, :].reshape((-1, seq_len-1))

    train_X, test_X, train_y, test_y = train_test_split(dataX, dataY,
                                                        test_size=0.2,
                                                        random_state=params.random_seed)

    # need to set parameters according to dataset
    params.time_steps = train_X.shape[1]
    params.input_size = train_X.shape[2]
    params.output_size = train_y.shape[1]
    params.regression_flag = True
    return train_X, test_X, train_y, test_y

# synthetic polynomial curves
def load_poly_synthetic(params):
    np.random.seed(params.random_seed)
    seq_len = 176; T=1.0
    x_axis = 2 * T*(np.arange(seq_len) - 0.5*seq_len) / seq_len
    dataX = []
    parse = re.search(r"poly_synthetic.(\d+).(\d+)", params.dataset)
    degree = int(parse.group(1))
    n_poly = int(parse.group(2))
    degrees = np.arange(1,degree+1)
    coefficients = np.random.uniform(-1, 1, size=(n_poly,degree))

    print coefficients

    curves = np.zeros((n_poly, seq_len))
    for j in range(n_poly):
        for d in range(degree):
            #try:
            #    plt.plot(np.arange(seq_len), curves[j])
            #except:
            #    print "failed to plot orginal curve"
            curves[j] += np.power( x_axis, degrees[d]) * coefficients[j,d] / (np.power(T,degrees[d]))

    #plt.show()

    for i in range(4000):
        amps_noise = np.random.normal(0, 0.1, size=(n_poly,))
        bias_noise = np.random.normal(0, 0.1)
        x = bias_noise + np.dot(amps_noise, curves)
        # plot sequence
        if i < 5:
            try:
                plt.plot(np.arange(seq_len), x)
                if i == 1:
                    plt.xlabel('time step')
                    plt.ylabel('sequence value')
                    plt.savefig("log/poly%d.png" % (n_poly))
            except:
                print "failed to plot orginal curve"
        dataX.append(x)
    #plt.show()
    #exit()
    dataX = np.array(dataX).reshape((-1, seq_len, 1))
    dataY = dataX[:, 1:, :].reshape((-1, seq_len-1))

    train_X, test_X, train_y, test_y = train_test_split(dataX, dataY,
                                                        test_size=0.2,
                                                        random_state=params.random_seed)

    # need to set parameters according to dataset
    params.time_steps = train_X.shape[1]
    params.input_size = train_X.shape[2]
    params.output_size = train_y.shape[1]
    params.regression_flag = True
    return train_X, test_X, train_y, test_y
