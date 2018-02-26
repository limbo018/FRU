# -*- coding: utf-8 -*-
""" Script for running RNNs with fixed parameters. """

import os
import sys 
import time
import math
import numpy as np
import csv 
import Params 
import load 
import rnn 


def train(params): 

    # fix random seed 
    np.random.seed(params.random_seed)

    print('%s starting......' % params.cell)

    if params.dataset.startswith('mnist'): 
        train_X, test_X, train_y, test_y = load.load_mnist(params)
    elif params.dataset.startswith('sine_synthetic') and not params.dataset.startswith('sine_synthetic_out'): 
        train_X, test_X, train_y, test_y = load.load_sine_synthetic(params)
        train_X, test_X, train_y, test_y = load.load_ucr(params)
    elif params.dataset.startswith('poly_synthetic'): 
        train_X, test_X, train_y, test_y = load.load_poly_synthetic(params)
    else:
        assert 0, "unknown dataset %s" % (params.dataset)

    #params.freqs = np.logspace(np.log2(0.25), np.log2(params.time_steps/3), 120-1, base=2).tolist()
    #params.freqs.append(0.0)
    #params.freqs.sort()
    #params.freqs = np.linspace(0, params.time_steps/3, 10).tolist()
    print "parameters = ", params 

    if params.dataset.startswith('ptb') or params.dataset.startswith('charptb'): 
        model = rnn_ptb.RNNModel(params)
    else:
        model = rnn.RNNModel(params)

    # load model 
    if params.load_model: 
        model.load("%s.%s" % (params.model_dir, params.cell))

    # train model 
    train_error, test_error = model.train(params, train_X, train_y, test_X, test_y)

    # save model
    if params.model_dir: 
        if os.path.isdir(os.path.dirname(params.model_dir)) == False:
            os.makedirs(params.model_dir)
        model.save("%s.%s" % (params.model_dir, params.cell))

    # predict 
    train_pred = model.predict(train_X, params.batch_size)
    test_pred = model.predict(test_X, params.batch_size)

    # must close model when finish 
    model.close() 

    # write prediction to file 
    if params.pred_dir: 
        if os.path.isdir(os.path.dirname(params.pred_dir)) == False:
            os.makedirs(params.pred_dir)
        with open("%s.%s.%s.y" % (params.pred_dir, params.dataset, params.cell), "w") as f: 
            content = ""
            for pred in [train_pred, test_pred]: 
                for entry in pred: 
                    for index, value in enumerate(entry): 
                        if index: 
                            content += ","
                        content += "%f" % (value)
                    content += "\n"
            f.write(content)
        with open("%s.%s.%s.X" % (params.pred_dir, params.dataset, params.cell), "w") as f: 
            content = ""
            for X in [train_X, test_X]: 
                for entry in X: 
                    for index, value in enumerate(entry.ravel()): 
                        if index: 
                            content += ","
                        content += "%f" % (value)
                    content += "\n"
            f.write(content)

    return train_error, test_error 

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
