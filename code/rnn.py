##
# @file   rnn.py
# @author Yibo Lin
# @date   Dec 2017
#

import math 
import pickle
import base64
import tensorflow as tf 
import numpy as np 
import sru
import fru 
import Params  
from tensorflow.python.ops import array_ops
from tensorflow.python.util import nest
import pdb 

class RNNModel (object):
    def __init__(self, params):
        self.rnn_cell = None 
        # feature 
        self.x = tf.placeholder("float", [None, params.time_steps, params.input_size])
        # label 
        self.y = tf.placeholder("float", [None, params.output_size])
        # train_flag placeholder
        self.train_flag = tf.placeholder(tf.bool, [], name="train_flag")
        # learning rate placeholder 
        self.learning_rate = tf.placeholder(tf.float32, [], name='learning_rate')
        # dropout keep rate placeholder 
        self.dropout_keep_rate = tf.placeholder(tf.float32, [], name='dropout_keep_rate')

        # set random seed before build the graph 
        tf.set_random_seed(params.random_seed)

        # build graph 
        logits = self.build(params)

        # prediction 
        # Define loss and optimizer
        # evaluation 
        if params.regression_flag: 
            self.pred = logits 
            self.loss_op = tf.reduce_mean(tf.pow(self.pred-self.y, 2))
            self.accuracy = self.loss_op
        else: 
            self.pred = tf.nn.softmax(logits)
            self.loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
                    logits=logits, labels=self.y))
            correct_pred = tf.equal(tf.argmax(self.pred, 1), tf.argmax(self.y, 1))
            self.accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

        # running session 
        config=tf.ConfigProto(device_count={'GPU' : int(params.gpu_flag)})
        config.gpu_options.per_process_gpu_memory_fraction = 0.1
        self.session = tf.Session(config=config)
        
        # batch size for validation 
        self.validate_batch_size = params.batch_size*4

        # metric name 
        if params.regression_flag: 
            self.metric = "RMS"
        else:
            self.metric = "accuracy"

    def __enter__(self): 
        return self 
    def __exit__(self, exc_type, exc_value, traceback): 
        self.close()

    """
    call this function to destroy globally defined variables in tensorflow
    """
    def close(self): 
        self.session.close()
        tf.reset_default_graph()

    def set_cell(self, params): 
        cells = []
        for layer in xrange(params.num_layers): 
            if params.cell == "RNN": 
                cell = tf.contrib.rnn.BasicRNNCell(
                        num_units=params.num_units
                        )
            elif params.cell == "LSTM":
                cell = tf.contrib.rnn.BasicLSTMCell(
                        num_units=params.num_units
                        )
            elif params.cell == "SRU":
                cell = sru.SimpleSRUCell(
                        num_stats=params.phi_size, 
                        mavg_alphas=tf.get_variable('alphas', initializer=tf.constant(params.alphas), trainable=False), 
                        output_dims=params.num_units, 
                        recur_dims=params.r_size
                        )
            elif params.cell == "FRU": 
                cell = fru.FRUCell(
                        num_stats=params.phi_size, 
                        freqs=params.freqs, 
                        freqs_mask=params.freqs_mask, 
                        seq_len=params.time_steps, 
                        output_dims=params.num_units, 
                        recur_dims=params.r_size, 
                        activation=tf.nn.relu
                        )
            else:
                assert 0, "unsupported cell %s" % (params.cell)

            # I found dropout wrapper affect performance 
            if params.dropout_keep_rate < 1.0: 
                cell = tf.contrib.rnn.DropoutWrapper(cell, output_keep_prob=self.dropout_keep_rate, dtype=np.float32)

            cells.append(cell)

        if params.num_layers > 1: 
            self.rnn_cell = tf.nn.rnn_cell.MultiRNNCell(cells)
        else:
            self.rnn_cell = cells[0]

        print "num_layers = ", params.num_layers

    def count_trainable_variables(self): 
        print "trainable_variables"
        for var in tf.trainable_variables():
            print var 
        # summarize #vars 
        num_vars = 0 
        for var in tf.trainable_variables(): 
            num = 1
            for dim in var.get_shape(): 
                num *= dim.value 
            num_vars += num 
        print "# trainable_variables = ", num_vars 

    def build(self, params): 
        self.set_cell(params)
    
        x = self.x
        # last linear layer 
        last_w = tf.get_variable("last_w", initializer=tf.truncated_normal([self.rnn_cell.output_size, params.output_size], stddev=0.1))
        last_b = tf.get_variable("last_b", initializer=tf.truncated_normal([params.output_size], stddev=0.1))

        # Unstack to get a list of 'time_steps' tensors of shape (batch_size, n_input)
        # assume time_steps is on axis 1
        x = tf.unstack(x, params.time_steps, 1)
        # get RNN cell output 
        if params.compute_initial_state_grad: 
            inputs = x
            first_input = inputs
            while nest.is_sequence(first_input):
                first_input = first_input[0]
            if first_input.get_shape().ndims != 1:
                input_shape = first_input.get_shape().with_rank_at_least(2)
                fixed_batch_size = input_shape[0]
                
                flat_inputs = nest.flatten(inputs)
                for flat_input in flat_inputs:
                    input_shape = flat_input.get_shape().with_rank_at_least(2)
                    batch_size, input_size = input_shape[0], input_shape[1:]
                    fixed_batch_size.merge_with(batch_size)
                    for i, size in enumerate(input_size):
                        if size.value is None:
                            raise ValueError(
                                    "Input size (dimension %d of inputs) must be accessible via "
                                    "shape inference, but saw value None." % i)
            else:
              fixed_batch_size = first_input.get_shape().with_rank_at_least(1)[0]

            if fixed_batch_size.value:
                batch_size = fixed_batch_size.value
            else:
                batch_size = array_ops.shape(first_input)[0]
            self.initial_state = self.rnn_cell.zero_state(batch_size, dtype=np.float32)
            output, states = tf.contrib.rnn.static_rnn(self.rnn_cell, x, initial_state=self.initial_state, dtype=np.float32)
        else:
            output, states = tf.contrib.rnn.static_rnn(self.rnn_cell, x, dtype=np.float32)
        # linear activation, using rnn inner loop last output 
        logits = tf.matmul(output[-1], last_w) + last_b
        print "output[-1].shape = ", output[-1].get_shape() 
        print "last_w.shape = ", last_w.get_shape()

        self.count_trainable_variables()

        return logits

    """
    @brief model training 
    @param params parameters 
    """
    def train(self, params, train_x, train_y, test_x, test_y):

        #optimizer = tf.train.GradientDescentOptimizer(learning_rate=self.learning_rate)
        optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
        if params.max_grad_norm > 0: 
            print("clip gradients")
            tvars = tf.trainable_variables()
            grads, _ = tf.clip_by_global_norm(tf.gradients(self.loss_op, tvars), params.max_grad_norm)
            train_op = optimizer.apply_gradients(
                    zip(grads, tvars),
                    global_step=tf.train.get_or_create_global_step())
        else: 
            train_op = optimizer.minimize(self.loss_op)

        if params.compute_initial_state_grad: 
            initial_state_grads = []
            if params.regression_flag: 
                for gi in range(0, params.output_size, 20): 
                    print gi 
                    if isinstance(self.initial_state, tf.contrib.rnn.LSTMStateTuple): 
                        #initial_state_grads = [tf.gradients(self.loss_op, self.initial_state.h), tf.gradients(self.loss_op, self.initial_state.c)]
                        initial_state_grads.append([tf.gradients(tf.reduce_mean(tf.pow(self.pred[:, gi:min(gi+20, params.output_size)]-self.y[:, gi:min(gi+20, params.output_size)], 2)), self.initial_state.h), tf.gradients(tf.reduce_mean(tf.pow(self.pred[:, gi:min(gi+20, params.output_size)]-self.y[:, gi:min(gi+20, params.output_size)], 2)), self.initial_state.c)])
                    else:
                        #initial_state_grads = tf.gradients(self.loss_op, self.initial_state)
                        initial_state_grads.append(tf.gradients(tf.reduce_mean(tf.pow(self.pred[:, gi:min(gi+20, params.output_size)]-self.y[:, gi:min(gi+20, params.output_size)], 2)), self.initial_state))
            else:
                print("gradients for regression")
                if isinstance(self.initial_state, tf.contrib.rnn.LSTMStateTuple): 
                    #initial_state_grads = [tf.gradients(self.loss_op, self.initial_state.h), tf.gradients(self.loss_op, self.initial_state.c)]
                    initial_state_grads = tf.gradients(self.loss_op, tf.trainable_variables())
                else:
                    initial_state_grads = tf.gradients(self.loss_op, self.initial_state)

        # Initialize the variables (i.e. assign their default value)
        init = tf.global_variables_initializer()

        # Start training 
        self.session.run(init)

        if params.compute_initial_state_grad: 
            initial_weights = self.session.run(tf.trainable_variables())
            print "initial_weights shape = ", len(initial_weights)
            opt = np.get_printoptions()
            np.set_printoptions(threshold='nan')
            for index, weights in enumerate(initial_weights): 
                print "initial_weights[%d] shape = %s" % (index, weights.shape)
                print weights
            np.set_printoptions(**opt)

        # only initialize if not train 
        if not params.train_flag: 
            print("model not trained")
            return None, None 

        train_error = []
        test_error = [] 
        iterations = 0 
        num_batches = math.ceil(len(train_x)/float(params.batch_size))
        for epoch in range(params.num_epochs): 
            if epoch == 0: 
                train_error.append(self.validate(train_x, train_y, batch_size=self.validate_batch_size))
                test_error.append(self.validate(test_x, test_y, batch_size=self.validate_batch_size))
                print("Epoch %d, iterations = %d, training %s = %.6f, testing %s = %.6f" % (-1, iterations, self.metric, train_error[-1], self.metric, test_error[-1]))

            # permute batches 
            perm = np.random.permutation(len(train_x))

            # run on batches 
            batch_index = 0 
            for batch_begin in range(0, len(train_x), params.batch_size): 
                # get batch x and y 
                batch_x = train_x[perm[batch_begin:min(batch_begin+params.batch_size, len(train_x))]]
                batch_y = train_y[perm[batch_begin:min(batch_begin+params.batch_size, len(train_x))]]
                # reduce learning rate every 1000 iterations
                learning_rate = params.initial_learning_rate*math.pow(params.lr_decay, iterations//1000)
                feed_dict = {self.x: batch_x, 
                        self.y: batch_y, 
                        self.train_flag: True, 
                        self.learning_rate: learning_rate, 
                        self.dropout_keep_rate: params.dropout_keep_rate}
                # Run optimization op (backprop)
                if params.compute_initial_state_grad and epoch in [0, 1, 5, 10, 20, 30, 40, 50, 60, 70, 80, 90] and batch_index == 0: 
                    initial_state_grads_values, _ = self.session.run([initial_state_grads, train_op], feed_dict=feed_dict)
                    for gi in range(len(initial_state_grads_values)): 
                        initial_state_grads_value = np.array(initial_state_grads_values[gi]).ravel()
                        #print "[%d]" % (gi)
                        #print initial_state_grads_value
                        l1_norm = np.linalg.norm(initial_state_grads_value, ord=1)
                        l2_norm = np.linalg.norm(initial_state_grads_value, ord=None)
                        linf_norm = np.linalg.norm(initial_state_grads_value, ord=np.inf)
                        lninf_norm = np.linalg.norm(initial_state_grads_value, ord=-np.inf)
                        print "initial_state_grads_values[%d] l1 = %.6f, l2 = %.6f, linf = %.6f, lninf = %.6f" % (
                                gi, 
                                l1_norm, 
                                l2_norm, 
                                linf_norm, 
                                lninf_norm
                                )
                else:
                    self.session.run(train_op, feed_dict=feed_dict)
                batch_index += 1
                iterations += 1

                # decay the display intervals for speedup 
                if batch_index % max(math.ceil(num_batches/float(max(params.display_epoch_num/pow(2, epoch//10)+1, 1))), 1) == 0: 
                    train_error.append(self.validate(train_x, train_y, batch_size=self.validate_batch_size))
                    test_error.append(self.validate(test_x, test_y, batch_size=self.validate_batch_size))
                    print("Epoch %s.%s, iterations = %s, training %s = %.6f, testing %s = %.6f, learning rate = %f" % 
                            ('{:03}'.format(epoch), '{:03}'.format(batch_index), '{:05}'.format(iterations), self.metric, train_error[-1], self.metric, test_error[-1], learning_rate))
            
            ## early exit if reach best training 
            #if params.regression_flag and train_error[-1] == 0.0: 
            #    break 
            #elif not params.regression_flag and train_error[-1] == 1.0:
            #    break 

            if np.isnan(train_error[-1]) or np.isinf(train_error[-1]) or np.isnan(test_error[-1]) or np.isinf(test_error[-1]): 
                print("found nan or inf, stop training")
                break 
            if params.tune_flag and epoch == 19:
                if not params.regression_flag: 
                    if train_error[-1] < 0.4: 
                        print("tune mode exit early")
                        break 

        print("Optimization Finished!")

        return train_error, test_error 

    """
    @brief prediction 
    @param params parameters 
    """
    def predict(self, x, batch_size=128): 
        # Launch the graph
        pred = np.zeros((len(x), self.pred.get_shape().as_list()[1]))
        # run on batches 
        for batch_begin in range(0, len(x), batch_size): 
            # get batch x and y 
            batch_x = x[batch_begin:min(batch_begin+batch_size, len(x))]
            # Run optimization op (backprop)
            pred[batch_begin:min(batch_begin+batch_size, len(x))] = self.session.run(self.pred, feed_dict={self.x: batch_x, 
                                           self.train_flag: False, 
                                           self.dropout_keep_rate: 1.0})
        return pred 

    """
    @brief validate prediction 
    @params x feature 
    @params y label 
    @param batch_size batch size 
    @return accuracy 
    """
    def validate(self, x, y, batch_size=128): 
        # error 
        cost = self.accuracy 
        # relative error  
        #cost = tf.reduce_sum(tf.pow(tf.divide(self.pred-self.y, self.y), 2))
        #cost = tf.reduce_sum(tf.pow((self.pred-self.y)/self.y, 2))
        validate_cost = 0.0 
        for batch_begin in range(0, len(x), batch_size): 
            # get batch x and y 
            batch_x = x[batch_begin:min(batch_begin+batch_size, len(x))]
            batch_y = y[batch_begin:min(batch_begin+batch_size, len(x))]
            feed_dict = {self.x: batch_x,
                    self.y: batch_y,
                    self.train_flag: False, 
                    self.dropout_keep_rate: 1.0}
            # Calculate batch loss and accuracy
            validate_cost += self.session.run(cost, feed_dict=feed_dict)*len(batch_y)
        return validate_cost/len(x)

    """
    @brief save model 
    @param filename file name 
    """
    def save(self, filename): 
        print "save model ", filename

        #def check_diagonal_dominated(x):
        #    d = np.diag(np.abs(x))
        #    s = np.sum(np.abs(x), axis=1) - d
        #    if np.all(d > s):
        #        return True
        #    return False

        ##np.set_printoptions(threshold='nan')
        #for var in tf.trainable_variables():
        #    print var.name
        #    print var.get_shape().as_list()
        #    if "recur_feats/Matrix" in var.name or "stats/Matrix" in var.name:
        #        mat = self.session.run(var)
        #        fname = "W1" if "recur_feats/Matrix" in var.name else "W2"
        #        with open(fname+".pkl", "wb") as f:
        #            pickle.dump(mat, f)
        #        mat_I = mat-np.eye(mat.shape[0], mat.shape[1])
        #        print "matrix = ", mat
        #        print "matrix-I = ", mat_I
        #        print "matrix fro = ", np.linalg.norm(mat)
        #        print "matrix-I fro = ", np.linalg.norm(mat_I)
        #        #print "diagonal dominated = ", check_diagonal_dominated(mat)

        saver = tf.train.Saver()
        saver.save(self.session, filename)

    """
    @brief load model 
    @param filename model file name 
    """
    def load(self, filename): 
        print "load model ", filename

        saver = tf.train.Saver()
        saver.restore(self.session, filename)

        # restore variables 
        graph = tf.get_default_graph()
        self.x = graph.get_tensor_by_name("x:0")
        self.y = graph.get_tensor_by_name("y:0")
        self.trainFlag = graph.get_tensor_by_name("trainFlag:0")
        self.learningRate = graph.get_tensor_by_name("learningRate:0")
