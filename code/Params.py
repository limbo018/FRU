##
# @file   Params.py
# @author Yibo Lin
# @date   Dec 2017
#

import json
import math

"""
Parameter class 
"""
class Params (object):
    def __init__(self): 
        self.cell = None # RNN cell 
        self.initial_learning_rate = math.exp(-10) # learning rate for SGD, [exp(-10), 1]
        self.lr_decay = 0.8 # the multiplier to multiply the learning rate by every 1k iterations, in range of [0.8, 0.999]
        self.num_epochs = 100 # number of epochs 
        self.dropout_keep_rate = 0.5 # percent of output units that are kept during dropout, in range (0, 1]
        self.num_units = 200 # number of units 
        self.phi_size = 200 # the dimensionality of \phi(6), in {1, ..., 256}
        self.r_size = 60 # the dimensionality of r(5), in {1, ..., 64}
        self.alphas = [0, 0.25, 0.5, 0.9, 0.99] # for SRU 
        self.freqs = [0.0, 0.5, 1.0, 2.0, 3.0] # for FRU 
        self.freqs_mask = 1.0 # for FRU, mask value when frequency is not equal to zero 
        self.time_steps = None # time steps, time_steps*input_size = sequence length  
        self.input_size = None # dimensionality of input features at each time step 
        self.output_size = None # dimensionality of label
        self.gpu_flag = True # use GPU or not 
        self.random_seed = 1000 # random seed 
        self.dataset = 'synthetic' # dataset name 
        self.batch_size = 128 # batch size 
        self.regression_flag = True # regression or classification 
        self.tune_params_flag = 'limited' # limited or full 
        self.model_dir = '' # directory to save model, will append .cell_name  
        self.pred_dir = '' # directory for prediction results, will append .dataset.cell_name.[Xy] 
        self.load_model = False # load model or not 
        self.train_flag = True # train model or not 
        self.batch_norm = False # batch normalization or not 
        self.display_epoch_num = 1 # display how many evaluations per epoch 
        self.num_layers = 1 # number of layers 
        self.max_grad_norm = 0 # for gradient clipping, only valid when positive  
        self.vocabulary_size = 10000 # vocabulary size for ptb dataset 
        self.tune_flag = False # tune mode, track accuracy and exit early if not good  
        self.compute_initial_state_grad = False # whether computer gradients for initial state
    """
    convert to json  
    """
    def toJson(self):
        data = dict()
        data['cell'] = self.cell
        data['initial_learning_rate'] = self.initial_learning_rate
        data['lr_decay'] = self.lr_decay
        data['num_epochs'] = self.num_epochs
        data['dropout_keep_rate'] = self.dropout_keep_rate
        data['num_units'] = self.num_units
        data['phi_size'] = self.phi_size
        data['r_size'] = self.r_size
        data['alphas'] = self.alphas
        data['freqs'] = self.freqs
        data['freqs_mask'] = self.freqs_mask
        data['time_steps'] = self.time_steps
        data['input_size'] = self.input_size
        data['output_size'] = self.output_size
        data['gpu_flag'] = self.gpu_flag
        data['batch_size'] = self.batch_size
        data['random_seed'] = self.random_seed
        data['dataset'] = self.dataset
        data['regression_flag'] = self.regression_flag
        data['tune_params_flag'] = self.tune_params_flag
        data['model_dir'] = self.model_dir
        data['pred_dir'] = self.pred_dir
        data['load_model'] = self.load_model
        data['train_flag'] = self.train_flag
        data['batch_norm'] = self.batch_norm
        data['display_epoch_num'] = self.display_epoch_num
        data['num_layers'] = self.num_layers
        data['max_grad_norm'] = self.max_grad_norm
        data['vocabulary_size'] = self.vocabulary_size
        data['tune_flag'] = self.tune_flag
        data['compute_initial_state_grad'] = self.compute_initial_state_grad
        return data 
    """
    load form json 
    """
    def fromJson(self, data):
        if 'cell' in data: self.cell = data['cell']
        if 'initial_learning_rate' in data: self.initial_learning_rate = data['initial_learning_rate']
        if 'lr_decay' in data: self.lr_decay = data['lr_decay']
        if 'num_epochs' in data: self.num_epochs = data['num_epochs']
        if 'dropout_keep_rate' in data: self.dropout_keep_rate = data['dropout_keep_rate']
        if 'num_units' in data: self.num_units = data['num_units']
        if 'phi_size' in data: self.phi_size = data['phi_size']
        if 'r_size' in data: self.r_size = data['r_size']
        if 'alphas' in data: self.alphas = data['alphas']
        if 'freqs' in data: self.freqs = data['freqs']
        if 'freqs_mask' in data: self.freqs_mask = data['freqs_mask']
        if 'time_steps' in data: self.time_steps = data['time_steps']
        if 'input_size' in data: self.input_size = data['input_size']
        if 'output_size' in data: self.output_size = data['output_size']
        if 'gpu_flag' in data: self.gpu_flag = data['gpu_flag']
        if 'batch_size' in data: self.batch_size = data['batch_size']
        if 'random_seed' in data: self.random_seed = data['random_seed']
        if 'dataset' in data: self.dataset = data['dataset']
        if 'regression_flag' in data: self.regression_flag = data['regression_flag']
        if 'tune_params_flag' in data: self.tune_params_flag = data['tune_params_flag']
        if 'model_dir' in data: self.model_dir = data['model_dir']
        if 'pred_dir' in data: self.pred_dir = data['pred_dir']
        if 'load_model' in data: self.load_model = data['load_model']
        if 'train_flag' in data: self.train_flag = data['train_flag']
        if 'batch_norm' in data: self.batch_norm = data['batch_norm']
        if 'display_epoch_num' in data: self.display_epoch_num = data['display_epoch_num']
        if 'num_layers' in data: self.num_layers = data['num_layers']
        if 'max_grad_norm' in data: self.max_grad_norm = data['max_grad_norm']
        if 'vocabulary_size' in data: self.vocabulary_size = data['vocabulary_size']
        if 'tune_flag' in data: self.tune_flag = data['tune_flag']
        if 'compute_initial_state_grad' in data: self.compute_initial_state_grad = data['compute_initial_state_grad']

    """
    dump to json file 
    """
    def dump(self, filename):
        with open(filename, 'w') as f:
            meta = self.toJson()
            json.dump(dict((key, value) for key, value in meta.iteritems() if value != None), f, sort_keys=True, indent=4, separators=(', ', ' : '))
    """
    load from json file 
    """
    def load(self, filename):
        with open(filename, 'r') as f:
            self.fromJson(json.load(f))
    """
    string 
    """
    def __str__(self):
        return str(self.toJson())
    """
    print 
    """
    def __repr__(self):
        return self.__str__()
