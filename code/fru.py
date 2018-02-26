import math 
import tensorflow as tf
from tensorflow.python.util import nest


class FRUCell(tf.contrib.rnn.RNNCell):
    """Implements a simple distribution based recurrent unit that keeps moving
    averages of the mean map embeddings of features of inputs.
    """
    """
    num_stats: phi size 
    freqs: array of w 
    freqs_mask: mask value when frequency is not equal to zero
    output_dims: output size 
    recur_dims: r size 
    seq_len: length of sequence 
    """

    def __init__(self, num_stats, freqs, freqs_mask, output_dims, recur_dims, seq_len, 
                 summarize=True, linear_out=False,
                 include_input=False, activation=tf.nn.relu):
        self._num_stats = num_stats
        self._output_dims = output_dims
        self._recur_dims = recur_dims
        self._freqs_array = freqs 
        self._nfreqs = len(freqs)
        self._freqs_mask_array = [0.0 if w == 0 and len(freqs) > 1 else freqs_mask for w in freqs]
        print "frequency_mask = ", self._freqs_mask_array
        self._summarize = summarize
        self._linear_out = linear_out
        self._activation = activation
        self._include_input = include_input

        # as tensorflow does not feed current time step to __call__ 
        # I have to manually record it 
        self._seq_len = seq_len
        self._cur_time_step = 0

        self.W = []
        self.b = []

    """
    nfreqs*num_stats
    """
    @property
    def state_size(self):
        return int(self._nfreqs * self._num_stats)

    @property
    def output_size(self):
        return self._output_dims

    """
    current time step 
    """
    @property
    def cur_time_step(self):
        return self._cur_time_step

    """
    go to next time step 
    """
    def next_time_step(self): 
        self._cur_time_step = (self._cur_time_step+1) % self._seq_len

    def __call__(self, inputs, state, scope=None):
        """
        recur*: r
        state*: mu 
        stats*: phi 
        freq*: frequency vector 
        """
        with tf.variable_scope(scope or type(self).__name__):
            self._freqs = tf.reshape(tf.get_variable("frequency", initializer=self._freqs_array, trainable=False), [1, -1, 1])
            self._phases = tf.reshape(tf.get_variable("phase", [self._nfreqs], initializer=tf.truncated_normal_initializer(stddev=0.1, dtype=tf.float32), trainable=True), [1, -1, 1])
            self._freqs_mask = tf.reshape(tf.get_variable("frequency_mask", initializer=self._freqs_mask_array, trainable=False), [1, -1, 1])
            # Make statistics on input.
            if self._recur_dims > 0:
                """
                r_t = f(W^r mu_{t-1} + b^r)
                """
                recur_output = self._activation(_linear(
                    state, self._recur_dims, True, scope='recur_feats'
                ), name='recur_feats_act')
                """
                phi_t = W^phi r_t + W^x x_t + b^phi 
                """
                stats = self._activation(_linear(
                    [inputs, recur_output], self._num_stats, True, scope='stats', 
                ), name='stats_act')
            else:
                stats = self._activation(_linear(
                    inputs, self._num_stats, True, scope='stats'
                ), name='stats_act')
            # Compute moving averages of statistics for the state.
            with tf.variable_scope('out_state'):
                state_tensor = tf.reshape(
                    state, [-1, self._nfreqs, self._num_stats], 'state_tensor'
                )
                stats_tensor = tf.reshape(
                    stats, [-1, 1, self._num_stats], 'stats_tensor'
                )
                """
                mu_t = mask*mu_{t-1} + cos(2*pi*w*t/T + 2*pi*phase)*phi_t
                """
                out_state = tf.reshape(self._freqs_mask*state_tensor +
                                       1.0/self._seq_len*tf.cos(2.0*math.pi/self._seq_len*self.cur_time_step*self._freqs + 2.0*math.pi*self._phases)*stats_tensor,
                                       [-1, self.state_size], 'out_state')
            # Compute the output.
            if self._include_input:
                output_vars = [out_state, inputs]
            else:
                output_vars = out_state
            """
            o_t = W^o mu_t + b^o
            """
            output = _linear(
                output_vars, self._output_dims, True, scope='output'
            )
            if not self._linear_out:
                output = self._activation(output, name='output_act')
            # update time step 
            self.next_time_step()

            # Retrieve RNN Variables
            if not self.W: 
                with tf.variable_scope('recur_feats', reuse=True):
                    self.W.append(tf.get_variable('Matrix'))
                    self.b.append(tf.get_variable('Bias'))
                with tf.variable_scope('stats', reuse=True):
                    self.W.append(tf.get_variable('Matrix'))
                    self.b.append(tf.get_variable('Bias'))
                with tf.variable_scope('output', reuse=True):
                    self.W.append(tf.get_variable('Matrix'))
                    self.b.append(tf.get_variable('Bias'))
                print("W = ", self.W)
                print("b = ", self.b)

            """
            o_t and mu_t 
            """
            return (output, out_state)


# No longer publicly expose function in tensorflow.
def _linear(args, output_size, bias, bias_start=0.0, scope=None):
    """Linear map: sum_i(args[i] * W[i]), where W[i] is a variable.

    Args:
      args: a 2D Tensor or a list of 2D, batch x n, Tensors.
      output_size: int, second dimension of W[i].
      bias: boolean, whether to add a bias term or not.
      bias_start: starting value to initialize the bias; 0 by default.
      scope: VariableScope for the created subgraph; defaults to "Linear".

    Returns:
      A 2D Tensor with shape [batch x output_size] equal to
      sum_i(args[i] * W[i]), where W[i]s are newly created matrices.

    Raises:
      ValueError: if some of the arguments has unspecified or wrong shape.
    """
    if args is None or (nest.is_sequence(args) and not args):
        raise ValueError("`args` must be specified")
    if not nest.is_sequence(args):
        args = [args]

    # Calculate the total size of arguments on dimension 1.
    total_arg_size = 0
    shapes = [a.get_shape().as_list() for a in args]
    for shape in shapes:
        if len(shape) != 2:
            raise ValueError(
                "Linear is expecting 2D arguments: %s" %
                str(shapes))
        if not shape[1]:
            raise ValueError(
                "Linear expects shape[1] of arguments: %s" %
                str(shapes))
        else:
            total_arg_size += shape[1]

    dtype = [a.dtype for a in args][0]

    # Now the computation.
    with tf.variable_scope(scope or "Linear"):
        matrix = tf.get_variable(
            "Matrix", [total_arg_size, output_size], initializer=tf.truncated_normal_initializer(stddev=0.1, dtype=dtype), dtype=dtype)
        if len(args) == 1:
            res = tf.matmul(args[0], matrix)
        else:
            res = tf.matmul(tf.concat(args, 1), matrix)
        if not bias:
            return res
        bias_term = tf.get_variable(
            "Bias", [output_size],
            dtype=dtype,
            initializer=tf.constant_initializer(bias_start, dtype=dtype)
        )

    return res + bias_term
