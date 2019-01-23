import tensorflow as tf
import math
import numpy as np


def lrelu(x, leak=0.2, name="lrelu"):
    """Leaky rectifier.
    Parameters
    ----------
    x : Tensor
        The tensor to apply the nonlinearity to.
    leak : float, optional
        Leakage parameter.
    name : str, optional
        Variable scope to use.
    Returns
    -------
    x : Tensor
        Output of the nonlinearity.
    """
    with tf.variable_scope(name):
        f1 = 0.5 * (1 + leak)
        f2 = 0.5 * (1 - leak)
        return f1 * x + f2 * abs(x)


class CompressionNet:
    """ Compression Network.
    This network converts the input data to the representations
    suitable for calculation of anomaly scores by "Estimation Network".

    Outputs of network consist of next 2 components:
    1) reduced low-dimensional representations learned by AutoEncoder.
    2) the features derived from reconstruction error.
    """

    def __init__(self, hidden_layer_sizes, activation=tf.nn.tanh, use_error_functions=True, use_cnn=False):
        """
        Parameters
        ----------
        hidden_layer_sizes : list of int
            list of the size of hidden layers.
            For example, if the sizes are [n1, n2],
            the sizes of created networks are:
            input_size -> n1 -> n2 -> n1 -> input_sizes
            (network outputs the representation of "n2" layer)
        activation : function
            activation function of hidden layer.
            the last layer uses linear function.
        """
        self.hidden_layer_sizes = hidden_layer_sizes
        self.activation = activation
        self.n_filters = [1, 10, 10, 10, 10, 10]
        self.filter_sizes = [3, 3, 3, 3, 3, 2]
        self.use_error_functions = use_error_functions
        self.use_cnn = use_cnn

    def compress_cnn(self, x):
        # ensure 2-d is converted to square tensor.
        if len(x.get_shape()) == 2:
            x_dim = np.sqrt(x.get_shape().as_list()[1])
            if x_dim != int(x_dim):
                raise ValueError('Unsupported input dimensions')
            x_dim = int(x_dim)
            x_tensor = tf.reshape(x, [-1, x_dim, x_dim, self.n_filters[0]])
        elif len(x.get_shape()) == 4:
            x_tensor = x
        else:
            raise ValueError('Unsupported input dimensions')
        current_input = x_tensor

        self.encoder = []
        self.shapes = []
        for layer_i, n_output in enumerate(self.n_filters[1:]):
            n_input = current_input.get_shape().as_list()[3]
            self.shapes.append(current_input.get_shape().as_list())
            W = tf.Variable(
                tf.random_uniform([self.filter_sizes[layer_i], self.filter_sizes[layer_i], n_input, n_output], -1.0 / math.sqrt(n_input), 1.0 / math.sqrt(n_input)))
            b = tf.Variable(tf.zeros([n_output]))
            self.encoder.append(W)
            output = lrelu(tf.add(tf.nn.conv2d(current_input, W, strides=[1, 2, 2, 1], padding='SAME'), b))
            current_input = output

        z = current_input
        z = tf.layers.Flatten()(z)
        return z

    def compress(self, x):
        self.input_size = x.shape[1]
        z = tf.layers.flatten(x)
        for layer in self.hidden_layer_sizes[:-1]:
            z = tf.layers.dense(z, layer, tf.nn.relu)
        z = tf.layers.dense(z, self.hidden_layer_sizes[-1])
        return z

    def reverse_cnn(self, z):
        assert len(z.shape) == 2, "latent code must be ? x num_features but is " + str(z.shape)
        current_input = tf.reshape(z, (-1, 1, 1, z.shape[-1]))
        self.encoder.reverse()
        self.shapes.reverse()
        for layer_i, shape in enumerate(self.shapes):
            W = self.encoder[layer_i]
            b = tf.Variable(tf.zeros([W.get_shape().as_list()[2]]))
            output = lrelu(tf.add(tf.nn.conv2d_transpose(current_input, W, tf.stack([tf.shape(z)[0], shape[1], shape[2], shape[3]]), strides=[1, 2, 2, 1], padding='SAME'), b))
            current_input = output

        output = tf.reshape(current_input, (-1, 784))
        return output

    def reverse(self, z):
        x_dash = z
        for layer in self.hidden_layer_sizes[:-1][::-1]:
            x_dash = tf.layers.dense(x_dash, layer, tf.nn.relu)
        x_dash = tf.layers.dense(x_dash, self.input_size)
        return x_dash

    def loss(self, x, x_dash):
        def euclid_norm(x):
            return tf.sqrt(tf.reduce_sum(tf.square(x), axis=1))

        # Calculate Euclid norm, distance
        norm_x = euclid_norm(x)
        norm_x_dash = euclid_norm(x_dash)
        dist_x = euclid_norm(x - x_dash)
        dot_x = tf.reduce_sum(x * x_dash, axis=1)

        # Based on the original paper, features of reconstruction error
        # are composed of these loss functions:
        #  1. loss_E : relative Euclidean distance
        #  2. loss_C : cosine similarity
        min_val = 1e-3
        loss_E = dist_x / (norm_x + min_val)
        loss_C = 0.5 * (1.0 - dot_x / (norm_x * norm_x_dash + min_val))
        return tf.concat([loss_E[:, None], loss_C[:, None]], axis=1)

    def extract_feature(self, x, x_dash, z_c):
        if self.use_error_functions:
            z_r = self.loss(x, x_dash)
            z_sum = tf.concat([z_c, z_r], axis=1)
        else:
            z_sum = z_c
        return z_sum

    def inference(self, x):
        """ convert input to output tensor, which is composed of
        low-dimensional representation and reconstruction error.

        Parameters
        ----------
        x : tf.Tensor shape : (n_samples, n_features)
            Input data

        Results
        -------
        z : tf.Tensor shape : (n_samples, n2 + 2)
            Result data
            Second dimension of this data is equal to
            sum of compressed representation size and
            number of loss function (=2)

        x_dash : tf.Tensor shape : (n_samples, n_features)
            Reconstructed data for calculation of
            reconstruction error.
        """

        with tf.variable_scope("CompNet"):

            if self.use_cnn:
                z_c = self.compress_cnn(x)
                x_dash = self.reverse_cnn(z_c)
            else:
                z_c = self.compress(x)
                self.reverse_tmpl = tf.make_template('reverse', self.reverse)
                x_dash = self.reverse_tmpl(z_c)

            # compose feature vector
            z = self.extract_feature(x, x_dash, z_c)

        return z, x_dash

    @staticmethod
    def reconstruction_error(x, x_dash):
        return tf.reduce_mean(tf.reduce_sum(
            tf.square(x - x_dash), axis=1), axis=0)
