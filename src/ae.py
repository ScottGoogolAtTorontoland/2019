import tensorflow as tf
import tensorflow_probability as tfd

from src.compression_net import CompressionNet

tfd = tfd.distributions
from .algorithm import Algorithm


class AE(Algorithm):

    def __init__(self, hidden_layers, n_clusters, comp_activation=tf.nn.tanh, use_error_functions=False, use_cnn=False, learning_rate=1e-4):
        super().__init__(hidden_layers, n_clusters)
        self.learning_rate = learning_rate
        self.comp_net = CompressionNet(hidden_layers, activation=comp_activation, use_error_functions=use_error_functions, use_cnn=use_cnn)
        self.reconstruction_loss, self.minimizer = [None] * 2

    def build_graph(self, x):
        self.z, self.x_dash = self.comp_net.inference(x)
        self.reconstruction_loss = CompressionNet.reconstruction_error(x, self.x_dash)
        self.minimizer = tf.train.AdamOptimizer(self.learning_rate).minimize(self.reconstruction_loss)
        return self.z, self.x_dash, self.reconstruction_loss, self.minimizer

    def get_decoded_centroids(self):
        centers = self.get_kmeans_centroids()
        z_placeholder = tf.placeholder(dtype=tf.float32, shape=[self.n_clusters, self.latent_size], name='x')
        return self.sess.run(self.comp_net.reverse_tmpl(z_placeholder), feed_dict={z_placeholder: centers})
