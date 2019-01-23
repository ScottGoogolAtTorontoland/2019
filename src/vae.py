import numpy as np
import tensorflow as tf
import tensorflow_probability as tfd

from .algorithm import Algorithm

tfd = tfd.distributions


class VAE(Algorithm):

    def __init__(self, hidden_layers, n_clusters):
        super().__init__(hidden_layers, n_clusters)
        self.prior, self.posterior, self.likelihood, self.divergence = [None] * 4

    def make_encoder(self, data, code_size):
        x = tf.layers.flatten(data)
        for layer in self.hidden_layers[:-1]:
            x = tf.layers.dense(x, layer, tf.nn.relu)
        loc = tf.layers.dense(x, code_size)
        scale = tf.layers.dense(x, code_size, tf.nn.softplus)
        return tfd.MultivariateNormalDiag(loc, scale)

    def make_prior(self, code_size):
        loc = tf.zeros(code_size)
        scale = tf.ones(code_size)
        return tfd.MultivariateNormalDiag(loc, scale)

    def make_decoder(self, code, data_shape):
        x = code
        for layer in self.hidden_layers[:-1][::-1]:
            x = tf.layers.dense(x, layer, tf.nn.relu)
        logit = tf.layers.dense(x, np.prod(data_shape))
        logit = tf.reshape(logit, [-1] + data_shape)
        return tfd.Independent(tfd.Bernoulli(logit), 2)

    def build_graph(self, x):
        n_features = x.shape[1]
        make_encoder = tf.make_template('encoder', self.make_encoder)
        make_decoder = tf.make_template('decoder', self.make_decoder)
        self.prior = self.make_prior(code_size=self.latent_size)
        self.posterior = make_encoder(x, code_size=self.latent_size)
        self.z = self.posterior.sample()
        self.x_dash = make_decoder(self.z, [n_features]).mean()

        self.likelihood = make_decoder(self.z, [n_features]).log_prob(x)
        self.divergence = tfd.kl_divergence(self.posterior, self.prior)
        self.loss = -tf.reduce_mean(self.likelihood - self.divergence)
        self.minimizer = tf.train.AdamOptimizer(0.001).minimize(self.loss)
        return self.z, self.x_dash, self.loss, self.minimizer

    def get_decoded_centroids(self):
        centers = self.get_kmeans_centroids()
        return self.sess.run(self.x_dash, feed_dict={self.z: centers})

    def get_custom_anomaly_scores(self, x):
        return [self.sess.run([self.loss], feed_dict={'x:0': row.reshape(1, -1)})[0] for row in x]

    def plot_custom_test_set(self, x, y, outlier=None):
        pass
