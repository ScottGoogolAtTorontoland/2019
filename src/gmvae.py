import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfd
from tensorflow.layers import dense
from tensorflow.python.ops.nn_ops import softmax_cross_entropy_with_logits_v2 as softmax_xent

from src.plotting import plot_latent_code
from .algorithm import Algorithm

tfd = tfd.distributions


class GMVAE(Algorithm):

    def __init__(self, hidden_layers, n_clusters, img_dims, encoder, decode_activation=None, use_deconvs=False):
        super().__init__(hidden_layers, n_clusters)
        self.img_dims = img_dims
        self.decode_activation = decode_activation
        self.encoder = encoder
        self.use_deconvs = use_deconvs
        self.xb, self.y_, self.qy_logit, self.qy, self.zm, self.zv, self.zm_prior, self.zv_prior, self.px_logit, self.z_representative, self.nent, self.zs,\
        self.decode, self.n_features, self.losses = [None] * 15

    def infer_y(self, x, k=10):
        """Infer y in logits as well as probability distribution"""
        reuse = len(tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='qy')) > 0
        with tf.variable_scope('qy'):
            for i, layer in enumerate(self.hidden_layers[:-1]):
                x = dense(x, layer, activation=tf.nn.relu, name="layer" + str(i + 1), reuse=reuse)
            qy_logit = dense(x, k, name='logit', reuse=reuse)
            qy = tf.nn.softmax(qy_logit, name='prob')
        return qy_logit, qy

    def encode(self, x, y, encoder=None, latent_size=6):
        """
        Infer mean and variance of latent representation, sample from it as well
        """
        reuse = len(tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='qz')) > 0
        # -- q(z)
        with tf.variable_scope('qz'):
            if encoder is None:
                encoder = tf.concat((x, y), 1, name='xy/concat')
                for i, layer in enumerate(self.hidden_layers[:-1]):
                    encoder = dense(encoder, layer, activation=tf.nn.relu, name='layer' + str(i+1), reuse=reuse)
            else:
                if len(encoder.shape) != 2:
                    encoder = tf.reshape(encoder, [-1, np.product(encoder.shape[1:])])
                encoder = dense(encoder, 512, name='encoder_crunch_512', activation=tf.nn.elu, reuse=reuse)
                encoder = dense(encoder, 128, name='encoder_crunch_128', activation=tf.nn.relu, reuse=reuse)
            zm = dense(encoder, latent_size, name='zm', reuse=reuse)
            zv = dense(encoder, latent_size, name='zv', activation=tf.nn.softplus, reuse=reuse)
            z = gaussian_sample(zm, zv, scope='z', vary_z=self.vary_z)
        return z, zm, zv

    def decoder(self, z, y, latent_size=6):
        """
        Infer mean and variance of input as well as reconstruction
        :param z: (?, LATENT)
        :param y: (?, k)
        :return: zm (?, LATENT), zv (?, LATENT), px_logit (?, x_input)
        """

        reuse = len(tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='px')) > 0
        # -- p(z)
        with tf.variable_scope('pz'):
            zm = dense(y, latent_size, name='zm', reuse=reuse)
            zv = dense(y, latent_size, name='zv', activation=tf.nn.softplus, reuse=reuse)
        # -- p(x)
        with tf.variable_scope('px'):
            if self.use_deconvs:
                h = tf.layers.conv2d_transpose(tf.reshape(z, [-1, 1, 1, latent_size]), 12, (3, 3))
                h = tf.layers.conv2d_transpose(h, 12, (3, 3), reuse=reuse)
                h = tf.layers.conv2d_transpose(h, 12, (3, 3), reuse=reuse)
                h = tf.layers.conv2d_transpose(h, 12, (3, 3), reuse=reuse)
                #h = tf.layers.conv2d_transpose(h, 12, (3, 3), reuse=reuse)
                #h = tf.layers.conv2d_transpose(h, 12, (3, 3), reuse=reuse)
                #h = tf.layers.conv2d_transpose(h, 12, (3, 3), reuse=reuse)
                #h = tf.layers.conv2d_transpose(h, 12, (3, 3), reuse=reuse)
                #h = tf.layers.conv2d_transpose(h, 12, (3, 3), reuse=reuse)
                #h = tf.layers.conv2d_transpose(h, 12, (3, 3), reuse=reuse)
                #h = tf.layers.conv2d_transpose(h, 12, (3, 3), reuse=reuse)
                #h = tf.layers.conv2d_transpose(h, 12, (3, 3), reuse=reuse)
                #h = tf.layers.conv2d_transpose(h, 1, (3, 3), reuse=reuse)
                h = tf.layers.Flatten()(h)
            else:
                h = dense(z, self.hidden_layers[-2], activation=tf.nn.relu, name='layer1', reuse=reuse)
                for i, layer in enumerate(self.hidden_layers[:-3][::-1]):
                    h = dense(h, layer, activation=tf.nn.relu, name="layer" + str(i + 2), reuse=reuse)
            px_logit = dense(h, self.n_features, activation=self.decode_activation, name='logit', reuse=reuse)
        return zm, zv, px_logit

    def build_graph(self, input):
        self.input = input
        self.vary_z = tf.placeholder_with_default([1.0], shape=1, name='vary_z')
        self.n_features = np.prod(self.input.shape[1:])

        self.xb = self.input
        assert len(self.xb.shape) == 2, self.xb

        with tf.name_scope('y_'):
            self.y_ = tf.fill(tf.stack([tf.shape(self.input)[0], self.n_clusters]), 0.0)

        self.qy_logit, self.qy = self.infer_y(self.xb, self.n_clusters)  # Assignment to one of k Gaussians

        for label_variable in [self.y_, self.qy, self.qy_logit]:
            assert len(label_variable.shape) == 2 and label_variable.shape[-1] == self.n_clusters

        self.zs, self.zm, self.zv, self.zm_prior, self.zv_prior, self.px_logit, self.z_representative = [[None] * self.n_clusters for _ in range(7)]
        self.decode = tf.make_template('decode', self.decoder)
        for i in range(self.n_clusters):
            with tf.name_scope('graphs/hot_at{:d}'.format(i)):
                y = tf.add(self.y_, tf.constant(np.eye(self.n_clusters)[i], dtype='float32', name='hot_at_{:d}'.format(i)))
                self.zs[i], self.zm[i], self.zv[i] = self.encode(self.xb, y, encoder=self.encoder, latent_size=self.latent_size)
                self.zm_prior[i], self.zv_prior[i], self.px_logit[i] = self.decode(self.zs[i], y, self.latent_size)
                _, _, self.z_representative[i] = self.decode(self.zm_prior[i], y, self.latent_size)

        for latent_encoding_variable in [self.zm, self.zs, self.zv, self.zm_prior, self.zv_prior]:
            for dimension_variable in latent_encoding_variable:
                assert len(dimension_variable.shape) == 2 and dimension_variable.shape[-1] == self.latent_size
        for decoded_variables in [self.px_logit, self.z_representative]:
            for dimension_variable in decoded_variables:
                assert len(dimension_variable.shape) == 2 and dimension_variable.shape[-1] == self.n_features

        with tf.name_scope('loss'):
            with tf.name_scope('neg_entropy'):
                self.nent = softmax_cross_entropy_with_two_logits(self.qy_logit, self.qy)
            self.losses = [None] * self.n_clusters
            for i in range(self.n_clusters):
                with tf.name_scope('loss_at{:d}'.format(i)):
                    self.losses[i] = labeled_loss(self.xb, self.px_logit[i], self.zs[i], self.zm[i], self.zv[i], self.zm_prior[i], self.zv_prior[i])
            with tf.name_scope('final_loss'):
                self.loss = tf.add_n([self.nent] + [self.qy[:, i] * self.losses[i] for i in range(self.n_clusters)])

        self.x_dash = self.px_logit[0]
        self.z = self.zs[0]
        self.minimizer = tf.train.AdamOptimizer(0.0001).minimize(self.loss)
        return self.z, self.x_dash, self.loss, self.minimizer

    def get_reconstruction(self, x):
        x_dash_gmvae, x_dash_y = self.sess.run([self.px_logit, self.qy], feed_dict={'x:0': x})
        gaussian_assignments = x_dash_y.argmax(1)
        # x_dash_gmvae = 10x 2153 x n_features
        # x_dash_y     =     2153 x 1
        x_dash = np.zeros_like(x)
        for sample_index, sample_assignment in enumerate(gaussian_assignments):
            x_dash[sample_index] = x_dash_gmvae[sample_assignment][sample_index]
        return x_dash

    def get_custom_anomaly_scores(self, x):
        return self.sess.run(self.loss, feed_dict={'x:0': x})

    def get_custom_assignments(self, x):
        return self.sess.run(self.qy_logit, feed_dict={'x:0': x}).argmax(1)

    def get_decoded_centroids(self):
        # centers = self.get_kmeans_centroids()
        # z_placeholder = tf.placeholder(dtype=tf.float32, shape=[10, self.latent_size], name='x')
        # return self.sess.run(decode(z_placeholder, self.y_, self.latent_size), feed_dict={z_placeholder: centers, 'x:0': np.zeros((1, self.n_features))})[2]
        return np.asarray([self.sess.run(self.z_representative[i], feed_dict={'x:0': np.zeros((1, self.n_features)), 'vary_z:0': [0]}) for i in range(self.n_clusters)])

    def get_closest_decoded_centroids(self, x):
        decoded_centroids = self.get_decoded_centroids()
        assignments = self.sess.run(self.qy, feed_dict={'x:0': x.reshape(-1, self.n_features)}).argmax(1)
        closest_centroids = np.zeros((x.shape[0], self.n_features))
        for i in range(x.shape[0]):
            closest_centroids[i, :] = decoded_centroids[assignments[i]]
        return closest_centroids

    def plot_custom_test_set(self, x, y, outlier=None):
        fig, ax = plt.subplots(nrows=2, ncols=self.n_clusters, figsize=(10 * self.n_clusters, 2 * self.n_clusters))
        z_results, qy_result = self.sess.run([self.zs, self.qy], feed_dict={'x:0': x.reshape(-1, self.n_features)})
        representatives = self.get_decoded_centroids()
        if self.img_dims is not None:
            plot_latent_code(fig, ax, self.n_clusters, 0, z_results, qy_result, representatives, y, self.img_dims, outlier=outlier)


def softmax_cross_entropy_with_two_logits(logits=None, labels=None):
    return softmax_xent(labels=tf.nn.softmax(labels), logits=logits)


def gaussian_sample(mean, var, vary_z=None, scope=None):
    """
    loc = tf.layers.dense(x, code_size)
    scale = tf.layers.dense(x, code_size, tf.nn.softplus)
    return tfd.MultivariateNormalDiag(loc, scale)
    """

    with tf.variable_scope(scope, 'gaussian_sample'):
        sample = tf.random_normal(tf.shape(mean), mean, vary_z * tf.sqrt(var))
        sample.set_shape(mean.get_shape())
        return sample


def log_bernoulli(x, logits, eps=0.0, axis=-1):
    return log_bernoulli_with_logits(x, logits, eps, axis)


def log_bernoulli_with_logits(x, logits, eps=0.0, axis=-1):
    if eps > 0.0:
        max_val = np.log(1.0 - eps) - np.log(eps)
        logits = tf.clip_by_value(logits, -max_val, max_val,
                                  name='clipped_logit')
    return -tf.reduce_sum(
        tf.nn.sigmoid_cross_entropy_with_logits(logits=logits, labels=x), axis)


def log_normal(x, mu, var, eps=0.0, axis=-1):
    if eps > 0.0:
        var = tf.add(var, eps, name='clipped_var')
    return -0.5 * tf.reduce_sum(
        tf.log(2 * np.pi) + tf.log(var) + tf.square(x - mu) / var, axis)


def labeled_loss(x, px_logit, z, zm, zv, zm_prior, zv_prior):
    xy_loss = -log_bernoulli_with_logits(x, px_logit)
    xy_loss += log_normal(z, zm, zv) - log_normal(z, zm_prior, zv_prior)
    return xy_loss - np.log(0.1)
