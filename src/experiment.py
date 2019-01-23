import os
import time

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfd
from scipy.stats import mode
from sklearn.externals import joblib
from sklearn.metrics import adjusted_rand_score, roc_auc_score
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm_notebook as tqdm

from config import OUTPUT_PATH
from src.ae import AE
from src.algorithm import Encoding
from src.gmvae import GMVAE
from src.vae import VAE

tfd = tfd.distributions


def accuracy(y_pred, y_true):
    """
    :param y_pred: (n, 1) Predicted cluster for each sample
    :param y_true: (n, 1) Label for each sample
    :return: Probability that a random sample was assigned to a cluster with mostly other samples of the same label
    """
    real_pred = np.zeros_like(y_pred)  # n x 1
    for i in np.unique(y_pred):
        cluster_samples_idx = y_pred == i  # All samples that have been assigned to cluster i
        true_classes_of_samples = y_true[cluster_samples_idx]
        if sum(cluster_samples_idx) == 0:  # No sample is assigned to the cluster
            continue
        real_pred[cluster_samples_idx] = mode(true_classes_of_samples).mode[0]
    return np.mean(real_pred == y_true)


class Experiment:
    MODEL_FILENAME = "model"
    SCALER_FILENAME = "model_scaler"

    def __init__(self, comp_hiddens, n_clusters, comp_activation=tf.nn.tanh, use_cnn=False,
                 minibatch_size=1024, epoch_size=100, learning_rate=0.0001, name="MNIST",
                 normalize=True, random_seed=123, sess=None, use_error_functions=False, binarize=False, gmvae_decode_activation=None,
                 encoding=Encoding.AE, img_dims=(28, 28), encoder=None, graph=None, encoder_input=None, encoder_input_data=None, use_deconvs=False, log_performance=True):
        """
        Parameters
        ----------
        comp_hiddens : list of int
            sizes of hidden layers of compression network
            For example, if the sizes are [n1, n2],
            structure of compression network is:
            input_size -> n1 -> n2 -> n1 -> input_sizes
        comp_activation : function
            activation function of compression network
        minibatch_size: int (optional)
            mini batch size during training
        epoch_size : int (optional)
            epoch size during training
        learning_rate : float (optional)
            learning rate during training
        normalize : bool (optional)
            specify whether input data need to be normalized.
            by default, input data is normalized.
        random_seed : int (optional)
            random seed used when fit() is called.
        """
        print(name, encoding)
        self.hidden_layers = comp_hiddens
        self.use_cnn = use_cnn

        self.comp_activation = comp_activation

        self.n_clusters = n_clusters

        self.minibatch_size = minibatch_size
        self.epoch_size = epoch_size
        self.learning_rate = learning_rate
        self.name = name
        self.img_dims = img_dims
        self.normalize = normalize
        self.scaler = None
        self.seed = random_seed
        self.encoding = encoding
        self.use_error_functions = use_error_functions
        self.gmvae_decode_activation = gmvae_decode_activation
        self.encoder = encoder
        self.encoder_input = encoder_input
        self.encoder_input_data = encoder_input_data
        self.use_deconvs = use_deconvs
        self.log_performance = log_performance
        self.input, self.x_dash, self.z, self.likelihood, self.loss, self.minimizer, self.saver, self.algorithm = [None] * 8

        if graph is None:
            self.graph = tf.get_default_graph()
        else:
            self.graph = graph
        self.sess = sess
        self.initialized = False
        self.binarize = binarize

    def __del__(self):
        if self.sess is not None:
            self.sess.close()

    def fit(self, x, y=None, x_test=None, y_test=None, outlier_classes=None):
        n_samples, n_features = x.shape
        timestamp = time.strftime('%Y-%m-%d-%H%M%S')

        if self.normalize:
            self.scaler = scaler = StandardScaler()
            x = scaler.fit_transform(x)

        with self.graph.as_default():
            # tf.set_random_seed(self.seed)
            # np.random.seed(seed=self.seed)

            # Create Placeholder
            self.input = tf.placeholder(dtype=tf.float32, shape=[None, n_features], name='x')

            if self.encoding == Encoding.AE:
                self.algorithm = AE(self.hidden_layers, self.n_clusters, comp_activation=self.comp_activation, use_error_functions=self.use_error_functions,
                                    use_cnn=self.use_cnn)
            elif self.encoding == Encoding.VAE:
                self.algorithm = VAE(self.hidden_layers, self.n_clusters)
            elif self.encoding == Encoding.GMVAE:
                self.algorithm = GMVAE(self.hidden_layers, self.n_clusters, self.img_dims, self.encoder, decode_activation=self.gmvae_decode_activation,
                                       use_deconvs=self.use_deconvs)
            else:
                raise Exception("Invalid parameter 'encoding' given: " + str(self.encoding))

            self.z, self.x_dash, self.loss, self.minimizer = self.algorithm.build_graph(self.input)

            init = tf.global_variables_initializer()

            if self.sess is None:
                self.sess = tf.Session(graph=self.graph)
            if not self.initialized:
                self.sess.run(init)
                self.initialized = True

            self.algorithm.set_session(self.sess)

            # Monitor
            reconstruction_loss_placeholder = tf.placeholder(tf.float32, shape=(), name="auc_reconstruction")
            tf.summary.scalar('Reconstruction_Loss', reconstruction_loss_placeholder)
            auc_reconstruction_placeholder = tf.placeholder(tf.float32, shape=(), name="auc_reconstruction")
            tf.summary.scalar('AUC_Reconstruction', auc_reconstruction_placeholder)
            auc_custom_placeholder = tf.placeholder(tf.float32, shape=(), name="auc_custom")
            tf.summary.scalar('AUC_Custom', auc_custom_placeholder)
            ari_kmeans_placeholder = tf.placeholder(tf.float32, shape=(), name="ari_kmeans")
            tf.summary.scalar('ARI_Kmeans', ari_kmeans_placeholder)
            ari_custom_placeholder = tf.placeholder(tf.float32, shape=(), name="ari_custom")
            tf.summary.scalar('ARI_Custom', ari_custom_placeholder)
            accuracy_kmeans_placeholder = tf.placeholder(tf.float32, shape=(), name="accuracy_kmeans")
            tf.summary.scalar('Accuracy_Kmeans', accuracy_kmeans_placeholder)
            accuracy_custom_placeholder = tf.placeholder(tf.float32, shape=(), name="accuracy_custom")
            tf.summary.scalar('Accuracy_Custom', accuracy_custom_placeholder)

            if self.img_dims is not None:
                cluster_centroids_placeholder = tf.placeholder(tf.float32, shape=(self.n_clusters,) + self.img_dims + (1,), name='centroids')
                tf.summary.image('Centroids', cluster_centroids_placeholder, max_outputs=self.n_clusters)

                if outlier_classes is not None:
                    reconstruction_img_in_placeholder = tf.placeholder(tf.float32, shape=(1,) + self.img_dims + (1,), name='reconstruction_in')
                    tf.summary.image('Reconstruction_In', reconstruction_img_in_placeholder, max_outputs=1)
                    reconstruction_img_out_placeholder = tf.placeholder(tf.float32, shape=(1,) + self.img_dims + (1,), name='reconstruction_out')
                    tf.summary.image('Reconstruction_Out', reconstruction_img_out_placeholder, max_outputs=1)
                    closest_centroid_placeholder = tf.placeholder(tf.float32, shape=(1,) + self.img_dims + (1,), name='closest_centroid')
                    tf.summary.image('Closest_centroid', closest_centroid_placeholder, max_outputs=1)

            merged_summary_op = tf.summary.merge_all()
            summary_writer = tf.summary.FileWriter(
                OUTPUT_PATH / "{}_{}_{}_{}_k={}_out={}_{}".format(timestamp, self.name, str(self.encoding), str(self.hidden_layers),
                                                                  str(self.n_clusters), str(outlier_classes), str(self.gmvae_decode_activation)),
                graph=tf.get_default_graph())

            # Number of batches
            n_batch = (n_samples - 1) // self.minibatch_size + 1

            # Training
            idx = np.arange(x.shape[0])
            np.random.shuffle(idx)

            scores = []
            for epoch in tqdm(range(self.epoch_size)):

                for batch in range(n_batch):
                    i_start = batch * self.minibatch_size
                    i_end = (batch + 1) * self.minibatch_size
                    x_batch = x[idx[i_start:i_end]]

                    feed_dict = {self.input: x_batch}
                    if self.encoder is not None:
                        feed_dict.update({self.encoder: self.encoder_input_data[idx[i_start:i_end]]})
                    self.sess.run([self.minimizer], feed_dict=feed_dict)
                #
                # if epoch % 20 == 0 or epoch == self.epoch_size - 1:
                #     for i in range(self.n_clusters):
                #         px = self.sess.run(self.algorithm.z_representative, feed_dict=feed_dict)
                #         plt.imshow(px[i].reshape(-1, 224, 224, 3)[0, :, :, :])
                #         plt.savefig(OUTPUT_PATH / "{}_representative_{}_k={}_rgb".format(self.name, epoch, i))
                #     print("Loss:", self.sess.run([self.loss], feed_dict=feed_dict))
                #     print("Probability for Gaussians:", self.sess.run([self.algorithm.qy], feed_dict=feed_dict))

                if self.log_performance and (epoch % 10 == 0 or epoch == self.epoch_size - 1):
                    self.algorithm.fit_kmeans(x_test)
                    auc_reconstruction, auc_custom, auc_custom = 0, 0, 0
                    if outlier_classes is not None:
                        auc_reconstruction = roc_auc_score(y_test.isin(outlier_classes), np.nan_to_num(self.algorithm.get_reconstruction_anomaly_scores(x_test)))
                        auc_custom = roc_auc_score(y_test.isin(outlier_classes), np.nan_to_num(self.algorithm.get_custom_anomaly_scores(x_test)))

                        if self.img_dims is not None:
                            test_sample = x_test[y_test.isin(outlier_classes)][0].reshape(1, -1)
                            reconstruction_img_out = self.algorithm.get_reconstruction(test_sample).reshape(1, *self.img_dims, 1)
                            self.algorithm.fit_kmeans(x)
                            closest_centroid = self.algorithm.get_closest_decoded_centroids(test_sample).reshape(1, *self.img_dims, 1)

                    self.algorithm.fit_kmeans(x_test)
                    y_prediction_kmeans = self.algorithm.get_kmeans_assignments()
                    y_prediction_custom = self.algorithm.get_custom_assignments(x_test)
                    ari_kmeans_score = adjusted_rand_score(y_test, y_prediction_kmeans)
                    ari_custom_score = adjusted_rand_score(y_test, y_prediction_custom)
                    accuracy_kmeans = accuracy(y_prediction_kmeans, y_test)
                    accuracy_custom = accuracy(y_prediction_custom, y_test)

                    if self.img_dims is not None:
                        self.algorithm.fit_kmeans(x)
                        cluster_centroids = self.algorithm.get_decoded_centroids().reshape(-1, *self.img_dims, 1)

                    #self.algorithm.fit_kmeans(x)

                    print("Rec-AUC: {:.2f} Custom-AUC: {:.2f} Kmeans-Acc: {:.2f} Custom-Acc: {:.2f} Kmeans-ARI: {:.2f} Custom-ARI: {:.2f}".format(
                        auc_reconstruction, auc_custom, accuracy_kmeans, accuracy_custom, ari_kmeans_score, ari_custom_score))

                    scores.append([auc_reconstruction, auc_custom, accuracy_kmeans, accuracy_custom, ari_kmeans_score, ari_custom_score, epoch])

                    summary_feed = {self.input: x_batch, reconstruction_loss_placeholder: self.algorithm.get_reconstruction_loss(x_test),
                                    auc_reconstruction_placeholder: auc_reconstruction, auc_custom_placeholder: auc_custom,
                                    ari_kmeans_placeholder: ari_kmeans_score, ari_custom_placeholder: ari_custom_score,
                                    accuracy_kmeans_placeholder: ari_kmeans_score, accuracy_custom_placeholder: accuracy_custom}

                    if outlier_classes is not None and self.img_dims is not None:
                        summary_feed.update({reconstruction_img_in_placeholder: test_sample.reshape(-1, *self.img_dims, 1),
                                             reconstruction_img_out_placeholder: reconstruction_img_out, cluster_centroids_placeholder: cluster_centroids,
                                             closest_centroid_placeholder: closest_centroid})

                    summary = self.sess.run([merged_summary_op], feed_dict=summary_feed)[0]
                    summary_writer.add_summary(summary, epoch)

                    # self.algorithm.plot_custom_test_set(x_test, y_test, outlier=outlier_class)

            tf.add_to_collection("save", self.input)
            tf.add_to_collection("save", self.z)

            self.saver = tf.train.Saver()
            return scores

    def predict(self, x):
        return self.algorithm.get_custom_anomaly_scores(x)

    def save(self, fdir):
        """ Save trained model to designated directory.
        This method have to be called after training.
        (If not, throw an exception)

        Parameters
        ----------
        fdir : str
            Path of directory trained model is saved.
            If not exists, it is created automatically.
        """
        if self.sess is None:
            raise Exception("Trained model does not exist.")

        if not os.path.exists(fdir):
            os.makedirs(fdir)

        model_path = os.path.join(fdir, self.MODEL_FILENAME)
        self.saver.save(self.sess, model_path)

        if self.normalize:
            scaler_path = os.path.join(fdir, self.SCALER_FILENAME)
            joblib.dump(self.scaler, scaler_path)

    def restore(self, fdir):
        """ Restore trained model from designated directory.

        Parameters
        ----------
        fdir : str
            Path of directory trained model is saved.
        """
        if not os.path.exists(fdir):
            raise Exception("Model directory does not exist.")

        model_path = os.path.join(fdir, self.MODEL_FILENAME)
        meta_path = model_path + ".meta"

        with tf.Graph().as_default() as graph:
            self.graph = graph
            self.sess = tf.Session(graph=graph)
            self.saver = tf.train.import_meta_graph(meta_path)
            self.saver.restore(self.sess, model_path)
            print(tf.get_collection())

            self.input, self.z = tf.get_collection("save")

        if self.normalize:
            scaler_path = os.path.join(fdir, self.SCALER_FILENAME)
            self.scaler = joblib.load(scaler_path)
