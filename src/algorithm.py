import numpy as np
import tensorflow_probability as tfd
from sklearn.cluster import KMeans

tfd = tfd.distributions


class Algorithm(object):

    def __init__(self, hidden_layers, n_clusters):
        self.hidden_layers = hidden_layers
        self.latent_size = hidden_layers[-1]
        self.n_clusters = n_clusters
        self.z, self.x_dash, self.loss, self.minimizer, self.sess, self.kmeans = [None] * 6

    def build_graph(self, x):
        raise NotImplementedError()

    def set_session(self, sess):
        """
        Set the TensorFlow session
        """
        self.sess = sess

    """Anomaly Detection"""

    def get_reconstruction(self, x):
        """
        Return the reconstructions of the given samples
        """
        return self.sess.run(self.x_dash, feed_dict={'x:0': x})

    def get_reconstruction_loss(self, x):
        """
        Get average L2 loss over x
        """
        x_dash = self.get_reconstruction(x)
        return np.mean(np.sum((x - x_dash) ** 2, axis=1))

    def get_reconstruction_anomaly_scores(self, x):
        """
        Return the anomaly scores of x - the higher the score, the higher the probability that the sample is abnormal
        """
        x_dash = self.get_reconstruction(x)
        anomaly_scores = np.sum((x - x_dash) ** 2, axis=1)
        return anomaly_scores

    def get_custom_anomaly_scores(self, x):
        """Calculate the anomaly scores based on algorithm-specific properties."""
        return self.get_reconstruction_anomaly_scores(x)

    """Clustering"""

    def fit_kmeans(self, x):
        """
        Cluster the latent encoding of x in k clusters
        """
        z = self.sess.run(self.z, feed_dict={'x:0': x})
        self.kmeans = KMeans(n_clusters=self.n_clusters).fit(np.nan_to_num(z))

    def get_kmeans_assignments(self):
        """
        Assign each sample from x one of k clusters via k-means based on z.
        """
        return self.kmeans.labels_

    def get_kmeans_centroids(self):
        """
        Assign each sample from x one of k clusters via k-means based on z.
        """
        return self.kmeans.cluster_centers_

    def get_custom_assignments(self, x):
        """
        Assign samples to clusters in a custom way
        """
        return self.get_kmeans_assignments()

    def get_decoded_centroids(self):
        """
        Return decoded centroids of shape (k x n_features)
        """
        raise NotImplementedError()

    def get_closest_decoded_centroids(self, x):
        decoded_centroids = self.get_decoded_centroids()
        assignments = self.kmeans.predict(self.sess.run(self.z, feed_dict={'x:0': x}))
        closest_centroids = []
        for i in range(x.shape[0]):
            closest_centroids.append(decoded_centroids[assignments[i]])
        return np.asarray(closest_centroids)

    def plot_custom_test_set(self, x, y, outlier=None):
        pass


class Encoding(object):
    AE = "AE"
    VAE = "VAE"
    GMVAE = "GMVAE"