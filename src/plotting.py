import sys
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from deepexplain.tensorflow import DeepExplain
import tensorflow as tf
import matplotlib
from matplotlib import offsetbox
from matplotlib import ticker
from matplotlib.cm import get_cmap
from matplotlib.lines import Line2D
from scipy.spatial.distance import cosine
from sklearn.metrics import auc
from scipy.stats import multivariate_normal
from sklearn.manifold import TSNE


def plot_energy_distribution(energy):
    plt.figure(figsize=[8, 3])
    plt.hist(energy, bins=50, log=True)
    plt.xlabel("DAGMM Energy")
    plt.ylabel("Number of Sample(s)")
    plt.show()


def plot_energy_per_sample(energy):
    plt.figure(figsize=[8, 3])
    plt.plot(energy, "o-")
    plt.xlabel("Index (row) of Sample")
    plt.ylabel("Energy")
    plt.show()


def plot_2d_outliers_by_energy(df, energy):
    plt.plot(df.iloc[:, 0], df.iloc[:, 1], ".")
    ano_index = np.arange(len(energy))[energy > np.percentile(energy, 99.8)]
    plt.plot(df.iloc[ano_index, 0], df.iloc[ano_index, 1], "x", c="r", markersize=8)
    plt.show()


def plot_2d_scatter_by_energy(df, energy):
    plt.scatter(df.iloc[:, 0], df.iloc[:, 1], s=energy + 70, cmap=plt.cm.YlOrRd, c=energy + 15)
    plt.show()


def plot_representatives(mu, z, x):
    n_gaussians = mu.shape[0]
    n_samples = z.shape[0]
    idx_mins = np.zeros(n_gaussians, dtype=int)
    for gauss_index in range(n_gaussians):
        min_distance = 1e10
        idx_min = -1
        for sample_idx in range(n_samples):
            cosine_distance = cosine(z[sample_idx, :], mu[gauss_index, :]).sum()
            if cosine_distance < min_distance:
                min_distance = cosine_distance
                idx_min = sample_idx
        idx_mins[gauss_index] = idx_min
    fig, axes = plt.subplots(nrows=1, ncols=n_gaussians, figsize=(3 * n_gaussians, 3))
    for i in range(n_gaussians):
        plot_sub_explanation(x[idx_mins[i], :].reshape(28, 28), axis=axes[i]).set_title("Mean of Gaussian " + str(i))
    plt.show()


def plot_representatives_with_gamma(x, gamma, title=None):
    idx_mins = np.argmax(gamma, axis=0)
    n_gaussians = gamma.shape[1]
    assert idx_mins.shape[0] == n_gaussians
    fig, axes = plt.subplots(nrows=1, ncols=n_gaussians, figsize=(3 * n_gaussians, 3))
    if title is not None:
        fig.suptitle(title)
    for i in range(n_gaussians):
        plot_sub_explanation(x[idx_mins[i], :].reshape(28, 28), axis=axes[i]).set_title("Mean of Gaussian " + str(i))
    return plt


def plot_parallel_coordinates(df):
    #    pd.plotting.parallel_coordinates(df, 'y', color=('#0000FF', '#FF0000'), cols=df.columns)
    #    plt.show()

    df = df.copy()
    df['y'] = pd.Categorical(df.y).as_ordered()
    cols = list(df.columns)
    cols.remove('y')
    x = [i for i, _ in enumerate(cols)]
    n_classes = len(df['y'].unique())
    cmap = plt.get_cmap('jet', n_classes)

    # create dict of categories: colours
    colours = {df['y'].cat.categories[i]: cmap(i / n_classes) for i, _ in enumerate(df['y'].cat.categories)}

    # Create (X-1) sublots along x axis
    fig, axes = plt.subplots(1, len(x) - 1, sharey=False, figsize=(15, 5))

    # Get min, max and range for each column
    # Normalize the data for each column
    min_max_range = {}
    for col in cols:
        min_max_range[col] = [df[col].min(), df[col].max(), np.ptp(df[col])]
        df[col] = np.true_divide(df[col] - df[col].min(), np.ptp(df[col]))

    # Plot each row
    for i, ax in enumerate(axes):
        for idx in df.index:
            y_category = df.loc[idx, 'y']
            ax.plot(x, df.loc[idx, cols], colours[y_category], alpha=0.5)
        ax.set_xlim([x[i], x[i + 1]])

    # Set the tick positions and labels on y axis for each plot
    # Tick positions based on normalised data
    # Tick labels are based on original data
    def set_ticks_for_axis(dim, ax, ticks):
        min_val, max_val, val_range = min_max_range[cols[dim]]
        step = val_range / float(ticks - 1)
        tick_labels = [round(min_val + step * i, 2) for i in range(ticks)]
        norm_min = df[cols[dim]].min()
        norm_range = np.ptp(df[cols[dim]])
        norm_step = norm_range / float(ticks - 1)
        ticks = [round(norm_min + norm_step * i, 2) for i in range(ticks)]
        ax.yaxis.set_ticks(ticks)
        ax.set_yticklabels(tick_labels)

    for dim, ax in enumerate(axes):
        ax.xaxis.set_major_locator(ticker.FixedLocator([dim]))
        set_ticks_for_axis(dim, ax, ticks=6)
        ax.set_xticklabels([cols[dim]])

    # Move the final axis' ticks to the right-hand side
    ax = plt.twinx(axes[-1])
    dim = len(axes)
    ax.xaxis.set_major_locator(ticker.FixedLocator([x[-2], x[-1]]))
    set_ticks_for_axis(dim, ax, ticks=6)
    ax.set_xticklabels([cols[-2], cols[-1]])

    # Remove space between subplots
    plt.subplots_adjust(wspace=0)

    # Add legend to plot
    plt.legend(
        [plt.Line2D((0, 1), (0, 0), color=colours[cat]) for cat in df['y'].cat.categories],
        df['y'].cat.categories,
        bbox_to_anchor=(1.2, 1), loc=2, borderaxespad=0.)

    plt.show()


def plot_error_by_energy(error, energy, y=None, title=None):
    """
    :param error: Error vector
    :param energy: Energy vector
    :param title: Title of the plot
    :param y: Vector that indicates an anomaly with 1
    """
    plt.plot(energy, error, '.')
    plt.xlabel("Energy")
    plt.ylabel("Reconstruction Error")
    if title is not None:
        plt.title(title)
    if y is not None:
        for en, er in zip(energy[y == 1], error[y == 1]):
            plt.plot(en, er, 'x', color='r')
    plt.show()


def plot_embedding(z, digits, title=None, xlabel="Latent Dimension"):
    """Plot the reconstruction error as a function of the latent dimension
    and add small thumbnails of the original images"""
    n_samples, n_features = z.shape
    reconstruction_error_index = -2
    digits = digits.reshape(-1, 28, 28)
    error = z[:, reconstruction_error_index]
    for latent_index in range(n_features - 2):
        latent = z[:, latent_index]
        plt.figure(figsize=(20, 20))
        ax = plt.subplot(111)
        if hasattr(offsetbox, 'AnnotationBbox'):
            shown_images = np.array([[1., 1.]])  # just something big
            for i in range(n_samples):
                X_instance = z[i, (latent_index, reconstruction_error_index)]
                dist = np.sum((X_instance - shown_images) ** 2, 1)
                if np.min(dist) < 4e-5:
                    # don't show points that are too close
                    continue
                shown_images = np.r_[shown_images, [X_instance]]
                imagebox = offsetbox.AnnotationBbox(offsetbox.OffsetImage(digits[i], cmap=plt.cm.gray_r), X_instance)
                ax.add_artist(imagebox)
        plt.xlim((latent.min(), 1.1 * latent.max()))
        plt.ylim((0.9 * error.min(), 1.1 * error.max()))
        plt.xlabel(xlabel + " " + str(latent_index + 1))
        plt.ylabel("Reconstruction Error")
        if title is not None:
            plt.title(title)
    plt.show()


def plot_embedding_with_gaussians(z, digits, gamma, title=None, xlabel="Latent Dimension"):
    """Plot the reconstruction error as a function of the latent dimension
    and add small thumbnails of the original images"""
    n_samples, n_features = z.shape
    n_gaussians = gamma.shape[1]
    reconstruction_error_index = -2
    digits = digits.reshape(-1, 28, 28)
    error = z[:, reconstruction_error_index]
    cmap = get_cmap('inferno')
    from collections import defaultdict
    for latent_index in range(n_features):
        used_gaussian_indexes = defaultdict(lambda: 0)
        latent = z[:, latent_index]
        plt.figure(figsize=(20, 20))
        ax = plt.subplot(111)
        if hasattr(offsetbox, 'AnnotationBbox'):
            shown_images = np.array([[1., 1.]])  # just something big
            for i in range(n_samples):
                X_instance = z[i, (latent_index, reconstruction_error_index)]
                gaussian_index = np.argmin(gamma[i, :])
                used_gaussian_indexes[gaussian_index] += 1
                dist = np.sum((X_instance - shown_images) ** 2, 1)
                if np.min(dist) < 4e-4:
                    # don't show points that are too close
                    continue
                shown_images = np.r_[shown_images, [X_instance]]
                color = cmap(gaussian_index / n_gaussians)
                offset_image = offsetbox.OffsetImage(digits[i], cmap=plt.cm.gray_r)
                imagebox = offsetbox.AnnotationBbox(offset_image, X_instance, bboxprops=dict(edgecolor=color, linewidth=3))
                ax.add_artist(imagebox)

        # Build Legend
        elements = []
        for index in range(n_gaussians):
            elements.append(Line2D([0], [0], color=cmap(index / n_gaussians), lw=4, label="Gaussian {0} ({1:.2%})".format(index, used_gaussian_indexes[index] / n_samples)))
        plt.legend(handles=elements)

        plt.xlim((latent.min(), 1.1 * latent.max()))
        plt.ylim((0.9 * error.min(), 1.1 * error.max()))
        plt.xlabel(xlabel + " " + str(latent_index + 1))
        plt.ylabel("Reconstruction Error")
        if title is not None:
            plt.title(title)
    return plt


def plot_latent_distributions(mu, sigma, z, y=None):
    """Plot all latent distributions with all fitted Gaussian bells. Optional: pass y to mark outliers"""

    class nf(float):
        def __repr__(self):
            str = '%.1f' % (self.__float__(),)
            if str[-1] == '0':
                return '%.0f' % self.__float__()
            else:
                return '%.1f' % self.__float__()

    num_latent_dimensions = mu.shape[1] - 2  # subtract two error functions
    num_gaussians = mu.shape[0]
    reconstruction_error_index = -2

    for latent_index in range(num_latent_dimensions):
        for gaussian_index in range(num_gaussians):
            latent = z[:, (latent_index, reconstruction_error_index)]
            mu_small = [mu[gaussian_index, latent_index], mu[gaussian_index, reconstruction_error_index]]
            sigma_small = np.zeros((2, 2))  # Variance in latent dimension x reconstruction error
            sigma_small[0, :] = np.array([sigma[gaussian_index, latent_index, latent_index],
                                          sigma[gaussian_index, latent_index, reconstruction_error_index]])
            sigma_small[1, :] = np.array([sigma[gaussian_index, reconstruction_error_index, latent_index],
                                          sigma[gaussian_index, reconstruction_error_index, reconstruction_error_index]])

            x_line = np.linspace(latent[:, 0].min(), latent[:, 0].max(), 100)
            y_line = np.linspace(latent[:, 1].min(), latent[:, 1].max(), 100)
            X, Y = np.meshgrid(x_line, y_line)
            pos = np.dstack((X, Y))

            rv = multivariate_normal(mu_small, sigma_small)
            Z = rv.pdf(pos)

            # Basic contour plot
            fig, ax = plt.subplots()
            CS = ax.contour(X, Y, Z)

            # Recast levels to new class
            CS.levels = [nf(val) for val in CS.levels]

            # Label levels with specially formatted floats
            if plt.rcParams["text.usetex"]:
                fmt = r'%r \%%'
            else:
                fmt = '%r %%'

            ax.clabel(CS, CS.levels, inline=True, fmt=fmt, fontsize=10)
            plt.plot(latent[:, 0], latent[:, 1], '.', alpha=0.15)
            plt.title("Gaussian " + str(gaussian_index + 1) + " on latent dimension " + str(latent_index + 1))
            plt.xlabel("z" + str(latent_index + 1))
            plt.ylabel("Reconstruction error")
            if y is not None:
                for anomaly in latent[y == 1]:
                    plt.plot(anomaly[0], anomaly[1], 'x', color='r')
    plt.show()


def plot_sub_explanation(data, cmap='RdBu_r', axis=plt, percentile=100):
    """Helper function to plot DeepExplain maps."""
    dx, dy = 0.05, 0.05
    xx = np.arange(0.0, data.shape[1], dx)
    yy = np.arange(0.0, data.shape[0], dy)
    xmin, xmax, ymin, ymax = np.amin(xx), np.amax(xx), np.amin(yy), np.amax(yy)
    extent = xmin, xmax, ymin, ymax
    cmap_xi = plt.get_cmap('Greys_r')
    cmap_xi.set_bad(alpha=0)

    abs_max = np.percentile(np.abs(data), percentile)
    abs_min = abs_max

    if len(data.shape) == 3:
        data = np.mean(data, 2)
    data[data == 0] = None
    axis.imshow(data, extent=extent, interpolation='none', cmap=cmap)#, vmin=-abs_min, vmax=abs_max)
    axis.axis('off')
    return axis


def plot_explanations(data, input_, output, model):
    assert tf.gradients(output, [input_])[0] is not None, "{} wrt. {} has no gradients".format(output, input_)
    with DeepExplain(session=model.sess, graph=model.graph) as de:
        attributions = {
            # Gradient-based
            'Saliency maps': de.explain('saliency', output, input_, data),
            'Gradient * Input': de.explain('grad*input', output, input_, data),
            'Integrated Gradients': de.explain('intgrad', output, input_, data),
            'Epsilon-LRP': de.explain('elrp', output, input_, data),
            'DeepLIFT (Rescale)': de.explain('deeplift', output, input_, data),
            # Perturbation-based
            '_Occlusion [1x1]': de.explain('occlusion', output, input_, data),
            '_Occlusion [3x3]': de.explain('occlusion', output, input_, data, window_shape=(3,))
        }

    n_cols = len(attributions) + 1
    xi = data.reshape(1, -1)
    fig, axes = plt.subplots(nrows=1, ncols=n_cols, figsize=(3 * n_cols, 3))
    plot_sub_explanation(xi.reshape(28, 28), cmap='Greys', axis=axes[0]).set_title('Original')
    for i, method_name in enumerate(sorted(attributions.keys())):
        plot_sub_explanation(attributions[method_name].reshape(28, 28), axis=axes[1 + i]).set_title(method_name)


def plot_tsne(z, y_test):
    X_embedded = TSNE(n_components=2).fit_transform(z)
    cmap = matplotlib.cm.get_cmap('inferno')
    fig, ax = plt.subplots(figsize=(20, 10))
    for label in np.unique(y_test):
        ix = np.where(y_test == label)
        ax.scatter(X_embedded[ix, 0], X_embedded[ix, 1], c=[cmap(label / 10.0)], label=label, s=100)
    ax.legend()


def plot_latent_code(fig, ax, n_clusters, epoch, z_result, qy_result, representatives, y_true, img_dims, outlier=None):
    """ Plot latent code and representatives of all clusters.
    :param epoch: epoch
    :param z_result: (k, -1, n_latent) Latent representation under each Gaussian
    :param qy_result: (-1, k) Probability of belonging to Gaussian k
    :param representatives: (k, n_features) Sample with highest probability of belonging to Gaussian k
    :return:
    """

    for k in range(n_clusters):
        plot_z_with_labels(np.asarray(z_result[k]), y_true, qy_result[:, k], ax=ax[0][k], outlier=outlier)
        ax[1][k].imshow(representatives[k].reshape(*img_dims))
    fig.savefig("Gaussian_z_{}".format(str(epoch).zfill(5)), dpi=100)


def plot_z_with_labels(z, y_test, prob, ax, outlier=None):
    if z.shape[1] != 2:
        z = z[:, :2]
        #z = PCA(n_components=2).fit_transform(z)

    cmap = matplotlib.cm.get_cmap('inferno')
    for label in np.unique(y_test):
        if label != outlier:
            s, c, m = 2, [cmap(label / 10.0)], '.'
        else:
            s, c, m = 10, ['red'], 'x'
        ix = np.where(y_test == label)
        for ix in ix:
            ax.scatter(z[ix, 0], z[ix, 1], c=c, label=label, s=s, alpha=prob[ix][0], marker=m)
    ax.legend()
    return ax


def plot_interpretation(data, model, output):
    input_ = model.input

    with DeepExplain(session=model.sess, graph=model.graph) as de:
        attributions = {
            # Gradient-based
            'Saliency maps': de.explain('saliency', output, input_, data),
            'Gradient * Input': de.explain('grad*input', output, input_, data),
            'Integrated Gradients': de.explain('intgrad', output, input_, data),
            'Epsilon-LRP': de.explain('elrp', output, input_, data),
            'DeepLIFT (Rescale)': de.explain('deeplift', output, input_, data),
            # Perturbation-based
            '_Occlusion [1x1]': de.explain('occlusion', output, input_, data),
            '_Occlusion [3x3]': de.explain('occlusion', output, input_, data, window_shape=(3,))
        }

    n_cols = len(attributions) + 3
    xi = data.reshape(1, -1)

    x_dash = model.sess.run(model.x_dash, feed_dict={'x:0': xi})
    centroid = model.algorithm.get_closest_decoded_centroids(xi)

    fig, axes = plt.subplots(nrows=1, ncols=n_cols, figsize=(3 * n_cols, 3))
    plot_sub_explanation(xi.reshape(28, 28), cmap='Greys', axis=axes[0]).set_title('Original')
    plot_sub_explanation(x_dash.reshape(28, 28), cmap='RdBu_r', axis=axes[1]).set_title('Reconstruction')
    plot_sub_explanation(centroid.reshape(28, 28), cmap='RdBu_r', axis=axes[2]).set_title('Centroid')
    for i, method_name in enumerate(sorted(attributions.keys())):
        plot_sub_explanation(attributions[method_name].reshape(28, 28), axis=axes[3 + i]).set_title(method_name)


def plot_interpretation_paper(data, model, output, model_ae, set_title=True):
    input_ = model.input

    with DeepExplain(session=model.sess, graph=model.graph) as de:
        attributions = {
            # Gradient-based
            'Saliency maps': de.explain('saliency', output, input_, data),
            'Gradient * Input': de.explain('grad*input', output, input_, data),
            'Integrated\nGradients': de.explain('intgrad', output, input_, data),
            'Epsilon-LRP': de.explain('elrp', output, input_, data),
            'DeepLIFT (Rescale)': de.explain('deeplift', output, input_, data),
            # Perturbation-based
            'Occlusion [1x1]': de.explain('occlusion', output, input_, data),
            'Occlusion [3x3]': de.explain('occlusion', output, input_, data, window_shape=(3,))
        }

    n_cols = len(attributions) + 2
    xi = data.reshape(1, -1)

    x_dash = model_ae.sess.run(model_ae.x_dash, feed_dict={'x:0': xi})
    fig, axes = plt.subplots(nrows=1, ncols=n_cols, figsize=(3 * n_cols, 3))
    plot_sub_explanation(xi.reshape(28, 28), cmap='Greys', axis=axes[0])
    plot_sub_explanation(data.reshape(28, 28) - x_dash.reshape(28, 28), cmap='RdBu_r', axis=axes[1])

    if set_title:
        axes[0].set_title('Original', fontsize=20)
        axes[1].set_title('Reconstruction\nHeatmap', fontsize=20)

    for i, method_name in enumerate(sorted(attributions.keys())):
        plot_sub_explanation(attributions[method_name].reshape(28, 28), axis=axes[2 + i])
        if set_title:
            axes[2+i].set_title(method_name, fontsize=20)


def get_optimal_pixel_value(sample, model, pixel_index):
    min_loss = 1e12
    min_value = -1
    for pixel_value in [0, 255]:
        sample[0, pixel_index] = pixel_value / 255.0
        loss = model.sess.run(model.algorithm.loss, feed_dict={'x:0': sample, 'vary_z:0': [0]})[0]
        if loss < min_loss:
            min_loss = loss
            min_value = pixel_value / 255.0
    return min_value


def get_flipped_image(data, model, output, attribution_method='_Occlusion [3x3]', n_pixels=200, intelligent=True, use_mask=False):
    """Flip the given sample towards normality w.r.t. the output method using the attribution method
    Returns the flipped image, the raw importance pixels and the scores"""
    input_ = model.input
    if use_mask:
        y_pred = model.sess.run(model.algorithm.qy_logit, feed_dict={'x:0': data}).argmax()
        mask = np.zeros((1, 10))
        mask[0, y_pred] = 1
        output *= mask

    with DeepExplain(session=model.sess, graph=model.graph) as de:
        if attribution_method == 'Random':
            res = np.expand_dims(np.random.permutation(data.shape[1]), axis=0)
        elif attribution_method == 'Saliency maps':
            res = de.explain('saliency', output, input_, data)
        elif attribution_method == 'Gradient * Input':
            res = de.explain('grad*input', output, input_, data)
        elif attribution_method == 'Integrated Gradients':
            res = de.explain('intgrad', output, input_, data)
        elif attribution_method == 'Epsilon-LRP':
            res = de.explain('elrp', output, input_, data)
        elif attribution_method == 'DeepLIFT (Rescale)':
            res = de.explain('deeplift', output, input_, data)
        elif attribution_method == '_Occlusion [1x1]':
            res = de.explain('occlusion', output, input_, data)
        elif attribution_method == '_Occlusion [3x3]':
            res = de.explain('occlusion', output, input_, data, window_shape=(3,))
        else:
            raise Exception("Unknown attribution method: " + attribution_method)
    scores = []
    important_pixels = (-res).argsort()[0]
    test_sample_flipping = data.copy()
    for i in range(n_pixels):
        new_value = get_optimal_pixel_value(test_sample_flipping.copy(), model, important_pixels[i]) if intelligent else 1 - test_sample_flipping[0][important_pixels[i]]
        test_sample_flipping[0][important_pixels[i]] = new_value
        scores.append(model.algorithm.get_custom_anomaly_scores(test_sample_flipping))

    return test_sample_flipping, res, scores


def get_flipping_auc_scores(data, model, output, n_pixels=100):
    methods = ['Random', 'Saliency maps', 'Gradient * Input', 'Integrated Gradients', 'Epsilon-LRP',
               'DeepLIFT (Rescale)', '_Occlusion [1x1]', '_Occlusion [3x3]']

    auc_scores = []
    for sample in data:
        sample_scores = []
        for method_index, method in enumerate(methods):
            method_image, res, scores = get_flipped_image(np.expand_dims(sample, axis=0), model, output, attribution_method=method, n_pixels=n_pixels)
            auc_score = auc(list(range(len(scores))), scores)
            sample_scores.append(auc_score)
        auc_scores.append(sample_scores)
    return auc_scores


def plot_pixel_flipping(data, model, output, n_pixels=200, intelligent=True, use_mask=False, model_ae=None, plot_scores=False):
    """Intelligent == True flips pixel to minimize loss of the image while False flips to 1-value"""

    methods = ['Random', 'Saliency maps', 'Gradient * Input', 'Integrated Gradients', 'Epsilon-LRP', 'DeepLIFT (Rescale)', '_Occlusion [1x1]', '_Occlusion [3x3]']
    n_rows = 3 if plot_scores else 2
    n_cols = len(methods) if model_ae is None else len(methods) + 1
    plt.subplots_adjust(wspace=0, hspace=0)
    fig, axes = plt.subplots(nrows=n_rows, ncols=n_cols, figsize=(3 * len(methods), 5), gridspec_kw = {'wspace':0, 'hspace':0})
    y_min, y_max = sys.maxsize, -sys.maxsize
    if model_ae is not None:
        baseline = model_ae.sess.run(model_ae.x_dash, {'x:0': data.reshape(1, -1)}).reshape(28, 28)
        #axes[0][0].set_title("AE Rec.", fontsize=20)
        axes[0][0].imshow(data.reshape(28, 28) - baseline, cmap="Reds")
        axes[0][0].axis("off")
        axes[1][0].axis("off")
        axes[1][0].imshow(baseline, cmap='Greys')

    for method_index, method in enumerate(methods):
        if model_ae is not None:
            method_index += 1
        
        method_image, res, scores = get_flipped_image(data, model, output, attribution_method=method, n_pixels=n_pixels, intelligent=intelligent, use_mask=use_mask)
        y_min = min(y_min, min(scores))
        y_max = max(y_max, max(scores))
        plot_sub_explanation(res.reshape(28, 28), axis=axes[0][method_index]).set_title(method, fontsize=20)
        axes[1][method_index].imshow(method_image.reshape(28, 28), cmap='Greys')
        axes[1][method_index].axis("off")
        if plot_scores:
            axes[2][method_index].set_title("%.2f" % auc(list(range(len(scores))), scores))
            axes[2][method_index].plot(scores)
        if method_index == 0 and plot_scores:
            axes[2][method_index].set_xlabel("Normalized Pixels")
            axes[2][method_index].set_ylabel("Anomaly Score")

    if plot_scores:
        for method_index, _ in enumerate(methods):
            axes[2][method_index].set_ylim((y_min * 0.99, y_max * 1.01))

    plt.subplots_adjust(wspace=0, hspace=0)
    plt.savefig("pixel_flipping_{}")
