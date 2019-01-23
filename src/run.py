"""This file should plot the 10x10 matrices of how to interpret different outliers."""

import sys

sys.path.append("..")
import numpy as np

np.set_printoptions(threshold=np.nan)
import matplotlib.pyplot as plt
from src.data_loading import load_mnist
from src.experiment import Experiment
import tensorflow as tf
import numpy as np
from deepexplain.tensorflow import DeepExplain

tf.logging.set_verbosity(tf.logging.INFO)

df = load_mnist()
df.sample(frac=1).reset_index(drop=True, inplace=True)
y = df['y']
del df['y']


def plot(data, xi=None, cmap='RdBu_r', axis=plt, percentile=100, dilation=3.0, alpha=0.8):
    dx, dy = 0.05, 0.05
    xx = np.arange(0.0, data.shape[1], dx)
    yy = np.arange(0.0, data.shape[0], dy)
    xmin, xmax, ymin, ymax = np.amin(xx), np.amax(xx), np.amin(yy), np.amax(yy)
    extent = xmin, xmax, ymin, ymax
    cmap_xi = plt.get_cmap('Greys_r')
    cmap_xi.set_bad(alpha=0)
    overlay = None
    if xi is not None:
        # Compute edges (to overlay to heatmaps later)
        xi_greyscale = xi if len(xi.shape) == 2 else np.mean(xi, axis=-1)
        in_image_upscaled = transform.rescale(xi_greyscale, dilation, mode='constant')
        edges = feature.canny(in_image_upscaled).astype(float)
        edges[edges < 0.5] = np.nan
        edges[:5, :] = np.nan
        edges[-5:, :] = np.nan
        edges[:, :5] = np.nan
        edges[:, -5:] = np.nan
        overlay = edges

    abs_max = np.percentile(np.abs(data), percentile)
    abs_min = abs_max

    if len(data.shape) == 3:
        data = np.mean(data, 2)
    axis.imshow(data, extent=extent, interpolation='none', cmap=cmap, vmin=-abs_min, vmax=abs_max)
    if overlay is not None:
        axis.imshow(overlay, extent=extent, interpolation='none', cmap=cmap_xi, alpha=alpha)
    axis.axis('off')
    return axis


def plot_explanations(data, input_, output, model_dagmm, axes_list, inlier_index, outlier_index):
    with DeepExplain(session=model_dagmm.sess) as de:
        attributions = {
            # Gradient-based
            'Saliency maps': de.explain('saliency', output, input_, data),
            # 'Gradient * Input':     de.explain('grad*input', output, input_, data),
            # 'Integrated Gradients': de.explain('intgrad',  output, input_, data),
            'Epsilon-LRP': de.explain('elrp', output, input_, data),
            # 'DeepLIFT (Rescale)':   de.explain('deeplift',  output, input_, data),
            # Perturbation-based
            # '_Occlusion [1x1]':      de.explain('occlusion',  output, input_, data),
            # '_Occlusion [3x3]':      de.explain('occlusion',  output, input_, data, window_shape=(3,))
        }

    n_cols = len(attributions)
    xi = data.reshape(1, -1)
    for method_id, method_name in enumerate(sorted(attributions.keys())):
        axis = plot(attributions[method_name].reshape(28, 28), xi=None, axis=axes_list[method_id][inlier_index][outlier_index])


EPOCHS = 100
config_dagmm = {'comp_hiddens': [32, 16, 14], 'est_hiddens': [14, 12], 'est_dropout_ratio': 0.1, 'only_ae': False, 'epoch_size': EPOCHS,
                'minibatch_size': 128, 'normalize': False, 'random_seed': 0, 'warm_up_epochs': 0}

plt.tight_layout()
plot_settings = {'ncols': 10, 'nrows': 10, 'figsize': (21, 21)}

# DAGMM
# Rec error
fig_rec, axes_rec = plt.subplots(**plot_settings)
# Saliency
fig_sal, axes_sal = plt.subplots(**plot_settings)
# LRP
fig_lrp, axes_lrp = plt.subplots(**plot_settings)

for i in range(10):
    for j in range(10):
        for axes in [axes_rec, axes_sal, axes_lrp]:
            axes[i][j].set_axis_off()

for fig, title in zip([fig_rec, fig_sal, fig_lrp], ["Reconstruction Error", "Saliency", "LRP"]):
    fig.suptitle(title)

for inlier in range(1):
    outliers = list(range(10))
    X = df[y == inlier].values

    tf.reset_default_graph()
    model = Experiment(**config_dagmm)
    model.fit(X)
    for outlier in outliers:
        x_outlier = df[y == outlier].iloc[0, :].values.reshape(1, -1)
        energy, mu, sigma, z, x_dash, gamma = model.predict(x_outlier)

        for axes in [axes_rec, axes_sal, axes_lrp]:
            axes[inlier][outlier].set_title("{} vs. {}".format(inlier, outlier))

        # Rec Error
        axes_rec[inlier][outlier].imshow(x_outlier.reshape(28, 28) - x_dash.reshape(28, 28), vmin=-1, vmax=1)

        # LRP + Sensitiviy on Energy
        plot_explanations(x_outlier, model.input, model.energy, model, [axes_sal, axes_lrp], inlier, outlier)
plt.show()
