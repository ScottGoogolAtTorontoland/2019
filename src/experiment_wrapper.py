from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import roc_auc_score
import itertools

from src.experiment import Experiment
from src.plotting import plot_embedding, plot_latent_distributions, plot_error_by_energy, plot_explanations, \
    plot_parallel_coordinates, plot_sub_explanation
from tqdm import tqdm


class ExperimentWrapper:
    def __init__(self, df, normal_classes, outlier_class=None, config=None, normalize=False, plot_train=True,
                 plot_test=True, explain_training=False, explain_testing=False, df_test=None, img_dims=None, name="MNIST"):
        """
        Initialize the experiment
        :param df: DataFrame containing the total data. The 'y' column contains the labels.
        :param normal_classes: list of normal classes
        :param outlier_class: an outlier class (optional)
        """
        df = df.astype(float)
        if normalize:
            y = df['y']
            del df['y']
            df = (df - df.mean()) / (df.max() - df.min())
            df['y'] = y
        self.df = df
        self.normal_classes = [] if normal_classes is None else normal_classes
        self.outlier_class = outlier_class
        self.config = {'comp_hiddens': [32, 16, 8], 'epoch_size': 250, 'minibatch_size': 128, 'normalize': False, 'name': name, 'img_dims': img_dims}
        self.models = defaultdict(list)
        self.df_test = df_test

        if config is not None:
            self.config.update(config)
        self.plot_train = plot_train
        self.plot_test = plot_test
        self.explain_training = explain_training
        self.explain_testing = explain_testing

    def run(self, pollution_space=None):
        if pollution_space is None:
            pollution_space = [0]
        scores = defaultdict(list)
        for pollution in pollution_space:
            score = self.run_with_pollution()
            scores[self.outlier_class][pollution] = score
        return scores

    def run_with_pollution(self, pollution, train_fraction=0.8):
        """
        Evaluate the ROC AUC score given a pollution fraction.
        :param pollution: Proportion of outliers that should stay in the data.
        :param train_fraction:
        :return: ROC AUC score for the experiment setup
        """
        assert train_fraction <= 1, "Fraction to train on must not be bigger than 1"
        assert pollution <= 1, "Fraction of anomalies to keep in training data must not be bigger than 1"

        if sum(self.df['y'] == self.outlier_class) == 0:
            raise Exception("There are no outliers in the given DataFrame")

        test_contains_outliers = False
        df = self.df.copy()

        if self.df_test is None:
            while not test_contains_outliers:
                df_shuffled = df.sample(frac=1).reset_index(drop=True)
                df_train = df_shuffled.iloc[:int(train_fraction * (len(df))), :]
                df_test = df_shuffled.iloc[int(train_fraction * (len(df))):, :]
                y_train = df_train['y']
                y_test = df_test['y']
                test_contains_outliers = sum(y_test == self.outlier_class) > 0
        else:
            df_test = self.df_test.copy()
            y_test = df_test['y']

        train_anomalies = df_train[y_train == self.outlier_class]
        train_anomalies = df_train.iloc[:int(pollution * len(train_anomalies)), :]
        df_train = df_train[y_train != self.outlier_class]
        df_train = df_train.loc[y_train.isin(self.normal_classes)]
        df_train = df_train.append(train_anomalies)
        y_train = df_train['y']
        test_labels = self.normal_classes.copy()
        if self.outlier_class is not None:
            test_labels.append(self.outlier_class)
        df_test = df_test[y_test.isin(test_labels)]
        y_test = df_test['y']
        del df_train['y'], df_test['y']

        tf.reset_default_graph()
        model = Experiment(**self.config)
        model.fit(df_train.values)

        for mode, visualize_model, explain_predictions, x, y in zip(["Training", "Testing"], [self.plot_train, self.plot_test],
                                                                    [self.explain_training, self.explain_testing], [df_train.values, df_test.values],
                                                                    [y_train == self.outlier_class, y_test == self.outlier_class]):
            print(mode)
            energy = None
            if visualize_model:
                # Visualize Trained Model
                energy, mu, sigma, z, x_dash = model.predict(x)
                try:
                    # Plot distribution of latent dimensions with respective images
                    x.reshape(-1, 28, 28)
                    # Plot Reconstruction-Error vs. Latent Dimensions with Images
                    plot_embedding(z, x, xlabel="Latent Dimension")
                    errors = z[:, (-2, -1)]
                    energy_errors = np.insert(errors, 0, 0, axis=1)
                    energy_errors[:, 0] = energy
                    # Plot Reconstruction-Error vs. Energy with Images
                    plot_embedding(energy_errors, x, xlabel="Energy")
                except ValueError as _:
                    # Plot parallel coordinates
                    df_latent = pd.DataFrame()
                    df_latent['y'] = y
                    for i in range(z.shape[1] - 2):
                        df_latent[i] = z[:, i]
                    for idx, error in enumerate(["reconstruction", "cosine"]):
                        df_latent[error] = z[:, -2 + idx]
                    assert df_latent[df_latent.isnull().any(axis=1)].shape[0] == 0, "df_latent contains NaN's"
                    plot_parallel_coordinates(df_latent)

                # Plot Reconstruction-Error vs. Latent Dimensions with Gaussians and Labels
                plot_latent_distributions(mu, sigma, z, y=y)
                # Plot Reconstruction-Error vs. Energy with Labels
                plot_error_by_energy(z[:, -2], energy, y=y, title=str(self.normal_classes) + " vs. " + str(self.outlier_class))

            if explain_predictions:
                if energy is None:
                    energy, mu, sigma, z, x_dash = model.predict(x)
                input_ = model.input
                targets = ['reconstruction_loss', 'energy', 'z']
                for arg_function in [np.argmin, np.argmax]:
                    print(arg_function)
                    data = x[arg_function(energy), :]
                    x_diff = (data - x_dash[arg_function(energy), :])
                    x_diff = x_diff.reshape(28, 28)
                    fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(3 * 8, 3))
                    plot_sub_explanation(x_diff, axis=axes).set_title("Reconstruction Error")
                    plt.plot()
                    for target in targets:
                        print(target)
                        plot_explanations(data.reshape(1, -1), input_, model.__getattribute__(target), model)
                        plt.show()
                    print("µ")
                    plot_explanations(data.reshape(1, -1), input_, model.gmm.mu_org, model)
                    plt.show()

        if self.outlier_class is None:
            return 1
        energy, _, _, _, _ = model.predict(df_test.values)
        score = roc_auc_score(y_test == self.outlier_class, energy)
        return score

    def evaluate_hyperparameters(self, hyperparams, epochs=250, seeds=2, clean_training=True, plot_boxplots=False):
        """

        :param hyperparams: {"warm_up_epochs": [True, False], ...}
        :return:
        """
        df = self.df.copy()
        y = df["y"]
        del df["y"]
        self.config.update({"epoch_size": epochs})

        # Keep inlier and outlier
        mask = y.isin(self.normal_classes) | (y == self.outlier_class)
        df = df[mask]
        y = y[mask]

        if self.df_test is None:
            # Split
            mask = np.random.randn(len(df)) < 0.8
            df_test = df[~mask]
            y_test = y[~mask]
            df = df[mask]
            y = y[mask]
        else:
            df_test = self.df_test.copy()
            y_test = df_test["y"]
            del df_test["y"]

        if clean_training:
            # Remove outlier from training
            df = df[y != self.outlier_class]
            y = y[y != self.outlier_class]

        assert sum(y_test == self.outlier_class) > 0

        df_runs = None

        hyperparam_combinations = list(itertools.product(*hyperparams.values()))
        with tqdm(total=seeds * len(list(hyperparam_combinations))) as pbar:
            for seed in range(seeds):
                self.config.update({"random_seed": seed})
                for hyperparam_combination in hyperparam_combinations:
                    name_value_list = list(zip(hyperparams.keys(), hyperparam_combination))
                    for name, value in name_value_list:
                        self.config.update({name: value})
                    tf.reset_default_graph()
                    model = Experiment(**self.config)
                    scores = model.fit(df.values, y, x_test=df_test.values, y_test=y_test, outlier_class=self.outlier_class)
                    self.models[str(hyperparam_combination)].append(model)
                    df_run = pd.DataFrame(data=scores, columns=["AUC", "Kmeans", "GMM", "Epoch"])
                    for name, value in name_value_list:
                        df_run[name] = str(value)
                    df_run["Seed"] = seed

                    if df_runs is None:
                        df_runs = df_run
                    else:
                        df_runs = df_runs.append(df_run, ignore_index=True)
                    df_runs.to_csv("Run_{}vs{}_{}x{}.csv".format(self.normal_classes, self.outlier_class, seeds, epochs))
                    pbar.update()

        if plot_boxplots:
            hyper_names = hyperparams.keys()
            fig, ax = plt.subplots(ncols=3, figsize=(18, 5))
            for hyperparam_combination in hyperparam_combinations:
                print(list(zip(hyper_names, hyperparam_combination)))
                df_slice = df_runs
                for name, value in zip(hyper_names, hyperparam_combination):
                    df_slice = df_slice[df_slice[name] == str(value)]
                for i, col in enumerate(["AUC", "Kmeans", "GMM"]):
                    df_slice.boxplot(col, by='Epoch', figsize=(5, 5), ax=ax[i], rot=90)
                    ax[i].set_ylim((0, 1))
                    if "warm_up_epochs" in df_slice.columns:
                        ax[i].axvline(x=float(df_slice["warm_up_epochs"].values[0]) / (epochs / len(df_slice["Epoch"].unique())) + 1, color='r')
                plt.ylim((0, 1))
                plt.title(str(list(zip(hyper_names, hyperparam_combination))))
                #plt.show()
