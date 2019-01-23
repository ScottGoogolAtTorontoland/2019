import warnings

warnings.filterwarnings('ignore')
import sys

sys.path.append('..')
import time
from src.data_loading import load_mnist, load_fashion_mnist, load_sat_image_data, load_forest_cover_data
from src.experiment import Experiment
import tensorflow as tf
import pandas as pd
import numpy as np
from tqdm import tqdm_notebook as tqdm

SEEDS = 100
EPOCHS = 50


def main():
    datasets = [
        ["Sat Images", load_sat_image_data, [0], [1], [32, 16, 8, 4, 2]],
        ["ForestCover", load_forest_cover_data, [2], [4], [32, 16, 2]],
        ["MNIST", load_mnist, [1, 2, 3, 4, 5, 6, 7, 8], [9, 0], [512, 256, 128, 10]],
        ["FMNIST", load_fashion_mnist, [1, 2, 3, 4, 5, 6, 7, 8], [9, 0], [512, 256, 128, 10]],
    ]
    algorithms = ['AE', 'VAE', 'GMVAE']
    dataset_names = [x[0] for x in datasets]

    df_runs = None
    timestamp = time.strftime('%Y-%m-%d-%H%M%S')

    seeds = np.random.randint(np.iinfo(np.uint32).max, size=SEEDS, dtype=np.uint32)
    for seed in tqdm(seeds):
        for dataset in tqdm(datasets):
            name, load_data_func, inliers, outliers, comp_hidden = dataset

            df, df_test = load_data_func()
            y_train = df["y"]
            del df["y"]
            y_test = df_test["y"]
            del df_test["y"]

            x_train, x_test = df.values, df_test.values
            clusters = len(inliers)

            df = df[~y_train.isin(outliers)]

            for algorithm in algorithms:
                config = {'comp_hiddens': comp_hidden, 'epoch_size': EPOCHS, 'minibatch_size': 128, 'random_seed': seed,
                          'normalize': False, 'encoding': algorithm, 'n_clusters': clusters, 'name': name, 'img_dims': None}

                tf.reset_default_graph()
                model = Experiment(**config)
                scores = model.fit(df.values, y_train, x_test=x_test, y_test=y_test, outlier_classes=outliers)
                df_run = pd.DataFrame(scores, columns=['auc_reconstruction', 'auc_custom', 'accuracy_kmeans',
                                                       'accuracy_custom', 'ari_kmeans_score', 'ari_custom_score', 'epoch'])
                df_run["seed"] = seed
                df_run["algorithm"] = algorithm
                df_run["dataset"] = name
                if df_runs is None:
                    df_runs = df_run
                else:
                    df_runs = df_runs.append(df_run, ignore_index=True)
                try:
                    df_runs.to_csv("{}_runs.csv".format(timestamp))
                except:
                    pass

    max_scores = df_runs.groupby(["dataset", "algorithm", "seed"]).max()
    for score_name in max_scores.keys():
        if score_name == 'epoch':
            continue
        print(score_name)
        df_result = pd.DataFrame(columns=algorithms, index=dataset_names)
        for dataset_name in dataset_names:
            for algorithm in algorithms:
                dataset_algorithm_result = max_scores.loc[dataset_name].loc[algorithm][score_name]
                df_result.loc[dataset_name, algorithm] = "{0:.1%}±{1:.1%}".format(dataset_algorithm_result.mean(), dataset_algorithm_result.std())
        print(df_result)


if __name__ == '__main__':
    main()
