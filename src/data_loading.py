import glob
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf
import scipy
from scipy.io import loadmat
from sklearn.datasets import make_blobs

from config import DATA_PATH


def load_gaussian_blobs():
    data, _ = make_blobs(n_samples=1000, n_features=2, centers=2, random_state=123)
    df = pd.DataFrame(data=data)
    df['y'] = 0
    df.iloc[300, :] = [-4, -5, 1]
    df.iloc[500, :] = [0, 2, 1]
    y = df["y"]
    df.drop("y", axis=1, inplace=True)
    df -= df.mean()
    df /= df.std()
    df["y"] = y
    return df


def load_mnist():
    mnist = tf.keras.datasets.mnist
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train = x_train / 255
    x_test = x_test / 255
    df = pd.DataFrame(data=x_train.reshape(-1, 784))
    df['y'] = y_train
    df_test = pd.DataFrame(data=x_test.reshape(-1, 784))
    df_test['y'] = y_test
    return df, df_test


def load_fashion_mnist():
    mnist = tf.keras.datasets.fashion_mnist
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train = x_train / 255
    x_test = x_test / 255
    df = pd.DataFrame(data=x_train.reshape(-1, 784))
    df['y'] = y_train
    df_test = pd.DataFrame(data=x_test.reshape(-1, 784))
    df_test['y'] = y_test
    return df, df_test


def load_gas_data(outlier_class=3):
    df = pd.DataFrame()
    for file in glob.glob(str(Path(DATA_PATH / "Gas")) + "/batch*.dat"):
        df = df.append(pd.read_csv(file, sep=' ', header=None))
    df.columns = ["y"] + list(df.columns)[1:]
    for c in df.columns[1:]:
        df[c] = df[c].map(lambda x: x.split(':')[1])
    y = df["y"]
    del df["y"]
    df = df.astype("float")
    df = (df - df.mean()) / (df.max() - df.min())
    df["y"] = y == outlier_class
    return df


def load_forest_cover_data():
    columns = ["Elevation", "Aspect", "Slope", "Horizontal_Distance_To_Hydrology", "Vertical_Distance_To_Hydrology", "Horizontal_Distance_To_Roadways",
               "Hillshade_9am", "Hillshade_Noon", "Hillshade_3pm", "Horizontal_Distance_To_Fire_Points"]
    columns.extend(["Wilderness_Area_" + str(i) for i in range(4)])
    columns.extend(["Soil_Type" + str(i) for i in range(40)])
    columns.append("y")
    df = pd.read_csv(str(Path(DATA_PATH / "ForestCover" / "covtype.data")), sep=',', header=None, names=columns, dtype=float)
    df = df[df["y"].isin([4, 2])]
    y = df["y"]
    del df["y"]
    df = df - df.min()
    df = df / df.max()
    df.fillna(0, inplace=True)
    df["y"] = y

    df_test = df[y == 4]
    df = df[y == 2]
    df = df.append(df_test, ignore_index=True)

    n = len(df)
    return df.loc[:int(0.8*n)], df.loc[int(0.8*n):]


def load_sat_image_data():
    file = glob.glob(str(DATA_PATH / "**" / "satimage-2.mat"))[0]
    mat = loadmat(file)
    df = pd.DataFrame(mat['X'])
    y = mat['y']
    df = df - df.min()
    df = df / df.max()
    df["y"] = y
    n = len(df)
    df_test = df.loc[int(0.8*n):]
    df = df.loc[:int(0.8*n)]
    return df, df_test


def load_kdd_cup(seed=0):
    """
    This approach is used by the DAGMM paper (Zong et al., 2018) and was first described in Zhai et al.,
    Deep structured energy based models for anomaly detection:
    "As 20% of data samples are labeled as “normal” and the rest are labeled as “attack”, “normal” samples are in a
    minority group; therefore, “normal” ones are treated as anomalies in this task" - Zong et al., 2018
    "[...]in each run, we take 50% of data by random sampling for training with the rest 50% reserved for testing,
    and only data samples from the normal class are used for training models.[...] - Zong et al., 2018"
    :return: (X_train, y_train), (X_test, y_test)
    """
    np.random.seed(seed)
    data = np.load(str(Path(DATA_PATH / "kdd_cup.npz")))

    labels = data['kdd'][:, -1]
    features = data['kdd'][:, :-1]

    normal_data = features[labels == 1]
    normal_labels = labels[labels == 1]

    attack_data = features[labels == 0]
    attack_labels = labels[labels == 0]

    n_attack = attack_data.shape[0]

    rand_idx = np.arange(n_attack)
    np.random.shuffle(rand_idx)
    n_train = n_attack // 2

    train = attack_data[rand_idx[:n_train]]
    train_labels = attack_labels[rand_idx[:n_train]]

    test = attack_data[rand_idx[n_train:]]
    test_labels = attack_labels[rand_idx[n_train:]]

    test = np.concatenate((test, normal_data), axis=0)
    test_labels = np.concatenate((test_labels, normal_labels), axis=0)

    return (train, train_labels), (test, test_labels)


def save_stl10_images():
    data = loadmat(str(Path(DATA_PATH / "STL10" / "train.mat")))
    X = data['X']
    X = X.reshape(-1, 3, 96, 96)
    X = np.transpose(X, (0, 3, 2, 1))
    for index in range(len(X)):
        scipy.misc.imsave(str(Path(DATA_PATH / "STL10" / "{}.jpg".format(index))), X[index, :, :, :])