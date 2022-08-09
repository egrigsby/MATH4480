import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow_datasets as tfds
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

###########################################################################

# PROJECT 0:

###########################################################################

# PROJECT 1:

def load_data_perceptron():
    np.random.seed(seed=1)
    x_train = np.r_[np.random.randn(100,2)*0.5 + 1, np.random.randn(100,2)*0.5 + -1]
    y_train = np.r_[-np.ones(100), np.ones(100)]

    return x_train, y_train

def plot_data_perceptron(x,y):
    plt.scatter(x[:,0], x[:,1], c = y)

def plot_decision_boundary_perceptron(w):
    t = np.linspace(-2,2,100)
    plt.plot(t, (-1/w[1])*w[0]*t)

###########################################################################

# PROJECT 2:

def load_data_sk(center_1=[-1,-1], center_2=[1,1], m=20, seed=None):
    if seed:
        np.random.seed(seed=seed)
    x_train = np.r_[np.random.randn(m,2)*0.5 + center_1, np.random.randn(m,2)*0.5 + center_2]
    y_train = np.r_[-np.ones(m), np.ones(m)]
    return x_train, y_train

def plot_data_sk(x,y):
    plt.figure(figsize=(14,8))
    plt.scatter(x[:,0], x[:,1], c=y)
    plt.axis('scaled')
    plt.xlim([np.min(x[:,0])-0.5, np.max(x[:,0])+0.5])
    plt.ylim([np.min(x[:,1])-0.5, np.max(x[:,1])+0.5])

def plot_hyperplane_sk(x, h, c=None, label=None):
    t = np.linspace(np.min(x[:,0])-0.5,np.max(x[:,0])+0.5,100)
    plt.plot(t, (-1/h[1])*(h[0]*t+h[2]), c=c, label=label)
    plt.legend()
    plt.axis('scaled')
    plt.xlim([np.min(x[:,0])-0.5, np.max(x[:,0])+0.5])
    plt.ylim([np.min(x[:,1])-0.5, np.max(x[:,1])+0.5])

def perceptron_sk(x_train,y_train):
    num_data = len(x_train)
    w, b = np.array([0,0]), 0
    while True:
        updated = False
        for i in range(num_data):
            if y_train[i]*(np.dot(w,x_train[i]) + b) <= 0:
                w = w + y_train[i]*x_train[i]
                b = b + y_train[i]
                updated = True
        if not updated:
            break
    return np.r_[w, b]

###########################################################################

# PROJECT 3:

def load_data_auto_mpg():
    dataset_path = tf.keras.utils.get_file("auto-mpg.data", "http://archive.ics.uci.edu/ml/machine-learning-databases/auto-mpg/auto-mpg.data")
    column_names = ['MPG','Cylinders','Displacement','Horsepower','Weight',
                'Acceleration', 'Model Year', 'Origin']
    dataset = pd.read_csv(dataset_path, names=column_names,
                        na_values = "?", comment='\t',
                        sep=" ", skipinitialspace=True)
    dataset = dataset.dropna()
    dataset['Origin'] = dataset['Origin'].map(lambda x: {1: 'USA', 2: 'Europe', 3: 'Japan'}.get(x))
    dataset = pd.get_dummies(dataset)

    train_dataset = dataset.sample(frac=0.8,random_state=0)
    test_dataset = dataset.drop(train_dataset.index)

    train_labels = train_dataset.pop('MPG')
    test_labels = test_dataset.pop('MPG')

    x_train = train_dataset.values
    x_test = test_dataset.values

    y_train = train_labels.values.reshape(-1,1)
    y_test = test_labels.values.reshape(-1,1)

    scaler = StandardScaler()
    scaler.fit(x_train)
    x_train = scaler.transform(x_train)
    x_test = scaler.transform(x_test)

    return train_dataset, x_train, y_train, x_test, y_test

###########################################################################

# PROJECT 4:

def load_data_breastcancer(n_features=2):
    DATA_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/wdbc.data"
    file_path = tf.keras.utils.get_file("wdbc_data", DATA_URL)
    df = pd.read_csv(file_path, header = None)
    y = df[1].values
    x = df.drop([0, 1], axis = 1).values
    y = np.array([int(s == 'M') for s in y])
    y = np.reshape(y, (-1,1))
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, stratify = y, random_state=2)
    # Could also try MinMaxScaler here
    scaler = StandardScaler()
    scaler.fit(x_train)
    x_train = scaler.transform(x_train)
    x_test = scaler.transform(x_test)
    return x_train[:,:n_features], x_test[:,:n_features], y_train, y_test

def plot_data_breastcancer_2_dim(x,y):
    plt.scatter(x[:,0], x[:,1], c=y.squeeze())

def plot_decision_boundary_breastcancer_2_dim(model):
    w, b = model.trainable_variables

    w = w.numpy()
    b = b.numpy()

    t = np.linspace(-0.5,0.6,200)
    plt.plot(t, -1/w[1]*(w[0]*t + b))

def load_data_fashion_mnist():
    (x_train, y_train), (x_val, y_val) = tf.keras.datasets.fashion_mnist.load_data()
    x_train = x_train/255.
    x_val = x_val/255.
    return x_train.reshape((-1, 784)), y_train, x_val.reshape((-1, 784)), y_val

def plot_data_fashion_mnist(images, labels):
    CLASS_NAMES = [ 'T-shirt/top', 'Trouser',  'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag',  'Ankle boot']
    images = images * 255.
    n = len(images)
    plt.figure(figsize=(10*n,15))
    for i in range(n):
        plt.subplot(n,1,i+1)
        plt.imshow(images[i].reshape((28,28)),  cmap=plt.cm.gray)
        plt.title(CLASS_NAMES[labels[i]])

###########################################################################

# Uneeded as of now:

def plot_regression_line(w, b):
    t = np.linspace(-0.1, 1.1, 2000)
    plt.plot(t, w*t + b, 'b-')

def load_regression_data():
    np.random.seed(seed = 1)
    x_train = np.random.random([100])
    y_train = 5*x_train - 3 + np.random.randn(100)

    x_test = np.random.random([100])
    y_test = 5*x_test - 3 + np.random.randn(100)

    x_train = x_train.reshape(-1,1)
    x_test = x_test.reshape(-1,1)

    y_train = y_train.reshape(-1,1)
    y_test = y_test.reshape(-1,1)

    return x_train, y_train, x_test, y_test

def plot_regression_data(x, y):
    plt.plot(x.squeeze(),y.squeeze(), '.')
    plt.xlabel('x')
    plt.ylabel('y')

def plot_regression_lines(weight_bias_memory, x, y):
    for i, (weight, bias) in enumerate(weight_bias_memory):
        if i % 10 == 0 or i == len(weight_bias_memory) - 1:
            plot_regression_line(weight, bias)
    plt.plot(x[:, -1], y, 'o')

def mse_from_weights(w, b, x, y):
    y_hat = np.dot(x, w) + b
    return np.mean((y-y_hat)**2)

def plot_gradient_descent_progression(weight_bias_memory, x_train, y_train):
    delta = 0.1
    w = np.arange(-2.0, 10.0, delta)
    b = np.arange(-8.0, 2.0, delta)
    W, B = np.meshgrid(w,b)
    mse = np.zeros((len(b),len(w)))

    for i in range(len(w)):
        for j in range(len(b)):
            mse[j,i] = mse_from_weights(w[i],b[j], x_train, y_train)

    weight_bias_memory = np.array(weight_bias_memory)
    fig, ax = plt.subplots()
    CS = ax.contour(W, B, mse, levels = 200)
    ax.plot(weight_bias_memory[:,0], weight_bias_memory[:,1], 'ro')
    plt.xlabel('w');
    plt.ylabel('b');

def load_auto_mpg_data():
    dataset_path = tf.keras.utils.get_file("auto-mpg.data", "http://archive.ics.uci.edu/ml/machine-learning-databases/auto-mpg/auto-mpg.data")
    column_names = ['MPG','Cylinders','Displacement','Horsepower','Weight',
                'Acceleration', 'Model Year', 'Origin']
    dataset = pd.read_csv(dataset_path, names=column_names,
                        na_values = "?", comment='\t',
                        sep=" ", skipinitialspace=True)
    dataset = dataset.dropna()
    dataset['Origin'] = dataset['Origin'].map(lambda x: {1: 'USA', 2: 'Europe', 3: 'Japan'}.get(x))
    dataset = pd.get_dummies(dataset)

    train_dataset = dataset.sample(frac=0.8,random_state=0)
    test_dataset = dataset.drop(train_dataset.index)

    train_labels = train_dataset.pop('MPG')
    test_labels = test_dataset.pop('MPG')

    x_train = train_dataset.values
    x_test = test_dataset.values

    y_train = train_labels.values.reshape(-1,1)
    y_test = test_labels.values.reshape(-1,1)

    scaler = StandardScaler()
    scaler.fit(x_train)
    x_train = scaler.transform(x_train)
    x_test = scaler.transform(x_test)

    return train_dataset, x_train, y_train, x_test, y_test

###########################################################################