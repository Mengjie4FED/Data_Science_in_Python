import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from matplotlib import cm

# Load the Iris data set
# Split into test/train data with frac kept for testing
# Scale by training data
# Input parameter 'frac' specifies the amount to hold out for 'blind' testing
def get_iris_data(frac=0.4, show_plot=False):
    
    # Number of training samples to show in plot
    num_show = 10

    # Load the Iris Data
    iris = sns.load_dataset("iris")
    
    # Now lets get the data and labels
    data = iris[['sepal_length', 'sepal_width', 'petal_length', 'petal_width']].values
    labels = np.array([i//50 for i in range(iris.shape[0])])
    
    # We want to split our data into training and testing
    # Note that we have both 'data' and 'labels'
    d_train, d_test, l_train, l_test \
    = train_test_split(data, labels, test_size=frac, random_state=23)
    

    # Now scale our data
    # Create and fit scaler
    sc = StandardScaler().fit(d_train)

    d_train_sc = sc.transform(d_train)
    d_test_sc = sc.transform(d_test)
    
    if show_plot:

        # Now we create our figure and axes for the plot we will make.
        fig, ax = plt.subplots(figsize=(10, 10))
        
        x = d_train_sc[:, 1]
        y = d_train_sc[:, 3]
        
        for idx in np.unique(l_train):
            i = int(idx)
            ax.scatter(x[l_train == i], y[l_train == i], label=f'Class {i}',
                       s=200, alpha = .5, cmap=cm.coolwarm) 

        xx = d_test_sc[:num_show, 1]
        yy = d_test_sc[:num_show, 3]
        ax.scatter(xx, yy, label='Test Data',
                   marker='x', s=400, alpha = .5, cmap=cm.coolwarm) 
        
        # Decorate and clean plot
        ax.set_xlabel('Sepal Width', fontsize=16)
        ax.set_ylabel('Petal Width', fontsize=16)
        ax.legend(loc = 7, labelspacing=2)
        ax.set_title("Iris Classification Demonstration", fontsize=18)
        sns.despine(offset=0, trim=True)
    
    return d_train_sc, d_test_sc, l_train, l_test

# Make a uniformly spaced grid of points across the space occupied by our data
# We assume a datsets is passed into this function that has two dimensions 
# coresponding to x, y

def get_mdata(data, grid_size = 100):
    
    # We grab the min and max of the points, and make the pace a bit bigger.
    # We could make this dynamic.
    
    x_min, x_max = data[:, 0].min() - .25, data[:, 0].max() + .25
    y_min, y_max = data[:, 1].min() - .25, data[:, 1].max() + .25

    # Meshgrid gives two 2D arrays of the ppoints
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, grid_size),
                         np.linspace(y_min, y_max, grid_size))
    
    # We want to return these points as an array of two-dimensional points
    return np.c_[xx.ravel(), yy.ravel()]

# Trim the Full Iris data set to two dimensions 
# (Sepal Width and Petal Width)
# return the data with attached labels
def trim_data(d_train_sc, l_train):
    
    data = np.zeros((d_train_sc.shape[0], 3))
    data[:, 0] = d_train_sc[:, 1]
    data[:, 1] = d_train_sc[:, 3]
    data[:, 2] = l_train[:]
    
    return data

# Create Colormaps for the plots
from matplotlib.colors import ListedColormap

# Plot the data and mesh for comparison
def splot_data(ax, data, mdata, z, label1, label2, sz, grid_size = 100):

    cmap_back = ListedColormap(sns.hls_palette(3, l=.4, s=.1))
    cmap_pts = ListedColormap(sns.hls_palette(3, l=.9, s=.9))

    ax.set_aspect('equal')

    # Decorate the plot
    ax.set_xlabel(label1)
    ax.set_ylabel(label2)
    
    # We need grid points and values to make the colormesh plot
    xx = mdata[:, 0].reshape((grid_size, grid_size))
    yy = mdata[:, 1].reshape((grid_size, grid_size))
    zz = z.reshape((grid_size, grid_size))

    ax.pcolormesh(xx, yy, zz, cmap=cmap_back, alpha = 0.9)
    
    # Now draw the points, with bolder colors.
    ax.scatter(data[:, 0], data[:, 1], c=data[:, 2], s=sz, cmap=cmap_pts)