import matplotlib.pyplot as plt
import umap
from matplotlib.colors import Normalize
import numpy as np

def plot_loss_curves(loss_dict):
    # Get the loss values of the results dictionary
    train_loss = loss_dict['train_loss']
    test_loss = loss_dict['test_loss']

    # Get the accuracy values of the results dictionary (training and test)
    train_accuracy = loss_dict['train_acc']
    test_accuracy = loss_dict['test_acc']

    # Figure out how many epochs there were
    epochs = range(len(loss_dict['train_loss']))

    plt.figure(figsize=(12,5))

    # Plot loss
    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_loss, label='train_loss')
    plt.plot(epochs, test_loss, label='test_loss')
    plt.title('Loss')
    plt.xlabel('Epochs')
    plt.legend()

    # Plot accuracy
    plt.subplot(1, 2, 2)
    plt.plot(epochs, train_accuracy, label='train_accuracy')
    plt.plot(epochs, test_accuracy, label='test_accuracy')
    plt.title('Accuracy')
    plt.xlabel('Epochs')
    plt.legend();

def plot_coop(epochs, datasets, model_results, zs_results):
    colors = ['red', 'green', 'cyan', 'blue', 'black']
    plt.figure(figsize=(5,4))
    for data, color in zip(datasets, colors):
        plt.plot(epochs, model_results[data]['test_acc'], 'r', label=data)
        plt.scatter(x=0, y=zs_results[data], marker='x', c=color, label='CLIP zero-shot')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.title('Context Optimization [context tokens: 16]')
    plt.legend(loc='lower right')
    plt.show()

def normalize_array(vec: np.ndarray):
    return vec / np.linalg.norm(vec, axis=1, keepdims=True)

def draw_umap(data, target, classnames, n_neighbors=15, min_dist=0.1, n_components=2, metric='euclidean', title=''):
    fit = umap.UMAP(n_neighbors=n_neighbors,min_dist=min_dist,n_components=n_components,metric=metric)
    u = fit.fit_transform(data)

    fig = plt.figure(figsize=(8,8))
    if n_components == 1:
        ax = fig.add_subplot(111)
        ax.scatter(u[:,0], range(len(u)), c=target, cmap='Spectral')
    if n_components == 2:
        ax = fig.add_subplot(111)
        ax.scatter(u[:,0], u[:,1], c=target, cmap='Spectral')
    if n_components == 3:
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(u[:,0], u[:,1], u[:,2], c=target, cmap='Spectral', s=20)

    # Create a ScalarMappable object to associate with the colorbar
    norm = Normalize(vmin=target.min(), vmax=target.max())
    sm = plt.cm.ScalarMappable(cmap='Spectral',norm=norm)
    sm.set_array([])

    # Add colorbar
    cbar = plt.colorbar(sm, boundaries=np.arange(len(classnames)+1)-0.5, label='Class Labels')
    cbar.set_ticks(np.arange(len(classnames)))
    cbar.set_ticklabels(classnames)
    plt.title(title, fontsize=18)

    # Show the plot
    plt.show()

def draw_coop_umap(image_features, text_features, text_features_learned, target, classes, mapper, title):
    mapper = umap.UMAP(n_components=3, n_neighbors=9)
    image_feat = mapper.fit_transform(image_features)
    #image_feat = normalize_array(image_feat)
    text_feat = mapper.fit_transform(text_features)
    #text_feat = normalize_array(text_feat)
    text_feat_learned = mapper.fit_transform(text_features_learned)
    #text_feat_learned = normalize_array(text_feat_learned)
    fig = plt.figure(figsize=(8,8))

    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(image_feat[:,0], image_feat[:,1], image_feat[:,2], c=target, cmap='Spectral', s=20)
    ax.scatter(text_feat[:,0], text_feat[:,1], text_feat[:,2], c=list(mapper.values()), marker='x', cmap='Spectral', s=100)
    ax.scatter(text_feat_learned[:,0], text_feat_learned[:,1], text_feat_learned[:,2], c=list(mapper.values()), marker='^', cmap='Spectral', s=100)

    # Create a ScalarMappable object to associate with the colorbar
    norm = Normalize(vmin=target.min(), vmax=target.max())
    sm = plt.cm.ScalarMappable(cmap='Spectral',norm=norm)
    sm.set_array([])

    # Add colorbar
    cbar = plt.colorbar(sm, boundaries=np.arange(len(classes)+1)-0.5, label='Class Labels')
    cbar.set_ticks(np.arange(len(classes)))
    cbar.set_ticklabels(classes)
    plt.title(title, fontsize=18)

    # Show the plot
    plt.show()