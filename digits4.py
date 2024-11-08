import pandas as pd
import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from matplotlib.offsetbox import OffsetImage, AnnotationBbox

# Load and filter the dataset to include only digits 0-4
train = pd.read_csv('MNIST.csv')
train = train[train['label'].isin([0, 1, 2, 3, 4])]

# Randomly sample a subset of the data to reduce memory usage
n_samples = 300  # Adjust this based on available memory
sampled_train = train.sample(n=n_samples, random_state=42)
x_subset = sampled_train.loc[:, 'pixel0':].values
y_subset = sampled_train['label'].values

# Run t-SNE on the subset data
tsne = TSNE(n_components=2, perplexity=30, learning_rate=200, random_state=0)
tsne_result = tsne.fit_transform(x_subset)

# Function to convert each row of pixel data to an image
def get_image(arr, img_shape=(28, 28)):
    return arr.reshape(img_shape)

# Plot each image at the corresponding t-SNE position
plt.figure(figsize=(12, 12))
ax = plt.gca()

for i in range(len(tsne_result)):
    # Extract the image and resize to 28x28 (assuming MNIST dimensions)
    img = get_image(x_subset[i])
    
    # Create an OffsetImage and place it at the t-SNE result position
    imagebox = OffsetImage(img, zoom=0.6, cmap='gray')
    ab = AnnotationBbox(imagebox, (tsne_result[i, 0], tsne_result[i, 1]), frameon=False)
    ax.add_artist(ab)

# Set title and remove axis for a cleaner look
plt.title("t-SNE Visualization of Digits 0-4 with Images")
plt.xticks([])
plt.yticks([])
plt.show()
