from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
import os
from PIL import Image
import numpy as np


def save_data(X, y, path, format='png'):
	"""Saves images (X) and their labels (y) in the specified path."""
	os.makedirs(path, exist_ok=True)  # Create directory if not exists

	for i in range(len(X)):
		image = X[i].reshape(28, 28).astype(np.uint8)  # Reshape and convert to uint8
		label = str(y[i])  # Convert label to string

		# Create a filename with padded index and label
		filename = f'image_{i:05d}_label_{label}.{format.lower()}'
		filepath = os.path.join(path, filename)

		# Convert image array to PIL Image and save
		try:
			img = Image.fromarray(image, mode='L')
			img.save(filepath)
		except Exception as e:
			print(f"Error saving {filename}: {e}")
   


def setup_data():
	# Fetch the MNIST dataset
	mnist = fetch_openml('mnist_784', version=1)

	# Extract images and labels
	X = mnist.data.astype(np.uint8)
	y = mnist.target.astype(np.uint8)
 
	# print(type(X))

	# Split the dataset (60% training, 40% testing)
	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=42)

	# Save the training and testing data as PNG
	save_data(X_train.to_numpy(), y_train.to_numpy(), 'train_data', format='png')
	save_data(X_test.to_numpy(), y_test.to_numpy(), 'test_data', format='png')


# Generate and save the dataset
setup_data()