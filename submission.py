import cv2
import numpy as np

def transform(array_of_images):
	X_train = []
	train_images = []
	for image in array_of_images:
		image = cv2.resize(image, (128,128))
		train_images.append(image)

	X_train = np.array(train_images, np.float32) / 255.
	mean_img = X_train.mean(axis=0)
	std_dev = X_train.std(axis=0)
	X_norm = (X_train - mean_img)/ std_dev

	return X_norm
