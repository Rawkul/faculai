import numpy as np
import os
from scipy import ndimage
import tensorflow as tf
from scipy.stats import sigmaclip
import tensorflow.keras.backend as K
import pkg_resources

class DetectionModel:
    def __init__(self, model_path=None, threshold = 0.5):
        if model_path is None:
            model_path = pkg_resources.resource_filename('faculai', 'unet.h5')
        
        # Clear memory before loading model
        tf.keras.backend.clear_session()
        tf.compat.v1.reset_default_graph()
        
        self.model = tf.keras.models.load_model(model_path, compile = False)
        self.threshold = threshold

    def slice_to_5(self, img):
        n = img.shape[0]
        aux = np.split(img, 5, axis = 2)
        aux = np.stack(aux, axis = 0)
        aux = aux.reshape((n * 5, 400, 380, 1), order = "F")
        return aux

    def merge_slices(self, img_array):
        north = np.concatenate(img_array[0:5,:,:,0], axis = 1)
        south = np.concatenate(img_array[5:10,:,:,0], axis = 1)
        return np.stack((north, south), axis = 0)

    def sigma_clip_stats(self, data, sigma=3):
        """
        Compute the sigma-clipped mean of an array.

        Parameters:
            data (ndarray): The data array.
            sigma (float): The number of standard deviations to use for clipping.

        Returns:
            float: The sigma-clipped mean.
        """
        # Remove nans
        dat = data[~np.isnan(data)]
        # Use the sigmaclip function to get the lower and upper limits removed
        clipped = sigmaclip(dat, low=sigma, high=sigma)[0]
        # Compute the mean of the clipped data
        mean = np.mean(clipped)
        sd = np.std(clipped)

        return [mean, sd]

    def fill_nans(self, img, sigma = 3):
        mean, sd = self.sigma_clip_stats(img, sigma)
        mask = np.isnan(img)
        shape = img[mask].shape
        noise = np.random.normal(mean, sd, size = shape)
        imgc = img.copy()
        imgc[mask] = noise
        return imgc

    def process_facula_slice(self, img, sigma = 3):
        return self.normalize(self.fill_nans(img, sigma))

    def process_facula_image(self, img, sigma = 3):
        output_img = self.slice_to_5(img)
        for i in range(output_img.shape[0]):
            output_img[i, :, :, 0] = self.process_facula_slice(output_img[i, :, :, 0], sigma)
        return output_img

    def normalize(self, img):
        mean = np.mean(img)
        std = np.std(img)
        return (img - mean) / std

    def __call__(self, img):
        if img.shape != (2, 400, 1900):
            raise ValueError("Input array doesn't have shape (2, 400, 1900).")
        input_img = self.process_facula_image(img)
        mask = self.model(input_img)
        mask = tf.where(tf.greater(mask, self.threshold), 
                        tf.ones_like(mask), 
                        tf.zeros_like(mask))
        mask = self.merge_slices(mask)
        # Get individual faculae and number of faculae using scipy ndimage
        # Count diagonals also as elements, not just squared
        group_structure = ndimage.generate_binary_structure(2, 2)
        
        # Make groups separately in the north and south
        faculae_north, n_north = ndimage.label(mask[0,:,:] == 1, structure = group_structure)
        faculae_south, n_south = ndimage.label(mask[1,:,:] == 1, structure = group_structure)
        
        # Join the groups in a single array
        faculae_south[faculae_south > 0] += n_north
        faculae = np.stack((faculae_north, faculae_south))
        num_faculae = n_north + n_south
        
        return faculae, num_faculae
