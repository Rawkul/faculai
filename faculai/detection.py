import numpy as np
from scipy import ndimage
import tensorflow as tf
from scipy.stats import sigmaclip
import pkg_resources

import matplotlib.pyplot as plt

def plot_images(images, num_x, num_y):
    fig, axs = plt.subplots(num_x, num_y, figsize=(10,10))  # Adjust the figure size as necessary

    for ax in axs.flat:
        ax.axis('off')

    for i, img in enumerate(images):
        axs[i // num_y, i % num_y].imshow(img, cmap='gray')
        axs[i // num_y, i % num_y].axis('off')

    plt.tight_layout()
    plt.show()


class DetectionModel:
    def __init__(self, model_path=None, threshold = 0.5, batch_size = 10):
        """
        Initialize the detection model.
        
        Parameters:
            model_path (str): The path to the model file. Default is None.
            threshold (float): The threshold for binary prediction. Default is 0.5.
            batch_size (int): The batch size for prediction. Default is 10. 
                              Lower this value if your card runs out of memory 
                              while detecting faculae.
        """
        if model_path is None:
            model_path = pkg_resources.resource_filename('faculai', 'unet.h5')
        
        # Clear memory before loading model
        tf.keras.backend.clear_session()
        tf.compat.v1.reset_default_graph()
        
        self.model = tf.keras.models.load_model(model_path, compile = False)
        self.threshold = threshold
        self.input_shape = self.model.input_shape[1:3]
        self.batch_size = batch_size

    def split_image(self, image):
        """
        Split an image into smaller sub-images.
        
        Parameters:
            image (ndarray): The image to split.
            
        Returns:
            list: A list of sub-images.
        """
        M, N = image.shape
        m, n = self.input_shape
        
        # Check if the size of the image is less than nxn
        if M < m or N < n:
            raise ValueError(f"Input image size ({M}x{N}) must be at least {m}x{n}.")
        
        num_x = M // m if M % m == 0 else M // m + 1
        num_y = N // n if N % n == 0 else N // n + 1
        
        result = []
        
        for i in range(num_x):
            for j in range(num_y):
                sub_image = np.full((m, n), np.nan)  # Initialize sub_image as nxn array filled with nan
                
                # Determine the slice sizes for the current sub_image
                slice_x = slice(i * m, min((i+1) * m, M))
                slice_y = slice(j * n, min((j+1) * n, N))
                
                sub_image[:slice_x.stop - slice_x.start, :slice_y.stop - slice_y.start] = image[slice_x, slice_y]
                
                result.append(sub_image)
        
        return result
    
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
        """
        Fill NaN values in an image.
        
        Parameters:
            img (ndarray): The image to process.
            sigma (float): The number of standard deviations to use for clipping.
            
        Returns:
            ndarray: The processed image.
        """
        mean, sd = self.sigma_clip_stats(img, sigma)
        mask = np.isnan(img)
        shape = img[mask].shape
        noise = np.random.normal(mean, sd, size = shape)
        imgc = img.copy()
        imgc[mask] = noise
        return imgc
    
    def normalize(self, img):
        """
        Normalize an image.
        
        Parameters:
            img (ndarray): The image to normalize.
            
        Returns:
            ndarray: The normalized image.
        """
        mean = np.mean(img)
        std = np.std(img)
        return (img - mean) / (std + 1e-20) # Avoid cases when std = 0 (all values in the image are the same)
    
    def process_pole_image(self, img, sigma = 3):
        """
        Process a pole image.
        
        Parameters:
            img (ndarray): The image to process.
            sigma (float): The number of standard deviations to use for clipping.
            
        Returns:
            ndarray: The processed image.
        """
        sliced_imgs = self.split_image(img)
        for i in range(len(sliced_imgs)):
            sliced_imgs[i] = self.fill_nans(sliced_imgs[i], sigma)
            sliced_imgs[i] = self.normalize(sliced_imgs[i])
        return np.stack(sliced_imgs, axis = 0)

    def process_facula_image(self, img, sigma = 3):
        """
        Process a facula image.
        
        Parameters:
            img (ndarray): The image to process.
            sigma (float): The number of standard deviations to use for clipping.
            
        Returns:
            ndarray: The processed image.
        """
        north = self.process_pole_image(img[0,:,:])
        south = self.process_pole_image(img[1,:,:])
        return np.concatenate((north, south), axis = 0)

    def reconstruct_pole_image(self, sub_images, M, N):
        """
        Reconstruct a pole image from sub-images.
        
        Parameters:
            sub_images (list): A list of sub-images.
            M (int): The number of rows in the original image.
            N (int): The number of columns in the original image.
            
        Returns:
            ndarray: The reconstructed image.
        """
        image = np.empty((M, N))
        m, n = self.input_shape
        
        num_x = M // m if M % m == 0 else M // m + 1
        num_y = N // n if N % n == 0 else N // n + 1

        for i in range(num_x):
            for j in range(num_y):
                # Determine the slice sizes for the current sub_image
                slice_x = slice(i * m, min((i+1) * m, M))
                slice_y = slice(j * n, min((j+1) * n, N))

                sub_image = sub_images[i * num_y + j]
                image[slice_x, slice_y] = sub_image[:slice_x.stop - slice_x.start, :slice_y.stop - slice_y.start]

        return image
    
    def separate_arrays(self, original_array):
        """
        Separate a 3D array into a list of 2D arrays.
        
        Parameters:
            original_array (ndarray): The array to separate.
            
        Returns:
            list: A list of 2D arrays.
        """
        m = original_array.shape[0]
        separated_arrays = [original_array[i,:,:] for i in range(m)]
        return separated_arrays

    def reconstruct_masks(self, original_image, output_masks):
        """
        Reconstruct masks from sub-images.
        
        Parameters:
            original_image (ndarray): The original image.
            output_masks (list): A list of sub-image masks.
            
        Returns:
            ndarray: The reconstructed mask.
        """
        l = output_masks.shape[0]
        north = self.separate_arrays(output_masks[:l//2,:,:])
        south = self.separate_arrays(output_masks[l//2:,:,:])
        
        M, N = original_image.shape[1:]
        north = self.reconstruct_pole_image(north, M, N)
        south = self.reconstruct_pole_image(south, M, N)
        
        reconstructed = np.stack((north, south), axis = 0)
        
        # Apply nan mask to avoid and mittigate possible failures in nan regions
        reconstructed[np.isnan(original_image)] = 0
        
        return reconstructed
    
    def __call__(self, img):
        """
        Call the detection model to process an image and predict masks.
        
        Parameters:
            img (ndarray): The image to process and predict.
            
        Returns:
            tuple: A tuple containing the detected faculae masks and the total number of faculae.
        """
        # Process image and predict it
        input_img = self.process_facula_image(img)
        mask = self.model.predict(input_img, batch_size = self.batch_size)
        mask = np.squeeze(mask)
        mask = np.where(np.greater(mask, self.threshold), 
                        np.ones_like(mask), 
                        np.zeros_like(mask))
        mask = self.reconstruct_masks(img, mask)
        
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
