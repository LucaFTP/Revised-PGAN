import numpy as np
import tensorflow as tf
from typing import Callable
from scipy.linalg import sqrtm
from argparse import ArgumentParser
from matplotlib import pyplot as plt
from skimage.transform import resize
from tensorflow.keras.applications.inception_v3 import preprocess_input # type: ignore

def compute_depth_from_res(res, min_res=4):
    """Compute number of progressive growing blocks based on resolution"""
    return int(np.log2(res) - np.log2(min_res))

def plot_images(generated_imgs, save_path):
    imgs_to_plot = 9
    n_grid = int(np.sqrt(imgs_to_plot))
    plt.figure(figsize=(10, 10))
    for i in range(imgs_to_plot):
        plt.subplot(n_grid, n_grid, i + 1)
        plt.imshow(generated_imgs[i, :, :, 0], cmap='inferno')
        plt.axis('off')
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"[GANMonitor] Saved image grid to {save_path}")

def scale_images(images, new_shape):
    images_list = []
    for image in images:
        new_image = resize(image, new_shape, 0)
        images_list.append(new_image)
    
    return np.asarray(images_list)

def dynamic_range_opt(array, epsilon=1e-6, mult_factor=1):
    array = (array + epsilon)/epsilon
    a = np.log10(array)
    b = np.log10(1/epsilon)
    return a/b * mult_factor

def inv_dynamic_range(synth_img, eps=1e-6, mult_factor=1):
    # synth_img = tf.image.resize(synth_img, (128, 128))
    image = synth_img[:]/mult_factor
    a = - eps**(1 - image)
    b = eps**image - 1
    return a*b

# calculate frechet inception distance
def calculate_fid(fid_model, mu1, sigma1, images2):
    images2 = preprocess_input(images2)
    # calculate activations
    act2 = fid_model.predict(images2)
    # calculate mean and covariance statistics
    mu2, sigma2 = act2.mean(axis=0), np.cov(act2, rowvar=False)
    # calculate sum squared difference between means
    ssdiff = np.sum((mu1 - mu2)**2.0)
    # calculate sqrt of product between cov
    covmean = sqrtm(sigma1.dot(sigma2))
    # check and correct imaginary numbers from sqrt
    if np.iscomplexobj(covmean):
         covmean = covmean.real
    # calculate score
    fid = ssdiff + np.trace(sigma1 + sigma2 - 2.0 * covmean)
 
    return fid

def prepare_real_images(fid_model, meta_data, target_size:int):
    real_images = []
    data_dir = "/leonardo_scratch/fast/uTS25_Fontana/ALL_ROT_npy_version/1024x1024/"

    meta_data = meta_data[meta_data['rot'] == 0]
    ## Apply normalization and reshape into target_size x target_size images
    for idx, data_point in meta_data.iterrows():
        image_arr = np.load(data_dir + str(data_point['id']) + '.npy').astype('float32')
        image_arr = tf.image.resize(np.expand_dims(image_arr, axis=-1), (target_size, target_size)).numpy()
        image_arr = dynamic_range_opt(image_arr, mult_factor=2.5)
        
        real_images.append(image_arr)

    ## Prepare the sampled images for the application of the InceptionV3
    real_set = np.repeat(real_images, 3, axis=3)
    real_set = scale_images(real_set, (299, 299, 3))
    act1 = fid_model.predict(preprocess_input(real_set))

    return act1.mean(axis=0), np.cov(act1, rowvar=False)

def prepare_fake_images(synthetic_set):
    ## Selecting num_samples images according to the shape of the data
    synth_imgs = np.repeat(synthetic_set, 3, axis=3)
    synth_imgs = scale_images(synth_imgs, (299, 299, 3))

    return synth_imgs

def parser(
    prog_name: str, dscr: str, get_args: Callable[[ArgumentParser], ArgumentParser]
) -> Callable[[Callable], Callable]:
    def decorator(function):
        def new_function(*args, **kwargs):
            prs = ArgumentParser(
                prog=prog_name,
                description=dscr,
            )

            prs = get_args(prs)
            args = prs.parse_args()
            function(args)

        return new_function

    return decorator