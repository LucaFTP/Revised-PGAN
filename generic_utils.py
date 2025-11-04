import re
import os
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

def dynamic_range_opt(x, epsilon=1e-6, mult_factor=1.0):
    x = (x + epsilon) / epsilon
    a = tf.math.log(x) / tf.math.log(tf.constant(10.0, dtype=tf.float32))
    b = tf.math.log(1.0/epsilon) / tf.math.log(tf.constant(10.0, dtype=tf.float32))
    return (a / b) * mult_factor

def inv_dynamic_range(image, eps=1e-6, mult_factor=1.0):
    x = image / mult_factor
    a = - tf.pow(eps, 1.0 - x)
    b = tf.pow(eps, x) - 1.0
    return a * b

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

def plot_loss(loss_path, start_size=4):

    image_size = start_size
    num_files = (len(os.listdir(loss_path)) + 1)
    fig, ax = plt.subplots(num_files//2,2, figsize=(15,25))
    ax = ax.flatten()
    i = 0
    s={}
    color = ['b','g','r','y']
    for file in os.listdir(loss_path):
        name = re.split('_|\.',file)[-2]
        iteration = re.split('_|\.',file)[1]
        if name in ['init', 'stabilize']:
            s[name + iteration] = np.load(loss_path+'/'+file,allow_pickle=True)
    s = sorted(s.items())
    for j in range(len(s)):
        ax[i].plot(s[j][1].item()['d_loss'], '.-')
        ax[i].plot(s[j][1].item()['g_loss'], '.-')

        ax[i+1].plot(s[j][1].item()['r_loss'], '.-')

        try:
            image_size = 2*image_size
            ax[i].set_title(f"Image Size: {image_size} x {image_size}")
            ax[i+1].set_title(f"Image Size: {image_size} x {image_size}")
        except:
            ax[i].set_title(f"Image Size: {start_size} x {start_size}")
            ax[i+1].set_title(f"Image Size: {start_size} x {start_size}")
        ax[i].legend(['Discriminator Loss', 'Generator Loss'])
        ax[i+1].legend(['Generated Mass Loss', 'Real Mass Loss'])

        i = i + 2
        
    output_path = loss_path + "image.png"
    plt.savefig(output_path, bbox_inches='tight')