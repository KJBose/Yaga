from __future__ import absolute_import, division, print_function, unicode_literals

import numpy as np
from PIL import Image

import tensorflow as tf
AUTOTUNE = tf.data.experimental.AUTOTUNE

import IPython.display as display
import matplotlib.pyplot as plt
import os
import pathlib

def create_img_data_dir(img_src, img_name):
    data_dir = tf.keras.utils.get_file(origin=img_src, fname=img_name, untar=True)
    data_dir = pathlib.Path(data_dir)

def display_images(data_dir):
    roses = list(data_dir.glob('roses/*'))
    for image_path in roses[:3]:
        display.display(Image.open(str(image_path)))
    
def keras_image_preprocess(data_dir):
    BATCH_SIZE = 32
    IMG_HEIGHT = 224
    IMG_WIDTH = 224
    STEPS_PER_EPOCH = np.ceil(image_count/BATCH_SIZE)
    
    image_generator = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255)
    train_data_gen = image_generator.flow_from_directory(directory=str(data_dir),
                                                        batch_size=BATCH_SIZE,
                                                        shuffle=True,
                                                        target_size=(IMG_HEIGHT, IMG_WIDTH),
                                                        classes = list(CLASS_NAMES))

# Inspect a batch:   
def show_batch(image_batch, label_batch):
  plt.figure(figsize=(10,10))
  for n in range(25):
      ax = plt.subplot(5,5,n+1)
      plt.imshow(image_batch[n])
      plt.title(CLASS_NAMES[label_batch[n]==1][0].title())
      plt.axis('off')

#short pure-tensorflow function that converts a file paths to an (image_data, label) pair:
def get_label(file_path):
  # convert the path to a list of path components
  parts = tf.strings.split(file_path, os.path.sep)
  # The second to last is the class-directory
  return parts[-2] == CLASS_NAMES

def decode_img(img):
  # convert the compressed string to a 3D uint8 tensor
  img = tf.image.decode_jpeg(img, channels=3)
  # Use `convert_image_dtype` to convert to floats in the [0,1] range.
  img = tf.image.convert_image_dtype(img, tf.float32)
  # resize the image to the desired size.
  return tf.image.resize(img, [IMG_WIDTH, IMG_HEIGHT])

def process_path(file_path):
  label = get_label(file_path)
  # load the raw data from the file as a string
  img = tf.io.read_file(file_path)
  img = decode_img(img)
  return img, label

def prepare_for_training(ds, cache=True, shuffle_buffer_size=1000):
  # This is a small dataset, only load it once, and keep it in memory.
  # use `.cache(filename)` to cache preprocessing work for datasets that don't
  # fit in memory.
  if cache:
    if isinstance(cache, str):
      ds = ds.cache(cache)
    else:
      ds = ds.cache()

  ds = ds.shuffle(buffer_size=shuffle_buffer_size)
  # Repeat forever
  ds = ds.repeat()
  ds = ds.batch(BATCH_SIZE)
  # `prefetch` lets the dataset fetch batches in the background while the model
  # is training.
  ds = ds.prefetch(buffer_size=AUTOTUNE)

  return ds

#function to convert image to a matrix(input, image= orig.png)
def image_preprocess(image)
    #img = Image.open('orig.png').convert('RGBA')
    img = Image.open(image).convert('RGBA')
    arr = np.array(img)

    #record the original shape
    shape = arr.shape

    # make a 1-dimensional view of arr
    flat_arr = arr.ravel()

    # convert it to a matrix
    vector = np.matrix(flat_arr)

    # do something to the vector
    #vector[:,::10] = 128

    # reform a numpy array of the original shape
    #arr2 = np.asarray(vector).reshape(shape)

    # make a PIL image
    #img2 = Image.fromarray(arr2, 'RGBA')
    #img2.show()
    
    return vector

if __init___ == __main__:
    img_src = 'https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz'
    img_name = 'flower_photos'
    img_dir = create_img_data_dir(img_src, img_name)
    image_count = len(list(data_dir.glob('*/*.jpg')))
    print(image_count)
    CLASS_NAMES = np.array([item.name for item in data_dir.glob('*') if item.name != "LICENSE.txt"])
    print(CLASS_NAMES)
    
    image_batch, label_batch = next(train_data_gen)
    show_batch(image_batch, label_batch)
    
    '''Load using tf.data
    The above keras.preprocessing method is convienient, but has two downsides:

    It's slow. See the performance section below.
    It lacks fine-grained control.
    It is not well integrated with the rest of TensorFlow.
    To load the files as a tf.data.Dataset first create a dataset of the file paths:'''

    list_ds = tf.data.Dataset.list_files(str(data_dir/'*/*'))
    for f in list_ds.take(5):
        print(f.numpy())

        # Set `num_parallel_calls` so multiple images are loaded/processed in parallel.
    labeled_ds = list_ds.map(process_path, num_parallel_calls=AUTOTUNE)

    for image, label in labeled_ds.take(1):
       print("Image shape: ", image.numpy().shape)
       print("Label: ", label.numpy())
    
    '''Basic methods for training
    To train a model with this dataset you will want the data:

    To be well shuffled.
    To be batched.
    Batches to be available as soon as possible.
    These features can be easily added using the tf.data api.'''
    train_ds = prepare_for_training(labeled_ds)
    image_batch, label_batch = next(iter(train_ds))
    
    #A large part of the performance gain comes from the use of .cache.
    uncached_ds = prepare_for_training(labeled_ds, cache=False)
    timeit(uncached_ds)
    
    #If the dataset doesn't fit in memory use a cache file to maintain some of the advantages:
    filecache_ds = prepare_for_training(labeled_ds, cache="./flowers.tfcache")
    timeit(filecache_ds)
