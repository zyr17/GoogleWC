# -*- coding: utf-8 -*-
"""
Created on Tue Jan 15 14:59:08 2019

@author: 13236
"""

from __future__ import absolute_import, division, print_function

# Import TensorFlow >= 1.10 and enable eager execution
import tensorflow as tf
tf.enable_eager_execution()

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

'''
checkpoint_dir = './training_checkpoints'
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
checkpoint = tf.train.Checkpoint(generator_optimizer=generator_optimizer,
                                 discriminator_optimizer=discriminator_optimizer,
                                 generator=generator,
                                 discriminator=discriminator)
'''


class Generator(tf.keras.Model):
  def __init__(self):
    super(Generator, self).__init__()
    self.fc1 = tf.keras.layers.Dense(7*7*64, use_bias=False)
    self.batchnorm1 = tf.keras.layers.BatchNormalization()
    
    self.conv1 = tf.keras.layers.Conv2DTranspose(64, (5, 5), strides=(1, 1), padding='same', use_bias=False)
    self.batchnorm2 = tf.keras.layers.BatchNormalization()
    
    self.conv2 = tf.keras.layers.Conv2DTranspose(32, (5, 5), strides=(2, 2), padding='same', use_bias=False)
    self.batchnorm3 = tf.keras.layers.BatchNormalization()
    
    self.conv3 = tf.keras.layers.Conv2DTranspose(1, (5, 5), strides=(2, 2), padding='same', use_bias=False)

  def call(self, x, training=True):
    x = self.fc1(x)
    x = self.batchnorm1(x, training=training)
    x = tf.nn.relu(x)

    x = tf.reshape(x, shape=(-1, 7, 7, 64))

    x = self.conv1(x)
    x = self.batchnorm2(x, training=training)
    x = tf.nn.relu(x)

    x = self.conv2(x)
    x = self.batchnorm3(x, training=training)
    x = tf.nn.relu(x)

    x = tf.nn.tanh(self.conv3(x))  
    return x

label_dim = 4
class Generatorn(tf.keras.Model):
  def __init__(self):
    super(Generatorn, self).__init__()
    self.fc1 = tf.keras.layers.Dense(7*7*64, use_bias=False)
    self.fc_label = tf.keras.layers.Dense(7*7*label_dim, use_bias=False)
    self.batchnorm1 = tf.keras.layers.BatchNormalization()
    
    self.conv1 = tf.keras.layers.Conv2DTranspose(64, (5, 5), strides=(1, 1), padding='same', use_bias=False)
    self.convlabel = tf.keras.layers.Conv2DTranspose(label_dim, (5, 5), strides=(1, 1), padding='same', use_bias=False)
    self.batchnorm2 = tf.keras.layers.BatchNormalization()
    
    self.conv2 = tf.keras.layers.Conv2DTranspose(32, (5, 5), strides=(2, 2), padding='same', use_bias=False)
    self.batchnorm3 = tf.keras.layers.BatchNormalization()
    
    self.conv3 = tf.keras.layers.Conv2DTranspose(1, (5, 5), strides=(2, 2), padding='same', use_bias=False)

  def call(self, x, label, training=True):
    label = tf.cast(label, dtype=tf.float32)
    #print(x.shape, label.shape) 
    x = self.fc1(x) #(256, 100)

    label = self.fc_label(label)
    #print(x.shape)
    x = self.batchnorm1(x, training=training) 
    #label = self.batchnorm1(label, training=training) 
    #print(x.shape)
    x = tf.nn.relu(x)
    #print(x.shape)

    x = tf.reshape(x, shape=(-1, 7, 7, 64))
    label = tf.reshape(label, shape=(-1, 7, 7, label_dim))
    #print(x.shape)

    x = self.conv1(x)
    label = self.convlabel(label)
    #print(x.shape)
    x = self.batchnorm2(x, training=training)
    #print(x.shape)
    x = tf.nn.relu(x)
    #print(x.shape)
    x = tf.concat([x, label], 3) #[256,7,7,64*2]

    x = self.conv2(x)
    #label = self.conv2(label)
    #print(x.shape)
    x = self.batchnorm3(x, training=training)
    #print(x.shape)
    x = tf.nn.relu(x)
    #print(x.shape)

    x = tf.nn.tanh(self.conv3(x))  
    #print(x.shape)
    
    #raise(Exception('pass'))
    return x

class Generatorpart(tf.keras.Model):
  def __init__(self):
    super(Generatorpart, self).__init__()
    self.fc1 = tf.keras.layers.Dense(7*7*64, use_bias=False)
    self.batchnorm1 = tf.keras.layers.BatchNormalization()
    
    self.conv1 = tf.keras.layers.Conv2DTranspose(64, (5, 5), strides=(1, 1), padding='same', use_bias=False)
    self.batchnorm2 = tf.keras.layers.BatchNormalization()
    
    self.conv2 = tf.keras.layers.Conv2DTranspose(32, (5, 5), strides=(2, 2), padding='same', use_bias=False)
    self.batchnorm3 = tf.keras.layers.BatchNormalization()
    
    self.conv3 = tf.keras.layers.Conv2DTranspose(1, (5, 5), strides=(2, 2), padding='same', use_bias=False)

    self.partconv1 = tf.keras.layers.Conv2D(16, (5, 5), strides=(2, 2), padding='same')
    self.partbatchnorm1 = tf.keras.layers.BatchNormalization()
    
    self.partconv2 = tf.keras.layers.Conv2D(64, (5, 5), strides=(2, 2), padding='same')
    self.partbatchnorm2 = tf.keras.layers.BatchNormalization()

  def call(self, x, part, training=True):

    part = tf.nn.leaky_relu(self.partconv1(part))
    part = self.partbatchnorm1(part, training=training) 

    part = tf.nn.leaky_relu(self.partconv2(part))
    part = self.partbatchnorm2(part, training=training) 
    #print(x.shape, label.shape) 
    x = self.fc1(x) #(256, 100)

    #print(x.shape)
    x = self.batchnorm1(x, training=training) 
    #label = self.batchnorm1(label, training=training) 
    #print(x.shape)
    x = tf.nn.relu(x)
    #print(x.shape)

    x = tf.reshape(x, shape=(-1, 7, 7, 64))
    x = tf.concat([x, part], 3) #[256,7,7,64+64]
    #print(x.shape)

    x = self.conv1(x)
    #print(x.shape)
    x = self.batchnorm2(x, training=training)
    #print(x.shape)
    x = tf.nn.relu(x)
    #print(x.shape)

    x = self.conv2(x)
    #label = self.conv2(label)
    #print(x.shape)
    x = self.batchnorm3(x, training=training)
    #print(x.shape)
    x = tf.nn.relu(x)
    #print(x.shape)

    x = tf.nn.tanh(self.conv3(x))  
    #print(x.shape)
    
    #raise(Exception('pass'))
    return x

def generator_restore(generator, checkpoint_dir):
    
    checkpoint = tf.train.Checkpoint(generator=generator)
    
    checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir))
    
    return generator


def generate_and_save_images(model, test_input, imgsavepath):

    predictions = model(test_input, training=False)

    fig = plt.figure(figsize=(4,4))
  
    for i in range(predictions.shape[0]):
        plt.subplot(4, 4, i+1)
        plt.imshow(predictions[i, :, :, 0] * 127.5 + 127.5, cmap='gray')
        plt.axis('off')
        
    plt.savefig(imgsavepath)
   
def generate_and_save_imagesn(model, test_input, labels, imgsavepath):

    labels = np.array(labels)
    predictions = model(test_input, labels, training=False)

    fig = plt.figure(figsize=(4,4))
  
    for i in range(predictions.shape[0]):
        plt.subplot(4, 4, i+1)
        plt.imshow(predictions[i, :, :, 0] * 127.5 + 127.5, cmap='gray')
        plt.axis('off')
        
    plt.savefig(imgsavepath)

def generate_and_save_imagespart(model, test_input, img, imgsavepath):
    
    predictions = model(test_input, img, training=False)

    predictions = (predictions + 1) / 2 * 255
    predictions = np.array(predictions, dtype=np.uint8).reshape(28, 28)

    res = Image.fromarray(predictions)
    res.save(imgsavepath)

def generate1(imgsavepath = 'imgs/image_generate.png', checkpointpath = 'training_checkpoints'):
    
    noise_dim = 100
    num_examples_to_generate = 16
    random_vector_for_generation = tf.random_normal([num_examples_to_generate,
                                                     noise_dim])
    generator = Generator()
    generate_and_save_images(generator_restore(generator, checkpointpath),
                             random_vector_for_generation,
                             imgsavepath)

def generaten(imgsavepath = 'imgs/image_generate.png', checkpointpath = 'training_checkpoints'):
    
    noise_dim = 100
    num_examples_to_generate = 16
    labels = [0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3]
    labels = tf.one_hot(labels, 4, 1, 0)
    random_vector_for_generation = tf.random_normal([num_examples_to_generate,
                                                     noise_dim])
    generator = Generatorn()
    generate_and_save_imagesn(generator_restore(generator, checkpointpath),
                             random_vector_for_generation,
                             labels,
                             imgsavepath)

def generatepart(inputimg, imgsavepath = 'imgs/image_generate.png', checkpointpath = 'training_checkpoints'):
    
    inputimg = tf.convert_to_tensor(np.array(inputimg).reshape(1, 28, 28, 1), dtype=tf.float32)
    inputimg = inputimg / 255 * 2 - 1

    noise_dim = 100
    num_examples_to_generate = 1
    random_vector_for_generation = tf.random_normal([num_examples_to_generate,
                                                     noise_dim])
    generator = Generatorpart()
    generate_and_save_imagespart(generator_restore(generator, checkpointpath),
                             random_vector_for_generation,
                             inputimg,
                             imgsavepath)
