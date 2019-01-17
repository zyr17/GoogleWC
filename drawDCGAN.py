# -*- coding: utf-8 -*-
"""
Created on Tue Jan 15 14:59:08 2019

@author: 13236
"""

from __future__ import absolute_import, division, print_function

# Import TensorFlow >= 1.10 and enable eager execution
import tensorflow as tf
tf.enable_eager_execution()

import os
import time
import numpy as np
import glob
import matplotlib.pyplot as plt
import PIL
import imageio
import math
import random
from IPython import display
from plot import plot

def clearhalf(olddata, fillcolor = 0):
    data = np.array(olddata)
    for one in data:
        k = math.tan((random.random() - 0.5) * math.pi * 23 / 24) + 0.001
        left = 0
        right = 0
        for i in range(28):
            mid = 14 + (i - 14) * k
            for j in range(28):
                if j < mid:
                    left += one[i][j]
                else:
                    right += one[i][j]
        fillleft = left < right
        for i in range(28):
            mid = 14 + (i - 14) * k
            for j in range(28):
                if (j < mid) == fillleft:
                    one[i][j] = fillcolor
    return data

imagenames = ['triangle']

LABEL_NUM = len(imagenames)
ONE_LABEL_SAMPLE = 10000 // LABEL_NUM

train_images = []
train_parts = []

random_img_for_generation = []

for i in range(len(imagenames)):
  tmpdata = np.load('data/' + imagenames[i] + '.npy')
  tmpdata = tmpdata.reshape(-1, 28, 28)
  timg = tmpdata[:ONE_LABEL_SAMPLE]
  random_img_for_generation.append(tmpdata[:-100])
  tpart = clearhalf(timg)
  train_images.append(timg)
  train_parts.append(tpart)

train_images = np.vstack(train_images)
train_parts = np.vstack(train_parts)

#train_labels = tf.one_hot(train_labels, LABEL_NUM, 1, 0)

#train_images = np.load('data/' + 'cloud.npy')
num_total = train_images.shape[0]
train_images = train_images.reshape((num_total,28,28))

#(train_images, train_labels), (_, _) = tf.keras.datasets.mnist.load_data()

train_images = train_images.reshape(train_images.shape[0], 28, 28, 1).astype('float32')
train_parts = train_parts.reshape(-1, 28, 28, 1).astype('float32')
# We are normalizing the images to the range of [-1, 1]
train_images = (train_images - 127.5) / 127.5
train_parts = (train_parts - 127.5) / 127.5

BUFFER_SIZE = len(train_images)
BATCH_SIZE = 256

train_dataset = tf.data.Dataset.from_tensor_slices({'img': train_images, 'part': train_parts}).shuffle(BUFFER_SIZE).batch(BATCH_SIZE)

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

class Discriminator(tf.keras.Model):
  def __init__(self):
    super(Discriminator, self).__init__()
    self.conv1 = tf.keras.layers.Conv2D(64, (5, 5), strides=(2, 2), padding='same')
    self.conv2 = tf.keras.layers.Conv2D(128, (5, 5), strides=(2, 2), padding='same')
    self.dropout = tf.keras.layers.Dropout(0.3)
    self.flatten = tf.keras.layers.Flatten()
    self.fc1 = tf.keras.layers.Dense(1)
    self.fc2 = tf.keras.layers.Dense(28*28)

  def call(self, x, part, training=True):
    x = tf.concat([x,part], 3)

    x = tf.nn.leaky_relu(self.conv1(x))
    
    x = self.dropout(x, training=training)
    x = tf.nn.leaky_relu(self.conv2(x))
    x = self.dropout(x, training=training)
    x = self.flatten(x)
    x = self.fc1(x)
    return x

generator = Generator()
discriminator = Discriminator()

# Defun gives 10 secs/epoch performance boost
generator.call = tf.contrib.eager.defun(generator.call)
discriminator.call = tf.contrib.eager.defun(discriminator.call)

def discriminator_loss(real_output, generated_output):
    # [1,1,...,1] with real output since it is true and we want
    # our generated examples to look like it
    real_loss = tf.losses.sigmoid_cross_entropy(multi_class_labels=tf.ones_like(real_output), logits=real_output)
    #print(real_output)
    #print(real_output.shape)
    #raise Exception('pass')
    
    # [0,0,...,0] with generated images since they are fake
    generated_loss = tf.losses.sigmoid_cross_entropy(multi_class_labels=tf.zeros_like(generated_output), logits=generated_output)

    total_loss = real_loss + generated_loss

    return real_loss, generated_loss

def generator_loss(generated_output):
    return tf.losses.sigmoid_cross_entropy(tf.ones_like(generated_output), generated_output)

discriminator_optimizer = tf.train.AdamOptimizer(1e-4)
generator_optimizer = tf.train.AdamOptimizer(1e-4)

checkpoint_dir = './training_checkpoints'
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
checkpoint = tf.train.Checkpoint(generator_optimizer=generator_optimizer,
                                 discriminator_optimizer=discriminator_optimizer,
                                 generator=generator,
                                 discriminator=discriminator)

EPOCHS = 150000
noise_dim = 100
num_examples_to_generate = 16

# keeping the random vector constant for generation (prediction) so
# it will be easier to see the improvement of the gan.
random_vector_for_generation = tf.random_normal([num_examples_to_generate,
                                                 noise_dim])
for i in range(LABEL_NUM):
    random_img_for_generation[i] = random_img_for_generation[i][:num_examples_to_generate // LABEL_NUM]
random_img_for_generation = np.vstack(random_img_for_generation)
random_part_for_generation = clearhalf(random_img_for_generation)
random_part_for_generation = random_img_for_generation
random_part_for_generation = random_part_for_generation.reshape(-1, 28, 28, 1).astype('float32')
random_part_for_generation = random_part_for_generation * 2 / 255 - 1

#print(random_img_for_generation.shape, random_part_for_generation.shape)

def generate_and_save_images(model, epoch, test_input, test_part):
  # make sure the training parameter is set to False because we
  # don't want to train the batchnorm layer when doing inference.
  predictions = model(test_input, test_part, training=False)

  fig = plt.figure(figsize=(4,8))
  
  #print(test_part.shape, predictions.shape)

  for i in range(predictions.shape[0]):
      plt.subplot(4, 8, i*2+1)
      plt.imshow(test_part[i, :, :, 0] * 127.5 + 127.5, cmap='gray')
      plt.axis('off')
      plt.subplot(4, 8, i*2+2)
      plt.imshow(predictions[i, :, :, 0] * 127.5 + 127.5, cmap='gray')
      plt.axis('off')
        
  plt.savefig('imgs/image_at_epoch_{:04d}.png'.format(epoch))
  plt.show()

GEN_TIME = 1
DISC_TIME = 1
   
def train(dataset, epochs, noise_dim):  
    
  gen_loss_record, disc_loss_record = [], []
  for epoch in range(epochs):
    start = time.time()
    
    index = 0
    for onedata in dataset:
      images = onedata['img']
      parts = onedata['part']
      '''
      print(images.shape, labels.shape)
      for i in range(5):
        print(labels[i])
      '''
      
      gen_time = GEN_TIME
      disc_time = DISC_TIME

      # generating noise from a uniform distribution
      noise = tf.random_normal([parts.shape[0], noise_dim])
      for tot_time in range(999999):
        with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
          #print(images.shape, labels.shape)
          generated_images = generator(noise, parts, training=True)
          generated_output = discriminator(generated_images, parts, training=True)
          gen_loss = generator_loss(generated_output)
          gen_loss_record.append(gen_loss.numpy())
          if disc_time > 0:
            real_output = discriminator(images, parts, training=True)
            [real_loss, fake_loss] = discriminator_loss(real_output, generated_output)
            disc_loss = real_loss + fake_loss
            disc_loss_record.append((disc_loss).numpy())
        if gen_time > 0:
          gradients_of_generator = gen_tape.gradient(gen_loss, generator.variables)
          generator_optimizer.apply_gradients(zip(gradients_of_generator, generator.variables))
          gen_time -= 1

        if disc_time > 0:
          gradients_of_discriminator = disc_tape.gradient(disc_loss, discriminator.variables)
          discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, discriminator.variables))
          disc_time -= 1

        if gen_time == 0 and disc_time == 0:
          break
 
      
      index += 1
      if index%20 == 0: 
          print('epoch={}, gen_loss={:.6f}, disc_loss = {:.6f}, real_loss={:.6f}, fake_loss={:.6f}'.format(epoch, gen_loss, disc_loss, real_loss, fake_loss))
      
    if epoch:
      #display.clear_output(wait=True)
      plot(epoch, gen_loss_record, 'gen_loss_record')
      plot(epoch, disc_loss_record, 'disc_loss_record')
      generate_and_save_images(generator,
                               epoch + 1,
                               random_vector_for_generation,
                               random_part_for_generation)
      
    
    # saving (checkpoint) the model every 15 epochs
    if (epoch + 1) % 15 == 0:
      checkpoint.save(file_prefix = checkpoint_prefix)
    
    print ('Time taken for epoch {} is {} sec'.format(epoch,
                                                      time.time()-start))
  # generating after the final epoch
  display.clear_output(wait=True)
  generate_and_save_images(generator,
                           epochs,
                           random_vector_for_generation,
                           random_part_for_generation)
  
train(train_dataset, EPOCHS, noise_dim)

# restoring the latest checkpoint in checkpoint_dir
checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir))

def display_image(epoch_no):
  return PIL.Image.open('imgs/image_at_epoch_{:04d}.png'.format(epoch_no))

display_image(EPOCHS)

with imageio.get_writer('dcgan.gif', mode='I') as writer:
  filenames = glob.glob('imgs/image*.png')
  filenames = sorted(filenames)
  last = -1
  for i,filename in enumerate(filenames):
    frame = 2*(i**0.5)
    if round(frame) > round(last):
      last = frame
    else:
      continue
    image = imageio.imread(filename)
    writer.append_data(image)
  image = imageio.imread(filename)
  writer.append_data(image)
    
# this is a hack to display the gif inside the notebook
os.system('cp dcgan.gif dcgan.gif.png')

display.Image(filename="dcgan.gif.png")
