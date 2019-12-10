# -*- coding: utf-8 -*-
"""
Created on Sat Nov 16 15:38:56 2019

@author: Meet
"""

#!git clone https://github.com/meet-soni5720/plant_disease_dataset
#!git clone https://github.com/spMohanty/PlantVillage-Dataset

import os
import glob
import matplotlib.pyplot as plt
import tensorflow as tf
import keras
import numpy as np
from keras.models import Sequential, Model, load_model
from keras.layers import Conv2D, Activation, BatchNormalization, Reshape, Dense, Input, LeakyReLU, Dropout, Flatten, ZeroPadding2D
from keras.optimizers import Adam
from keras import layers
import argparse
from keras import backend as K
from ast import literal_eval
from PIL import Image
import imageio

K.clear_session()

os.getcwd()
os.listdir()
path1 = os.path.join(os.getcwd(),'PlantVillage-Dataset')
os.listdir(path1)
path2 = os.path.join(path1,'raw/color')
os.listdir(path2)
path3 = os.path.join(path2,'Orange___Haunglongbing_(Citrus_greening)')
os.listdir(path3)
#print(path3)

class DCGAN:
    def __init__(self, discriminator_path, generator_path, output_directory, img_size):
        self.img_size = img_size
        self.upsample_layers = 5
        self.starting_filters = 64
        self.kernel_size = 4
        self.channels = 3
        self.discriminator_path = discriminator_path
        self.generator_path = generator_path
        self.output_directory = output_directory
        self.latent_space_size = 1000

    def build_generator(self):
        noise_shape = (self.latent_space_size,)

        model = Sequential()
        model.add(layers.Dense(8*8*1024, use_bias=False, input_shape=(self.latent_space_size,)))
        model.add(layers.BatchNormalization())
        model.add(layers.LeakyReLU(alpha=0.2))

        model.add(layers.Reshape((8, 8, 1024)))
        assert model.output_shape == (None, 8, 8, 1024) # Note: None is the batch size

        model.add(layers.Conv2DTranspose(512, kernel_size = self.kernel_size, strides=(2, 2), padding='same', use_bias=False))
        assert model.output_shape == (None, 16, 16, 512)
        model.add(layers.BatchNormalization())
        model.add(layers.LeakyReLU(alpha=0.2))

        model.add(layers.Conv2DTranspose(256, kernel_size = self.kernel_size, strides=(2, 2), padding='same', use_bias=False))
        assert model.output_shape == (None, 32, 32, 256)
        model.add(layers.BatchNormalization())
        model.add(layers.LeakyReLU(alpha=0.2))
    
        model.add(layers.Conv2DTranspose(128, kernel_size = self.kernel_size, strides=(2, 2), padding='same', use_bias=False))
        assert model.output_shape == (None, 64, 64, 128)
        model.add(layers.BatchNormalization())
        model.add(layers.LeakyReLU(alpha=0.2))
        
        model.add(layers.Conv2DTranspose(3, kernel_size = self.kernel_size, strides=(2, 2), padding='same', use_bias=False, activation = 'tanh'))
        assert model.output_shape == (None, 128, 128, 3)
        ''' 
        model.add(layers.BatchNormalization())
        model.add(layers.LeakyReLU(alpha = 0.2))

        model.add(layers.Conv2DTranspose(3, kernel_size = (7,7) , strides=(2, 2), padding='same', use_bias=False, activation='tanh'))
        assert model.output_shape == (None, 256, 256, 3)'''

        model.summary()

        noise = Input(shape=noise_shape)
        img = model(noise)

        return Model(noise, img)

    def build_discriminator(self):

        img_shape = (self.img_size[0], self.img_size[1], self.channels)

        model = Sequential()

        model.add(Conv2D(32, kernel_size=self.kernel_size, strides=2, input_shape=img_shape, padding="same"))  # 256x256 -> 128x128
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.5))

        model.add(Conv2D(64, kernel_size=self.kernel_size, strides=2, padding="same"))  # 128x128-> 64x64
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.5))
        model.add(BatchNormalization())
  
        model.add(Conv2D(128, kernel_size=self.kernel_size, strides=2, padding="same"))  # 64x64 -> 32x32
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.5))
        model.add(BatchNormalization())
        
        model.add(Conv2D(256, kernel_size=self.kernel_size, strides=2, padding="same"))  # 32x32 -> 16x16
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.5))
        model.add(BatchNormalization())
        
        model.add(Conv2D(256, kernel_size=self.kernel_size, strides=2, padding="same"))  # 16x16 -> 8x8
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.5))
        model.add(BatchNormalization())

        model.add(Flatten())
        ''' model.add(Dense(1024))
        model.add(LeakyReLU(alpha=0.2))
        model.add(BatchNormalization())'''
        model.add(Dense(1,activation = 'sigmoid'))

        model.summary()

        img = Input(shape=img_shape)
        validity = model(img)

        return Model(img, validity)

    def build_gan(self): 
        optimizer = Adam(lr = 0.0002, beta_1 =  0.5)

        # See if the specified model paths exist, if they don't then we start training new models

        if os.path.exists(self.discriminator_path) and os.path.exists(self.generator_path):
            self.discriminator = load_model(self.discriminator_path)
            self.generator = load_model(self.generator_path)
            print("Loaded models...")
        else:
            self.discriminator = self.build_discriminator()
            self.discriminator.compile(loss='binary_crossentropy',
                                       optimizer=optimizer,
                                       metrics=['accuracy'])

            self.generator = self.build_generator()
            self.generator.compile(loss='binary_crossentropy', optimizer=optimizer)

        # These next few lines setup the training for the GAN model
        z = Input(shape=(self.latent_space_size,))
        img = self.generator(z)

        self.discriminator.trainable = False

        valid = self.discriminator(img)

        self.combined = Model(z, valid)
        self.combined.compile(loss='binary_crossentropy', optimizer=optimizer)

    def load_imgs(self, image_path):
        X_train = []
        for i in glob.glob(image_path):
            basewidth = 128
            img = Image.open(i)
            wpercent = (basewidth/float(img.size[0]))
            hsize = int((float(img.size[1])*float(wpercent)))
            img = img.resize((basewidth,hsize), Image.ANTIALIAS)
            img = np.asarray(img)
            X_train.append(img)
        return np.asarray(X_train)

    def train(self, epochs, image_path, batch_size=128, save_interval=100):
        self.build_gan()
        X_train = self.load_imgs(image_path)
        print("Training Data Shape: ", X_train.shape)

        # Rescale images from -1 to 1
        X_train = (X_train.astype(np.float32) - 127.5) / 127.5

        half_batch = batch_size // 2

        for epoch in range(epochs):


            # Train Generator
            noise = np.random.normal(0, 1, (batch_size, self.latent_space_size))
            g_loss = self.combined.train_on_batch(noise, np.ones((batch_size, 1)))



            # Train Discriminator
            idx = np.random.randint(0, X_train.shape[0], half_batch)
            imgs = X_train[idx]

            # Sample noise and generate a half batch of new images
            noise = np.random.normal(0, 1, (half_batch, self.latent_space_size))
            gen_imgs = self.generator.predict(noise)

            # Train the discriminator (real classified as ones and generated as zeros)
            d_loss_real = self.discriminator.train_on_batch(imgs, (np.ones((half_batch, 1)))*0.9)
            d_loss_fake = self.discriminator.train_on_batch(gen_imgs, np.zeros((half_batch, 1)))
            d_loss = 0.5*np.add(d_loss_real, d_loss_fake) #to make it in range of 100

            # Print progress
            print(f"{epoch} [D loss: {d_loss[0]} | D Accuracy: {100 * d_loss[1]}] [G loss: {g_loss}]")

            # If at save interval => save generated image samples, save model files
            if epoch % (save_interval) == 0:

                self.save_imgs(epoch)

                save_path = self.output_directory + "/models"
                if not os.path.exists(save_path):
                    os.makedirs(save_path)
                self.discriminator.save(save_path + "/discrim.h5")
                self.generator.save(save_path + "/generat.h5")

    def gene_imgs(self, count):
        # Generate images from the currently loaded model
        noise = np.random.normal(0, 1, (count, self.latent_space_size))
        return self.generator.predict(noise)

    def save_imgs(self, epoch):
        r, c = 5, 5

        # Generates r*c images from the model, saves them individually and as a gallery

        imgs = self.gene_imgs(r*c)
        imgs = 0.5 * imgs + 0.5

        for i, img_array in enumerate(imgs):
            path = f"{self.output_directory}/generated_{self.img_size[0]}x{self.img_size[1]}"
            if not os.path.exists(path):
                os.makedirs(path)
            imageio.imwrite(path + f"/{epoch}_{i}.png", img_array)

        nindex, height, width, intensity = imgs.shape
        nrows = nindex // c
        assert nindex == nrows * c
        # want result.shape = (height*nrows, width*ncols, intensity)
        gallery = (imgs.reshape(nrows, c, height, width, intensity)
                  .swapaxes(1, 2)
                  .reshape(height * nrows, width * c, intensity))

        path = f"{self.output_directory}/gallery_generated_{self.img_size[0]}x{self.img_size[1]}"
        if not os.path.exists(path):
            os.makedirs(path)
        imageio.imwrite(path + f"/{epoch}.png", gallery)

    def generate_imgs(self, count, threshold, modifier):
        self.build_gan()

        # Generates (count) images from the model ensuring the discriminator scores them between the threshold values
        # and saves them

        imgs = []
        for i in range(count):
            score = [0]
            while not(threshold[0] < score[0] < threshold[1]):
                img = self.gene_imgs(1)
                score = self.discriminator.predict(img)
            print("Image found: ", score[0])
            imgs.append(img)

        imgs = np.asarray(imgs).squeeze()
        imgs = 0.5 * imgs + 0.5

        print(imgs.shape)
        for i, img_array in enumerate(imgs):
            path = f"{self.output_directory}/generated_{threshold[0]}_{threshold[1]}"
            if not os.path.exists(path):
                os.makedirs(path)
            imageio.imwrite(path + f"/{modifier}_{i}.png", img_array)
            
image_size = (128,128)
thresholds = [0.6,1]
dcgan = DCGAN(os.path.join(os.getcwd(),'generator_weights'),os.path.join(os.getcwd(),'discriminator_weights'),os.path.join(os.getcwd(),'test_gans'),image_size)
dcgan.train(epochs = 5000, image_path = os.path.join(path3, '*.JPG') ,batch_size=128,save_interval=100)

path_image = os.path.join(os.getcwd(),'test_gans')
c = os.listdir(path_image)
d = os.path.join(path_image,'gallery_generated_128x128')
images = os.listdir(d)
print(images)

s = np.array(Image.open(os.path.join(d,'1500.png')))             # images are saved upto 10000 epochs so you can try at number less than 10000 at interval 100
print(s.shape)
plt.imshow(s)