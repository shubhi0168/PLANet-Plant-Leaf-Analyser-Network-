#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
import matplotlib.pyplot as plt


# In[ ]:


train_path = 'D:\plantvillage_deeplearning_paper_dataset\plantvillage_deeplearning_paper_dataset\color'
test_path = 'images/test/'
batch_size = 32
image_size = 128
num_class = 29


# In[ ]:



train_datagen = ImageDataGenerator(validation_split=0.2,
                                   rotation_range=40,
        width_shift_range=0.2,
        height_shift_range=0.2,
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest')


train_generator = train_datagen.flow_from_directory(
                        directory=train_path,
                        
                        target_size=(image_size,image_size),
                        batch_size=batch_size,
                        class_mode='categorical',
                        color_mode='rgb',
                        subset="training",
                        shuffle=True)

validation_generator = train_datagen.flow_from_directory(
                        directory=train_path,
                        target_size=(image_size,image_size),
                        batch_size=batch_size,
                        class_mode='categorical',
                        color_mode='rgb',
                        subset="validation",
                        shuffle=True)


# In[ ]:


x_batch, _ =train_generator.next()
fig=plt.figure()
columns = 2
rows = 2
for i in range(1, columns*rows):
    num = np.random.randint(batch_size)
    image = x_batch[num].astype(np.float)
    fig.add_subplot(rows, columns, i)
    plt.imshow(image)
plt.show()


# In[ ]:


image


# In[ ]:


p=[]
for i in range(38):
    p.append(list(train_generator.labels).count(i))
    print(str(i)+":"+str(p[i]))


# In[ ]:


index = p
plt.bar(range(38), index)
plt.xlabel('Genre', fontsize=5)
plt.ylabel('No of Movies', fontsize=5)


plt.show()


# In[ ]:


p[22]


# In[ ]:


for i,j in zip(train_generator.class_indices,p):
    print(i)
    print(j)


# In[ ]:


import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()


# In[ ]:


tf.__version__


# In[ ]:


import tensorflow.keras
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Activation, Dropout, Flatten, Dense
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.inception_v3 import InceptionV3

#Load the VGG model
base_model = InceptionV3(weights='imagenet', include_top=False, input_shape=(image_size, image_size, 3))


# In[ ]:


x_batch, y_batch = train_generator.next()


# In[ ]:


fig=plt.figure()
columns = 5
rows = 5
for i in range(1, columns*rows):
    num = np.random.randint(batch_size)
    image = x_batch[num].astype(np.float)
    fig.add_subplot(rows, columns, i)
    plt.imshow(image)
plt.show()


# In[ ]:


#Load the VGG model
base_model = InceptionV3(weights='imagenet', include_top=False, input_shape=(image_size, image_size, 3))

print(base_model.summary())

    # Freeze the layers 
# for layer in base_model.layers:
#     layer.trainable = True
 
# # Create the model
model = keras.models.Sequential()

# # Add the vgg convolutional base model
model.add(base_model)
 
# # Add new layers
model.add(Flatten())
model.add(Dense(1024, activation='relu'))
model.add(Dense(1024, activation='relu'))
model.add(Dense(num_class, activation='softmax'))
 
# # Show a summary of the model. Check the number of trainable parameters    
print(model.summary())
model.compile(loss='categorical_crossentropy',
          optimizer='adam',
          metrics=['accuracy'])


# In[ ]:





# In[ ]:


model.fit_generator(
    train_generator,
    steps_per_epoch = train_generator.samples // batch_size,
    validation_data = validation_generator, 
    validation_steps = validation_generator.samples // batch_size,
    epochs = 40)


# In[ ]:


from keras import backend as K
K.tensorflow_backend._get_available_gpus()


# In[ ]:


model.save('scratch_model.h5')
print("Model saved")


# In[ ]:


get_ipython().system('conda install h5py')


# In[ ]:


model.metrics[1]


# In[ ]:




