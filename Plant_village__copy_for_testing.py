#!/usr/bin/env python
# coding: utf-8

# In[24]:


import tensorflow as tf,os,re,shutil,numpy as np,time,matplotlib.pyplot as plt
from os import path as Path
from PIL import Image
from tensorflow.keras.preprocessing.image import ImageDataGenerator 
from tensorflow.keras.models import Sequential , load_model
from tensorflow.keras import optimizers,backend as K 
from tensorflow.keras.layers import Dense,MaxPool2D,Conv2D,Flatten,AveragePooling2D,BatchNormalization,ZeroPadding2D,Input,Activation

from keras.preprocessing import image
import numpy as np

from matplotlib import style
style.use('dark_background')

import warnings
warnings.filterwarnings("ignore")


# In[2]:


path = "PlantVillage-Dataset/raw/color"
folders = os.listdir(path)


# In[3]:


nofiles=0
for i in os.listdir(path):
    nofiles+=len(os.listdir(path+'/'+i))
print(nofiles)


# In[4]:


_max= 0
flag=True
numerator = 0
denominator= len(os.listdir(path))
for folder in os.listdir(path):
    length = len(os.listdir(path+'/'+folder))
    print(folder,length)
    new_path = folder
    if flag :
        _min = length
        flag = False
    if length > _max:
        _max = length
        max_path = new_path
    if length<_min:
        _min = length
        min_path=new_path
    numerator+=length
        
print('\n',_max,max_path,'\n',_min,min_path,'\n','average size=',int(numerator/denominator))


# In[5]:


folders_healthy = [i for i in folders if re.search(r"healthy",i)]
folders_with_disease = [i for i in folders if not re.search(r"healthy",i)]

names =[]
for i in folders:
    check = re.compile(r"^[^_,]+")  # meaning of regex - ^ (begining of the str) [^_,] (except _ and ,) + (one or more)
    match = check.match(i)
    if match:
        names.append(match.group())

Dict = {}
for name in set(names):
    pos_array = []
    pos_array = [i for i in range(len(names)) if names[i] == name]
    Dict[name] = pos_array    
    
    

print("\nHealthy_folders:\n",folders_healthy)
print("\nNot_Healthy_folders:\n",folders_with_disease)
print("\nNames of species:\n",set(names))
print("\nDictionary:\n",Dict)


# In[6]:


def make_folders(remove_folders=False):
    
    if Path.exists("Health") and remove_folders:
        print('Deleting prexisting folders')
        shutil.rmtree("Health")
        shutil.rmtree("Species")

        print('Creating Species and Health folders')
        os.mkdir('Species')
        os.mkdir('Health') 
        
        for key in Dict.keys():
            array = Dict[key]
            os.makedirs('Species/'+key,exist_ok=True)
            
            for i in array:
                source = path +'/'+ folders[i]
                destination = 'Species/' + key
                
            for file in os.listdir(source):
                file_name = source + '/' + file
                shutil.copy(file_name,destination)
        print('Species folder successfully created for all {} classes'.format(len(os.listdir('Species'))))        
                
        if not Path.exists('Health/healthy'):
            os.makedirs ('Health/healthy')
            os.makedirs('Health/not_healthy')
            
            for folder in folders_healthy:
                print('healthy: {}'.format(folder))
                for file in os.listdir(path+'/'+folder):
                    shutil.copy(path+'/'+folder +'/'+ file,"Health/healthy")
            print('\n Heathy folder created\n')
            for folder in folders_with_disease:
                print('disease: {}'.format(folder))
                for file in os.listdir(path+'/'+folder):
                    shutil.copy(path+'/'+folder +'/'+ file,"Health/not_healthy")      
            print('\n not_heathy folder created\n')
    else: print('Folders already made')
        
make_folders()


# In[7]:


img_width, img_height = 256,256
batch_size = 32
input_shape = (img_width, img_height, 3) 


# In[8]:


model_health = Sequential([

                    Conv2D(32, (2, 2), input_shape = (256, 256, 3),data_format='channels_last'),
                    Activation('relu'),
                    MaxPool2D(pool_size =(2, 2)),

                    Conv2D(64, (2, 2)),
                    Activation('relu'),
                    MaxPool2D(pool_size =(2, 2)),

                    Conv2D(128, (2, 2)),
                    Activation('relu'),
                    MaxPool2D(pool_size =(2, 2)),

                    Flatten(),
                    Dense(1,activation='sigmoid'),
    ])

model_health.compile(loss ='binary_crossentropy', optimizer = 'adam', metrics =['accuracy']) 
print('\t\t\tHealth Model\n\n')
model_health.summary()


# In[9]:


model_species = Sequential([

                    Conv2D(32, (2, 2), input_shape = (256, 256, 3),data_format='channels_last'),
                    Activation('relu'),
                    MaxPool2D(pool_size =(2, 2)),

                    Conv2D(64, (2, 2)),
                    Activation('relu'),
                    MaxPool2D(pool_size =(2, 2)),

                    Conv2D(128, (2, 2)),
                    Activation('relu'),
                    MaxPool2D(pool_size =(2, 2)),

                    Flatten(),
                    Dense(14,activation='softmax'),
    ])

model_species.compile(loss ='categorical_crossentropy', optimizer = 'adam', metrics =['accuracy']) 
print('\t\t\tSpecies Model\n\n')
model_species.summary()


# In[10]:


def create_train_and_validation(train_data_dir):
    if train_data_dir=='Health':
        class_mode='binary'
    else:
        class_mode='categorical'

    train_datagen = ImageDataGenerator(rescale = 1./256,
                                   rotation_range = 60,
                                   width_shift_range=0.1,
                                   height_shift_range=0.1,
                                   zoom_range=10,
                                   horizontal_flip=True,
                                   vertical_flip=True,
                                   data_format ="channels_last",validation_split=0.2) 

    train_generator = train_datagen.flow_from_directory(train_data_dir,
                                                        target_size =(img_width, img_height),
                                                        batch_size = batch_size,
                                                        class_mode = class_mode,
                                                        subset='training') 

    validation_generator = train_datagen.flow_from_directory(train_data_dir,
                                                            target_size =(img_width, img_height),
                                                            batch_size = batch_size,
                                                            class_mode = class_mode,
                                                            subset='validation') 
    return train_datagen,train_generator,validation_generator


# In[11]:


run_species,run_health = False,False


# In[12]:


if run_species:
    train_datagen_species,train_generator_species,validation_generator_species = create_train_and_validation("Species")
    past_time = time.time()
    history_species = model_species.fit_generator(generator=train_generator_species,
                                 steps_per_epoch=train_generator_species.samples//batch_size,
                                 epochs=10,
                                 validation_data=validation_generator_species,
                                 validation_steps=validation_generator_species.samples//batch_size)
    model_species.save('species_model.h5')
    present_time = time.time()
    time_taken_species = (present_time-past_time)/3600
    print('The program took {} hours to run'.format(time_taken_species))


# In[13]:


if run_health:
    train_datagen_health,train_generator_health,validation_generator_health = create_train_and_validation("Health")
    past_time = time.time()
    history_health = model_health.fit_generator(generator=train_generator_health,
                                 steps_per_epoch=train_generator_health.samples//batch_size,
                                 epochs=10,
                                 validation_data=validation_generator_health,
                                 validation_steps=validation_generator_health.samples//batch_size)
    model_species.save('health_model2.h5')
    present_time = time.time()
    time_taken_health = (present_time-past_time)/3600
    print('The program took {} hours to run'.format(time_taken_health))


# In[27]:


image_path = 'PlantVillage-Dataset/raw/color/Apple___healthy/0055dd26-23a7-4415-ac61-e0b44ebfaf80___RS_HL 5672.JPG'
img = image.load_img(image_path, target_size=input_shape)
plt.imshow(img)

