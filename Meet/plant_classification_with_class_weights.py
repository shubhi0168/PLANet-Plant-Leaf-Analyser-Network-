import os
import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow import keras
from keras import backend as K
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras import optimizers
import matplotlib.pyplot as plt
%matplotlib inline

K.clear_session()
print(K.tensorflow_backend._get_available_gpus())

#model architecture

model = tf.keras.models.Sequential([
   tf.keras.layers.Conv2D(input_shape=(128,128,3),filters = 32,kernel_size=(2,2),strides = (1,1),padding='same',data_format='channels_last',activation = 'relu'),
   tf.keras.layers.BatchNormalization(),
   tf.keras.layers.MaxPool2D(),
   tf.keras.layers.Dropout(0.25),
   tf.keras.layers.Conv2D(filters = 64,kernel_size=(3,3),padding='same',data_format = 'channels_last',activation='relu'),
   tf.keras.layers.BatchNormalization(),
   tf.keras.layers.MaxPool2D(),
   tf.keras.layers.Dropout(0.25),
   tf.keras.layers.Conv2D(filters = 64,kernel_size=(3,3),strides = (1,1),padding='same',data_format='channels_last',activation = 'relu'),
   tf.keras.layers.BatchNormalization(), 
   tf.keras.layers.MaxPool2D(),
   tf.keras.layers.Dropout(0.25),
   tf.keras.layers.Conv2D(filters = 128,kernel_size=(3,3),padding='same',data_format = 'channels_last',activation = 'relu'),
   tf.keras.layers.BatchNormalization(),
   tf.keras.layers.MaxPool2D(),
   tf.keras.layers.Dropout(0.25),
   tf.keras.layers.Conv2D(filters = 128,kernel_size=(3,3),padding='same',data_format = 'channels_last',activation = 'relu'),
   tf.keras.layers.BatchNormalization(),
   tf.keras.layers.MaxPool2D(),
   tf.keras.layers.Dropout(0.25),
   tf.keras.layers.Flatten(data_format = 'channels_last'),
   tf.keras.layers.Dense(1024,activation = 'relu',kernel_regularizer = tf.keras.regularizers.l2(l = 0.09)),
   tf.keras.layers.BatchNormalization(),
   tf.keras.layers.Dense(256,activation = 'relu',kernel_regularizer = tf.keras.regularizers.l2(l = 0.09)),
   tf.keras.layers.BatchNormalization(),
   tf.keras.layers.Dense(9,activation='softmax')
])

adam = optimizers.Adam(lr = 0.001,decay = 1e-6)
model.compile(optimizer = adam,
              loss='categorical_crossentropy',
              metrics=['accuracy'])
              
#obtaining class weights
path1 = 'D:/plantdisease/plant_classification'
list = os.listdir(path1)
total_photos = 37851
count = 0
class_weights = dict()
for i in list:
    path2 = os.path.join(path1,i)
    num_class_photos = len(os.listdir(path2))
    class_weights[count] = total_photos/num_class_photos
    count += 1
    
image_size = 128
train_data_generator = tf.keras.preprocessing.image.ImageDataGenerator(
                                   rescale=1./255,
                                   width_shift_range = 0.1,
                                   horizontal_flip=True,
                                   shear_range = 0.2,
                                   zoom_range = 0.2,
                                   validation_split = 0.2,
                                   )

'''test_data_generator = tf.keras.preprocessing.image.ImageDataGenerator(
                                   rescale = 1./255,)'''


train_generator = train_data_generator.flow_from_directory(
        'D:/plantdisease/plant_classification',
        target_size=(image_size, image_size),
        batch_size=16,
        class_mode='categorical',
        color_mode = 'rgb',
        shuffle = True,
        seed = 42,
        subset = 'training')

validation_generator = train_data_generator.flow_from_directory(
       'D:/plantdisease/plant_classification',
        target_size=(image_size, image_size),
        batch_size = 16,
        class_mode='categorical',
        color_mode = 'rgb',
        shuffle = True,
        seed = 42,
        subset = 'validation')
        
        
ilepath="D:/check_points_for_keras/weights-improvement-{epoch:02d}-{val_accuracy:.2f}.hdf5"
callbacks =[EarlyStopping(patience=3, restore_best_weights=True), 
           ReduceLROnPlateau(patience=2), 
           ModelCheckpoint(filepath, monitor='val_accuracy', verbose=0, save_best_only=True, mode='max')]
           
STEP_SIZE_TRAIN=train_generator.n//train_generator.batch_size
STEP_SIZE_VALID=validation_generator.n//validation_generator.batch_size
output = model.fit_generator(train_generator ,
                         steps_per_epoch= STEP_SIZE_TRAIN,
                         epochs= 20,
                         validation_data = validation_generator ,
                         validation_steps = STEP_SIZE_VALID,
                         verbose = 1,
                         callbacks = callbacks,
                         class_weight = class_weights
                         )
model.save('D:/saved_model/plant_classification.h5')

acc_val = output.history['accuracy']
val_acc = output.history['val_accuracy']
loss = output.history['loss']
val_loss = output.history['val_loss']

epochs = range(1,len(acc_val)+1)
plt.plot(epochs, acc_val, 'b', label='Training accuracy')
plt.plot(epochs, val_acc, 'r', label='Validation accuracy')
plt.title('Training and Validation accuracy')
plt.legend()

plt.figure()

plt.plot(epochs, loss, 'b', label='Training accuracy')
plt.plot(epochs, val_loss, 'r', label='Training accuracy')
plt.title('Training and Validation loss')
plt.legend()
plt.show()
