import os
import cv2
import argparse

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from  scipy.misc.pilutil import *
from keras.callbacks import TensorBoard, ModelCheckpoint



tf.random.set_random_seed(42)



parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, required=True)
parser.add_argument('--epochs', type=int, default=25, required=False)
parser.add_argument('--batch_size', type=int, default=4, required=False)
parser.add_argument('--data_location', default='data/', type=str, required=False)
parser.add_argument('--size', default=384, type=int, required=False)
parser.add_argument('--restore', default=False, type=bool, required=False)
args = parser.parse_args()


data_location = args.data_location

training_images_loc = data_location + f'{args.dataset}/aug/images/'
training_label_loc = data_location + f'{args.dataset}/aug/label/'

train_files = os.listdir(training_images_loc)
train_data = []
train_label = []

desired_size = args.size

for i in train_files:
    im = imread(training_images_loc + i)
    label = imread(training_label_loc + i.split('fundus')[0] + 'fundus_ref.bmp',mode="L")

    im = im[:desired_size,:desired_size]
    label = label[:desired_size,:desired_size]

    train_data.append(im)
    _, temp = cv2.threshold(label, 127, 255, cv2.THRESH_BINARY)
    train_label.append(temp)


train_data = np.array(train_data)
train_label = np.array(train_label)

x_train = train_data.astype('float32') / 255.
y_train = train_label.astype('float32') / 255.
x_train = np.reshape(x_train, (
len(x_train), desired_size, desired_size, 3))  # adapt this if using `channels_first` image data format
y_train = np.reshape(y_train, (len(y_train), desired_size, desired_size, 1))  # adapt this if using `channels_first` im

TensorBoard(log_dir='./logs', histogram_freq=0,
            write_graph=True, write_images=True)


from  SA_UNet import *
model=SA_UNet(input_size=(desired_size,desired_size,3),start_neurons=16,lr=1e-3,keep_prob=0.82,block_size=7)
model.summary()
weight=f"Model/{args.dataset}_SA_UNet.h5"
restore=args.restore

if restore and os.path.isfile(weight):
    model.load_weights(weight)

model_checkpoint = ModelCheckpoint(weight, monitor='val_accuracy', verbose=1, save_best_only=False)


history=model.fit(x_train, y_train,
                epochs=args.epochs, #first  100 with lr=1e-3,,and last 50 with lr=1e-4
                batch_size=args.batch_size,
                validation_split=0.15,
                shuffle=True,
                callbacks= [TensorBoard(log_dir='./logs'), model_checkpoint])

print(history.history.keys())

# summarize history for accuracy
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('SA-UNet Accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'validate'], loc='lower right')
plt.show()