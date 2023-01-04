import tensorflow as tf
import logging
tf.get_logger().setLevel(logging.ERROR)
import numpy as np
import matplotlib.pyplot as plt
import cv2

mainPath = './dataset/'
imgSize = (64,64)
batchSize = 64

from keras.utils import image_dataset_from_directory

xTrain = image_dataset_from_directory(mainPath,subset="training",validation_split=0.2,image_size=imgSize,batch_size=batchSize,seed=123)
xVal = image_dataset_from_directory(mainPath,subset="validation",validation_split=0.2,image_size=imgSize,batch_size=batchSize,seed=123)
classes = xTrain.class_names
print(classes)

N = 10
plt.figure()
for img,label in xTrain.take(1):
    for i in range(N):
        plt.subplot(2,int(N/2),i+1)
        plt.imshow(img[i].numpy().astype("uint8"))
        plt.title(classes[label[i]])
        plt.axis("off")

plt.show()

from keras import layers
from keras import Sequential

# iskreno nisam sigurna da li je ovo potrebno, ali uradila je na vezbama pa sta znam, pogledaj
dataAugmentation = Sequential (
    [
        layers.RandomFlip("horizontal",input_shape=(imgSize[0],imgSize[1],3)),
        layers.RandomRotation(0.25),
        layers.RandomZoom(0.1),
    ]
)

N=10
plt.figure()
for img,lab in xTrain.take(1):
    plt.title(classes[lab[0]])
    for i in range(N):
        aug_img = dataAugmentation(img)
        plt.subplot(2,int(N/2),i+1)
        plt.imshow(aug_img[0].numpy().astype('uint8'))
        plt.axis('off')

plt.show()

from keras.optimizers import Adam
from keras.losses import SparseCategoricalCrossentropy

model = Sequential (
    [
        dataAugmentation,
        layers.Rescaling(1./255,input_shape=(64,64,3)),
        layers.Conv2D(16,3,padding='same',activation='relu'),
        layers.MaxPooling2D(),
        layers.Conv2D(32,3,padding='same',activation='relu'),
        layers.MaxPooling2D(),
        layers.Conv2D(64,3,padding='same',activation='relu'),
        layers.MaxPooling2D(),
        layers.Flatten(),
        layers.Dense(128,activation='relu'),
        layers.Dense(len(classes),activation='softmax')
    ]
)


model.summary()
model.compile(Adam(learning_rate=0.001),loss=SparseCategoricalCrossentropy(),metrics='accuracy')
print("DO OVDE RADI KAKO TREBA NAKON COMPILE")
history = model.fit(xTrain,epochs=50,validation_data=xVal,verbose=0)




