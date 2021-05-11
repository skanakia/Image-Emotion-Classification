import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np
import cv2
import random

imgTable = pd.read_table(r'/Users/sameer/PycharmProjects/AI_CSC668/legend.csv', delimiter=',')

df = pd.DataFrame(imgTable)

trainSet = []
trainLabels = []
testSet = []
testLabels = []


happinessCollection = 0
sadnessCollection = 0
angerCollection = 0
neutralCollection = 0
disgustCollection = 0

a = random.choices(population=range(4, 10000), k=2000)

for i in range(4, 10000):
    imgName = df.loc[i, "image"]
    imgLabel = df.loc[i, "emotion"]
    img = cv2.imread(r'/Users/sameer/PycharmProjects/AI_CSC668/images/' + imgName)


    if len(img) == 350 and i in a and imgLabel.lower() == 'happiness':
        testSet.append(img)
        testLabels.append(0)
        happinessCollection = happinessCollection + 1
    elif len(img) == 350 and i in a and imgLabel.lower() == 'sadness':
        testSet.append(img)
        testLabels.append(1)
        sadnessCollection = sadnessCollection + 1
    elif len(img) == 350 and i in a and imgLabel.lower() == 'anger':
        testSet.append(img)
        testLabels.append(2)
        angerCollection = angerCollection + 1
    elif len(img) == 350 and i in a and imgLabel.lower() == 'neutral':
        testSet.append(img)
        testLabels.append(3)
        neutralCollection = neutralCollection + 1
    elif len(img) == 350 and i in a and imgLabel.lower() == 'disgust':
        testSet.append(img)
        testLabels.append(4)
        disgustCollection = disgustCollection + 1
    elif len(img) == 350 and imgLabel.lower() == 'happiness':
        trainSet.append(img)
        trainLabels.append(0)
        happinessCollection = happinessCollection + 1
    elif len(img) == 350 and imgLabel.lower() == 'sadness':
        trainSet.append(img)
        trainLabels.append(1)
        sadnessCollection = sadnessCollection + 1
    elif len(img) == 350 and imgLabel.lower() == 'anger':
        trainSet.append(img)
        trainLabels.append(2)
        angerCollection = angerCollection + 1
    elif len(img) == 350 and imgLabel.lower() == 'neutral':
        trainSet.append(img)
        trainLabels.append(3)
        neutralCollection = neutralCollection + 1
    elif len(img) == 350 and imgLabel.lower() == 'disgust':
        trainSet.append(img)
        trainLabels.append(4)
        disgustCollection = disgustCollection + 1


(train_images, train_labels) = (trainSet, trainLabels)
(test_images, test_labels) = (testSet, testLabels)

train_images = np.asarray(train_images, dtype=np.uint8)
train_labels = np.asarray(train_labels,  dtype=np.int)

test_images = np.asarray(test_images, dtype=np.uint8)
test_labels = np.asarray(test_labels,  dtype=np.int)

# Normalize pixel values to be between 0 and 1
train_images, test_images = train_images / 255.0, test_images / 255.0

tf.convert_to_tensor(train_labels)
tf.convert_to_tensor(train_images, dtype= tf.float32)
tf.convert_to_tensor(test_images, dtype= tf.float32)
tf.convert_to_tensor(test_labels)



model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(350, 350, 3)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2,2)))


model.add(layers.Flatten())
model.add(layers.Dense(64, activation='relu', ))
model.add(layers.Dense(5))

model.summary()

model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])


history = model.fit(train_images, train_labels, epochs=10,
                    validation_data=(test_images, test_labels))


test_loss, test_acc = model.evaluate(test_images,  test_labels, verbose=2)

print(test_acc)