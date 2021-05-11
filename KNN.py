import pandas as pd
import numpy as np
import cv2
import random

imgTable = pd.read_table(r'/Users/sameer/PycharmProjects/AI_CSC668/legend.csv', delimiter=',')

df = pd.DataFrame(imgTable)

testSet = []
testLabels = []
trainSet = []
trainLabels = []

happinessCollection = 0
sadnessCollection = 0
disgustCollection = 0
angerCollection = 0
neutralCollection = 0

a = random.choices(population=range(4, 13000), k=2000)


for i in range(4, 13000):
    imgName = df.loc[i, "image"]
    imgLabel = df.loc[i, "emotion"]
    img = cv2.imread(r'/Users/sameer/PycharmProjects/AI_CSC668/images/' + imgName)
    img = cv2.resize(img, (32, 32), interpolation=cv2.INTER_AREA)

    if i in a and imgLabel.lower() == 'happiness':
        testSet.append(img)
        testLabels.append(0)
        happinessCollection = happinessCollection + 1
    elif i in a and imgLabel.lower() == 'sadness':
        testSet.append(img)
        testLabels.append(1)
        sadnessCollection = sadnessCollection + 1
    elif i in a and imgLabel.lower() == 'anger':
        testSet.append(img)
        testLabels.append(2)
        angerCollection = angerCollection + 1
    elif i in a and imgLabel.lower() == 'neutral':
        testSet.append(img)
        testLabels.append(3)
        neutralCollection = neutralCollection + 1
    elif i in a and imgLabel.lower() == 'disgust':
        testSet.append(img)
        testLabels.append(4)
        disgustCollection = disgustCollection + 1
    elif imgLabel.lower() == 'happiness':
        trainSet.append(img)
        trainLabels.append(0)
        happinessCollection = happinessCollection + 1
    elif imgLabel.lower() == 'sadness':
        trainSet.append(img)
        trainLabels.append(1)
        sadnessCollection = sadnessCollection + 1
    elif imgLabel.lower() == 'anger':
        trainSet.append(img)
        trainLabels.append(2)
        angerCollection = angerCollection + 1
    elif imgLabel.lower() == 'neutral':
        trainSet.append(img)
        trainLabels.append(3)
        neutralCollection = neutralCollection + 1
    elif imgLabel.lower() == 'disgust':
        trainSet.append(img)
        trainLabels.append(4)
        disgustCollection = disgustCollection + 1



print("Neutral:")
print(neutralCollection)
print("Happy: ")
print(happinessCollection)
print("Sad: ")
print(sadnessCollection)
print("Anger: ")
print(angerCollection)
print("Disgust: ")
print(disgustCollection)



emotionTypeCountArray = [happinessCollection,sadnessCollection, angerCollection, neutralCollection, disgustCollection]



trainSet2 = np.array(trainSet)
trainLabels2 = np.array(trainLabels)
testSet2 = np.array(testSet)
testLabels2 = np.array(testLabels)


dataset1_size = trainSet2.shape[0]
dataset2_size = testSet2.shape[0]
data1 = trainSet2.reshape(dataset1_size,-1)
data2 = testSet2.reshape(dataset2_size,-1)




def default_progress_fn(i, total):
    pass

class KNN:
    def __init__ (self, k, progress_fn=default_progress_fn):
        """ Pass k as hyperparameter"""
        self.k = k
        self.progress_fn = progress_fn

    def train(self, X, y):
        """
            X is example training matrix. Every row of X contains one training example. Each training example
            may have `d` features. If there are `m` such examples then X is `m x d` matrix.
            y is the label matrix corrosponding to each training example. Hence y is `m x 1` matrix.
        """

        self.tX = X
        self.ty = np.reshape(y, (len(trainSet), 1))

    def predict(self, X):
        """
            Predict y based on test data X.
        """

        print(self.tX.shape)
        print(X.shape)
        x = np.array(X, ndmin=2)
        num_training = X.shape[0]
        YPred = np.zeros(num_training, dtype = self.ty.dtype)

        for i in range(num_training):
            # Euclidean distance is used to find out distance between two datapoint.
            distances = np.reshape(np.sqrt(np.sum(np.square(self.tX - x[i, :]), axis=1)), (-1, 1))
            # Along with the distance stack the labels so that we can vote easily
            distance_label = np.hstack((distances, self.ty))
            # Simple majority voting based on the minimum distance
            sorted_distance = distance_label[distance_label[:, 0].argsort()]
            k_sorted_distance = sorted_distance[:self.k, :]
            fullArr = [0, 0, 0, 0, 0]
            (labels, occurrence) = np.unique(k_sorted_distance[:, 1], return_counts=True)
            for index in range(len(labels)):
                fullArr[int(labels[index])] = occurrence[index]
            weighted_occurrence = [0, 0, 0, 0, 0]
            for index in range(5):
                weighted_occurrence[index] = fullArr[index] / emotionTypeCountArray[index]
            label = np.argmax(weighted_occurrence)
            YPred[i] = label

            self.progress_fn(i, num_training)

        return YPred

knn = KNN(75)
knn.train(data1, trainLabels2)
guessTupleArr = knn.predict(data2)

accuracyCount = 0

happyAccCount = 0
sadAccCount = 0
angerAccCount = 0
neutralAccCount = 0
disgustAccCount = 0

for i in range(len(guessTupleArr)):
    if guessTupleArr[i] == testLabels[i]:
        accuracyCount = accuracyCount + 1
    if guessTupleArr[i] == testLabels[i] and guessTupleArr[i] == 0:
        happyAccCount = happyAccCount + 1
    elif guessTupleArr[i] == testLabels[i] and guessTupleArr[i] == 1:
        sadAccCount = sadAccCount + 1
    elif guessTupleArr[i] == testLabels[i] and guessTupleArr[i] == 2:
        angerAccCount = angerAccCount + 1
    elif guessTupleArr[i] == testLabels[i] and guessTupleArr[i] == 3:
        neutralAccCount = neutralAccCount + 1
    elif guessTupleArr[i] == testLabels[i] and guessTupleArr[i] == 4:
        disgustAccCount = disgustAccCount + 1

hapTot = 0
sadTot = 0
angerTot = 0
neutralTot = 0
disgustTot = 0

for label in testLabels:
    if label == 0:
        hapTot = hapTot + 1
    elif label == 1:
        sadTot = sadTot + 1
    elif label == 2:
        angerTot = angerTot + 1
    elif label == 3:
        neutralTot = neutralTot + 1
    elif label == 4:
        disgustTot = disgustTot + 1

accuracy = accuracyCount / len(testLabels)
happyAcc = happyAccCount / hapTot
sadAcc = sadAccCount / sadTot
angerAcc = angerAccCount / angerTot
neutralAcc = neutralAccCount / neutralTot
disgustAcc = disgustAccCount / disgustTot

print("accuracy of the model is:")
print(accuracy)

print("Accurate predictions of Happiness:")
print(happyAcc)
print("Accurate predictions of Sadness:")
print(sadAcc)
print("Accurate predictions of Anger:")
print(angerAcc)
print("Accurate predictions of Neutral:")
print(neutralAcc)
print("Accurate predictions of Disgust:")
print(disgustAcc)
