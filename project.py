#without embedder

import glob
import cv2
import matplotlib.pyplot as plt
import numpy as np
#import images 

bike= [cv2.imread(file) for file in glob.glob("train//bike//*.jpg")]
cars = [cv2.imread(file,0) for file in glob.glob("train/cars/*.jpg")]
person = [cv2.imread(file,0) for file in glob.glob("train/person/*.jpg")]
none = [cv2.imread(file,0) for file in glob.glob("train/none/*.jpg")]
ground = [cv2.imread(file,0) for file in glob.glob("train/ground/*.jpg")]

#import labels or creATE LABEL
labels=['bike' for file in glob.glob("train/bike/*.jpg")]
labels.extend(['car' for file in glob.glob("train/cars/*.jpg")])
labels.extend(['ground' for file in glob.glob("train/ground/*.jpg")])
labels.extend(['none' for file in glob.glob("train/none/*.jpg")])
labels.extend(['person' for file in glob.glob("train/person/*.jpg")])

#create dataset 
dataset=[]
dataset.extend(bike)
dataset.extend(cars)
dataset.extend(ground)
dataset.extend(none)
dataset.extend(person)    

dimension=(480,480)

for i in range(0,2377):
    dataset[i]=cv2.resize(dataset[i],dimension,interpolation = cv2.INTER_AREA)
    

dataset=np.reshape(dataset,(2377,480*480))


from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score

train,test,train_label,test_label=train_test_split(dataset,labels,test_size=0.2,random_state=0)

clf=RandomForestClassifier()

clf.fit(train,train_label)
train_accuracy=np.mean(cross_val_score(clf,train,train_label))

predict=clf.predict(test)

from sklearn.metrics import confusion_matrix

cm=confusion_matrix(test_label,predict)