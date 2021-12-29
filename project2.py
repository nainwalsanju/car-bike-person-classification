#with embeddings
import glob
import cv2
import matplotlib.pyplot as plt
import numpy as np
from keras.utils.vis_utils import plot_model
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
#from keras.applications.vgg16 import preprocess_input,decode_predictions
#from keras.applications.vgg16 import VGG16

import numpy as np
from keras_squeezenet import SqueezeNet
from keras.applications.imagenet_utils import preprocess_input, decode_predictions
from keras.preprocessing import image


# load the model

#import images 

bike= [cv2.imread(file) for file in glob.glob("train/bike/*.jpg")]
cars = [cv2.imread(file) for file in glob.glob("train/cars/*.jpg")]
person = [cv2.imread(file) for file in glob.glob("train/person/*.jpg")]
none = [cv2.imread(file) for file in glob.glob("train/none/*.jpg")]
ground = [cv2.imread(file) for file in glob.glob("train/ground/*.jpg")]

#import labels or creATE LABEL
labels=[0 for file in glob.glob("train/bike/*.jpg")]
labels.extend([1 for file in glob.glob("train/cars/*.jpg")])
labels.extend([2 for file in glob.glob("train/ground/*.jpg")])
labels.extend([3 for file in glob.glob("train/none/*.jpg")])
labels.extend([4 for file in glob.glob("train/person/*.jpg")])

#create dataset 
dataset=[]
dataset.extend(bike)
dataset.extend(cars)
dataset.extend(ground)
dataset.extend(none)
dataset.extend(person)

dimension=(227,227)

for i in range(0,2377):
    dataset[i]=cv2.resize(dataset[i],dimension,interpolation = cv2.INTER_AREA)
    
#get embeddings from image
#model =VGG16(weights='imagenet',include_top=False)

model=SqueezeNet()
plot_model(model, to_file='vgg.png')

embeddings=[]
for img in dataset:
    img = img_to_array(img)
    img=np.expand_dims(img,axis=0)
    img = preprocess_input(img)
    embeddings.append(model.predict(img))

#final_dataset=np.reshape(embeddings,(2377,1*7*7*512))
final_dataset=np.reshape(embeddings,(2377,1000))

#split dataset



from sklearn.model_selection import train_test_split

train,test,train_label,test_label=train_test_split(final_dataset,labels,test_size=0.2,random_state=50)


#Neural Network
from keras.models import Sequential
from keras.layers import Dense

clf=Sequential()
#input layer
clf.add(Dense(500,activation='relu',input_dim=1000))

#hidden layer
clf.add(Dense(250,activation='relu'))

clf.add(Dense(100,activation='relu'))

clf.add(Dense(50,activation='relu'))

clf.add(Dense(5,activation='relu'))

#final layer
clf.add(Dense(1))

   
clf.compile(optimizer='adam'  ,loss='mse')

clf.fit(train,train_label,epochs=500)

eval_model=clf.evaluate(train,train_label)

y_pred=clf.predict(test)

y_pred=np.array(y_pred)
y_pred=np.round(y_pred)
y_pred=y_pred.astype(int)

 
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(test_label, y_pred)
 
 
accuracy=(cm[0][0]+cm[1][1]+cm[2][2]+cm[3][3]+cm[4][4])/len(test)

