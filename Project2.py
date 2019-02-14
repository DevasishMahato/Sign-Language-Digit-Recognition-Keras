import numpy as np
import cv2
import matplotlib.pyplot as plt
from keras.models import Sequential,load_model
from keras.layers import Conv2D,MaxPooling2D,Activation,Flatten,Dense,Dropout
import h5py
from keras import regularizers
from keras.callbacks import EarlyStopping
from keras.preprocessing.image import ImageDataGenerator
x = np.load('X.npy')
y = np.load('Y.npy')

x9=x[:204]
x9_train = x[:163]
x9_test = x[163:204]
#y9_train = y[1855:2021]
y9_train = y[1855:2018]
y9_test = y[2021:2062]


x0=x[204:409]
#x0_train = x[204:368]
x0_train = x[204:367]
x0_test = x[368:409]
y0_train = y[:163]
y0_test = y[163:204]

x7=x[409:615]
x7_train = x[409:574]
x7_test = x[574:615]
y7_train = y[1443:1608]
y7_test = y[1608:1649]

x6=x[615:822]
x6_train = x[615:781]
x6_test = x[781:822]
y6_train = y[1236:1402]
y6_test = y[1402:1443]

x1=x[822:1028]
#x1_train = x[822:987]
x1_train = x[822:986]
x1_test = x[987:1028]
y1_train = y[204:368]
y1_test = y[368:409]

x8=x[1028:1236]
#x8_train = x[1028:1194]
x8_train = x[1028:1193]
#x8_test = x[1194:1236]
x8_test = x[1194:1235]
y8_train = y[1649:1814]
y8_test = y[1814:1855]

x4=x[1236:1443]
#x4_train = x[1236:1402]
x4_train = x[1236:1401]
x4_test = x[1402:1443]
y4_train = y[822:987]
y4_test = y[987:1028]

x3=x[1443:1649]
x3_train = x[1443:1608]
x3_test = x[1608:1649]
#y3_train = y[615:781]
y3_train = y[615:780]
y3_test = y[781:822]

x2=x[1649:1855]
x2_train = x[1649:1814]
x2_test = x[1814:1855]
y2_train = y[409:574]
y2_test = y[574:615]

x5=x[1855:]
x5_train = x[1855:2021]
x5_test = x[2021:]
y5_train = y[1028:1194]
#y5_test = y[1194:1236]
y5_test = y[1194:1235]

#Training and Testing Sets
x_train = np.concatenate((x0_train,x1_train,x2_train,x3_train,x4_train,x5_train,x6_train,x7_train,x8_train,x9_train))
x_test = np.concatenate((x0_test,x1_test,x2_test,x3_test,x4_test,x5_test,x6_test,x7_test,x8_test,x9_test))
y_train = np.concatenate((y0_train,y1_train,y2_train,y3_train,y4_train,y5_train,y6_train,y7_train,y8_train,y9_train))
y_test = np.concatenate((y0_test,y1_test,y2_test,y3_test,y4_test,y5_test,y6_test,y7_test,y8_test,y9_test))

#width = 64 height = 64
x_train = x_train.reshape(x_train.shape[0],64,64,1)
x_test = x_test.reshape(x_test.shape[0],64,64,1)

#Training set generator
train_gen = ImageDataGenerator(shear_range=0.2,zoom_range=0.2,horizontal_flip='true',fill_mode='nearest')
train = train_gen.flow(x_train,y_train,batch_size=32)


#Network
model = Sequential()

#conv1
model.add(Conv2D(32,(3,3),input_shape=(64,64,1),padding='same',kernel_regularizer=regularizers.l2(0.0001)))
model.add(Activation('relu'))
model.add(MaxPooling2D((2,2),strides=(2,2),padding='same'))

#conv2
model.add(Conv2D(32,(3,3),padding='same',kernel_regularizer=regularizers.l2(0.0001)))
model.add(Activation('relu'))
model.add(MaxPooling2D((2,2),strides=(2,2)))

#conv3
model.add(Conv2D(64,(3,3),padding='same',kernel_regularizer=regularizers.l2(0.0001)))
model.add(Activation('relu'))
model.add(MaxPooling2D((2,2),strides=(2,2)))

#conv4
model.add(Conv2D(64,(3,3),padding='same',kernel_regularizer=regularizers.l2(0.0001)))
model.add(Activation('relu'))
model.add(MaxPooling2D((2,2),strides=(2,2)))

#flatten
model.add(Flatten())
model.add(Dense(128))
model.add(Activation('relu'))

#classifier
model.add(Dense(10))
model.add(Activation('softmax'))

#compile
model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
#fit
model.fit_generator(train,epochs=20,validation_data=(x_test,y_test),callbacks=[EarlyStopping(monitor='val_acc',patience=2)],verbose=2)

#save model
model.save('Project2.h5')

#predict
img = cv2.imread('5.jpg')
img = cv2.resize(img,(64,64))
img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
img.astype('float32')
img = img/255
img = np.reshape(img,(1,64,64,1))
#predict generator
pred_gen = ImageDataGenerator()
pred = pred_gen.flow(img,batch_size=1)
p = model.predict_generator(pred)
print(p.argmax())

model = load_model('Project2.h5')
#Video live stream
cam = cv2.VideoCapture(0)
for i in range (100):
    ret, img = cam.read()
    g_img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    g_img = cv2.resize(g_img,(64,64))
    g_img.astype('float32')
    g_img = g_img/255
    img = np.reshape(g_img,(1,64,64,1))
    cv2.imshow("Hand Sign Recognition",g_img)
    #predict generator
    pred_gen = ImageDataGenerator()
    pred = pred_gen.flow(img,batch_size=1)
    p = model.predict_generator(pred)
    print(p.argmax())
cv2.waitKey(0)
