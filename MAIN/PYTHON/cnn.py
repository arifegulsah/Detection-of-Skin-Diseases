# -*- coding: utf-8 -*-
"""
Created on Sat May  7 21:23:22 2022

@author: arife
"""

import pp
from keras.models import Sequential
from keras.layers import Dense,Dropout,BatchNormalization,Activation


##variables from pp file
Xtrain = pp.Xtrain
ytrain = pp.ytrain
X_test = pp.Xtest
y_test = pp.ytest


#modeli oluşturalım
model = Sequential()
#eğitim verisinde kaç tane stun yani model için girdi sayısı var onu alalım
n_cols = Xtrain.shape[1]
#model katmanlarını ekleyelim
model.add(Dense(16, input_shape=(n_cols,)))
model.add(Activation("relu"))
model.add(BatchNormalization())
model.add(Dropout(0.5))
model.add(Dense(9))
model.add(Activation("relu"))
model.add(BatchNormalization())
model.add(Dropout(0.5))
model.add(Dense(6))
model.add(Activation("relu"))
model.add(BatchNormalization())
model.add(Dropout(0.5))
model.add(Dense(3, activation='softmax'))
model.summary()



model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])


history =model.fit(Xtrain, 
                ytrain,
                validation_data=(X_test, y_test),
                batch_size=16, 
                shuffle=True,
                verbose=1,
                epochs=500)

