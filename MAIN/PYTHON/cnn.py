# -*- coding: utf-8 -*-
"""
Created on Sat May  7 21:23:22 2022

@author: arife
"""


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,Dropout,BatchNormalization,Activation

from tensorflow.keras.layers import Conv2D, MaxPool2D, Flatten, MaxPooling2D
from tensorflow.keras.optimizers import SGD


model = Sequential()
model.add(Conv2D(256,(3,3),padding="same", activation="relu", input_shape=((128, 151, 3))))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(BatchNormalization())

#model.add(Dropout(0.2))

model.add(Conv2D(128, (3,3), padding="same", activation="relu"))
model.add(Conv2D(128, (3,3), padding="same", activation="relu"))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(BatchNormalization())

#model.add(Dropout(0.2))

model.add(Conv2D(256, (3,3), padding="same", activation="relu"))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(BatchNormalization())
model.add(Dropout(0.2))

#model.add(Dropout(0.4))

model.add(Flatten())
model.add(Dense(256,activation="relu"))
#model.add(Dropout(0.2))
model.add(Dense(128,activation="relu"))
#model.add(Dropout(0.4))
#model.add(Dense(64,activation="relu"))
model.add(Dropout(0.3))

model.add(Dense(4, activation="softmax"))

model.summary()

print(Xtrain[1].shape)


model.compile(optimizer=SGD(lr=0.000001, momentum=0.09), loss='binary_crossentropy', metrics=['accuracy'])

#model.compile(loss='binary_crossentropy', optimizer='adam', metrics = ['accuracy'])


#Xtrain = np.asarray(Xtrain).astype('float32')
#ytrain = np.asarray(ytrain)


history = model.fit(Xtrain, 
                ytrain,
                batch_size=16, 
                #validation_data=(Xtest, ytest),
                shuffle=True,
                verbose=1,
                epochs=250,
                use_multiprocessing = True)

pd.DataFrame(history.history).plot(figsize=(8,5))
plt.show()




score = model.evaluate(Xtest, ytest, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])




y_pred = model.predict(Xtest)


maxpredicts = []
for element in y_pred:
    temp = np.argmax(element)
    maxpredicts.append(temp)
    
y_pred = np.array(maxpredicts)    


np.argmax(y_pred[7])

print(y_pred)



## Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(ytest, y_pred)



from sklearn.metrics import classification_report
print(classification_report(ytest, y_pred))

