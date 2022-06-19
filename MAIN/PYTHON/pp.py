# -*- coding: utf-8 -*-
"""
Created on Thu Mar 17 18:26:21 2022

@author: arife
"""
import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import imageio

#imageları numpy arraye çeviren fonksiyon
def get_image_data(files):
    '''Returns np.ndarray of images read from the image data directory'''
    IMAGE_FILE_ROOT = '../Test_Data/photos/' 
    return np.asanyarray([imageio.imread("{}{}".format(IMAGE_FILE_ROOT, file)) for file in files])

#arrayi verilem imagı çizdiren fonskiyon
def show_image(image, ax = plt, title = None, show_size = False):
    '''Plots a given np.array image'''
    ax.imshow(image)
    if title:
        if ax == plt:
            plt.title(title)
        else:
            ax.set_title(title)
    if not show_size:
        ax.tick_params(bottom = False, left = False, labelbottom = False, labelleft = False)


datam = pd.read_csv("../Test_Data/test.csv");
print(datam.head(2))




X = get_image_data(datam['Name'].values);
y = datam.iloc[:,1];

print(y.unique());


"""
#birden fazla imageı plotluyor
def show_images(images, titles = None, show_size = False):
    '''Plots many images from the given list of np.array images'''
    cols = 4
    f, ax = plt.subplots(nrows=int(np.ceil(len(images)/cols)),ncols=cols, figsize=(14,5))
    ax = ax.flatten()
    for i, image in enumerate(images):
        if titles:
            show_image(image, ax = ax[i], title = titles[i], show_size = show_size)
        else:
            show_image(image, ax = ax[i], title = None, show_size = show_size)
    plt.show()
"""

#fotoğrafların resize edilmesi, en uygun widht height hesaplanmalı

#imageların tek tek enlerini ve boylarını array olarak döndüren fonksiyon
def get_images_wh(images):
    '''Returns a tuple of lists, representing the widths and heights of the give images, respectively.'''
    widths = []
    heights = []
    for image in images:
        h, w, rbg = image.shape
        widths.append(w)
        heights.append(h)
    return (widths, heights)

#verilen arrayin ortalamasını bulan fonksiyon
#bu fonskiyon ile enlerin ve boyların arraylerini parametre olarak vereceğiz ve ortalamalarını bulacağız
#bu sayede ideal witdh ve ideal height dönmüş olacak
def get_best_average(dist, cutoff = .5):
    '''Returns an integer of the average from the given distribution above the cutoff.'''
    # requires single peak normal-like distribution
    hist, bin_edges = np.histogram(dist, bins = 25);
    total_hist = sum(hist)
    
    # associating proportion of hist with bin_edges
    hist_edges = [(vals[0]/total_hist,vals[1]) for vals in zip(hist, bin_edges)]
    
    # sorting by proportions (assumes normal-like dist such that high freq. bins are close together)
    hist_edges.sort(key = lambda x: x[0])
    lefts = []
    
    
    # add highest freq. bins to list up to cutoff % of total
    while cutoff > 0:
        vals = hist_edges.pop()
        cutoff -= vals[0]
        lefts.append(vals[1])
   
    # determining leftmost and rightmost range, then returning average
    ##diff = np.abs(np.diff(lefts)[0]) # same diff b/c of bins
    leftmost = min(lefts)
    rightmost = max(lefts) # + diff
    return int(np.round(np.mean([rightmost,leftmost])))
    

wh = get_images_wh(X)
wh

size = 18
plt.title("Widths of skin diseases images", fontsize = size * 4/3, pad = size/2)
plt.ylabel("Frequency", size = size)
plt.xlabel("width (pixels)", size = size)
plt.hist(wh[0], bins = 25);


size = 18
plt.title("Heights of skin diseases images", fontsize = size * 4/3, pad = size/2)
plt.ylabel("Frequency", size = size)
plt.xlabel("width (pixels)", size = size)
plt.hist(wh[1], bins = 25);

wh[0]
wh[1]
show_image(X[5])


IDEAL_WIDTH, IDEAL_HEIGHT = get_best_average(wh[0]), get_best_average(wh[1]);
IDEAL_WIDTH, IDEAL_HEIGHT

""" İDEAL GENİŞLİK VE UZUNLUKLARI BULDUĞUMUZA GÖRE ŞİMDİ DE BÜTÜN VERİ SETİNİ BU BOYUTLARDA RESIZE ETMELİYİZ """

import os
import os.path
from PIL import Image

f = r'../Test_Data/photos'
for file in os.listdir(f):
    f_img = f+"/"+file
    img = Image.open(f_img)
    img = img.resize((151, 128))
    #img = img.resize((IDEAL_WIDTH, IDEAL_HEIGHT))
    img = img.convert('RGB')
    img.save(f_img)


print(X[0].shape)
print(X[0])

datam2 = pd.read_csv("../Test_Data/test.csv")

X2 = get_image_data(datam2['Name'].values);
y2 = datam2.iloc[:,1];


from sklearn import preprocessing
le = preprocessing.LabelEncoder()
y2 = le.fit_transform(y2)


print(X2[0].shape)

from sklearn.model_selection import train_test_split

random_state = 42
# For reproducibility
#np.random.seed(random_state);

Xtrain, Xtest, ytrain, ytest = train_test_split(X2, y2, test_size=0.20, shuffle=True, random_state=random_state)

#Xtrain = np.delete(Xtrain, 0, 1)

#Xtrain = pd.DataFrame(Xtrain)
"""
from sklearn import preprocessing
le = preprocessing.LabelEncoder()
ytrain = le.fit_transform(ytrain)
ytest = le.fit_transform(ytest)"""








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




model2 = Sequential()
model2.add(Conv2D(32, kernel_size = (3, 3), activation='relu', input_shape=(128, 151, 3)))
model2.add(MaxPooling2D(pool_size=(2,2)))
model2.add(BatchNormalization())
model2.add(Conv2D(64, kernel_size=(3,3), activation='relu'))
model2.add(MaxPooling2D(pool_size=(2,2)))
model2.add(BatchNormalization())
model2.add(Conv2D(96, kernel_size=(3,3), activation='relu'))
model2.add(MaxPooling2D(pool_size=(2,2)))
model2.add(BatchNormalization())
model2.add(Conv2D(96, kernel_size=(3,3), activation='relu'))
model2.add(MaxPooling2D(pool_size=(2,2)))
model2.add(BatchNormalization())
model2.add(Conv2D(64, kernel_size=(3,3), activation='relu'))
model2.add(MaxPooling2D(pool_size=(2,2)))
model2.add(BatchNormalization())
model2.add(Dropout(0.2))
model2.add(Flatten())
model2.add(Dense(256, activation='relu'))
model2.add(Dropout(0.2))
model2.add(Dense(128, activation='relu'))
model2.add(Dense(4, activation = 'softmax'))


model2.compile(loss='binary_crossentropy', optimizer='adam', metrics = ['accuracy'])

model2.fit(Xtrain,ytrain, batch_size = 50, epochs = 2, verbose = 1)











from sklearn.model_selection import GridSearchCV

# define the grid search parameters
batch_size = [10, 20, 40, 60, 80, 100]
epochs = [10, 50, 100]
optimizer = ['SGD', 'RMSprop', 'Adagrad', 'Adadelta', 'Adam', 'Adamax', 'Nadam']
learn_rate = [0.001, 0.01, 0.1, 0.2, 0.3]
momentum = [0.0, 0.2, 0.4, 0.6, 0.8, 0.9]
init_mode = ['uniform', 'lecun_uniform', 'normal', 'zero', 'glorot_normal', 'glorot_uniform', 'he_normal', 'he_uniform']
activation = ['softmax', 'softplus', 'softsign', 'relu', 'tanh', 'sigmoid', 'hard_sigmoid', 'linear']
weight_constraint = [1, 2, 3, 4, 5]
dropout_rate = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
neurons = [1, 5, 10, 15, 20, 25, 30]

param_grid = dict(batch_size=batch_size, epochs=epochs, optimizer=optimizer,init_mode=init_mode,dropout_rate=dropout_rate,neurons=neurons)
grid = GridSearchCV(estimator=model, param_grid=param_grid, n_jobs=-1, cv=3)
grid_result = grid.fit(Xtrain, ytrain)

grid.score(Xtrain, ytrain)



# summarize results
print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
means = grid_result.cv_results_['mean_test_score']
stds = grid_result.cv_results_['std_test_score']
params = grid_result.cv_results_['params']
for mean, stdev, param in zip(means, stds, params):
    print("%f (%f) with: %r" % (mean, stdev, param))
    
    




plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')
plt.show()



plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')
plt.show()




"""Test loss: 7.666234970092773
Test accuracy: 0.5263158082962036
optimizer='rmsprop',

,SGD(lr=0.000001
Test loss: 2.5285251140594482
Test accuracy: 0.31578946113586426
"""



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




y_test_arg=np.argmax(ytest,axis=1)
Y_pred = np.argmax(model.predict(Xtest),axis=1)
print('Confusion Matrix')
print(confusion_matrix(y_test_arg, Y_pred))


from sklearn.metrics import classification_report



print(classification_report(ytest, y_pred))









