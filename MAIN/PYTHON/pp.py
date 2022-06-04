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
    IMAGE_FILE_ROOT = '../Test_Data/fotolar/' 
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


datam = pd.read_csv("../Test_Data/acnedeneme.csv");
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

f = r'../Test_Data/fotolar'
for file in os.listdir(f):
    f_img = f+"/"+file
    img = Image.open(f_img)
    img = img.resize((IDEAL_WIDTH, IDEAL_HEIGHT))
    img.save(f_img)


print(X[0].shape)
print(X[0])

datam2 = pd.read_csv("../Test_Data/denemem.csv")

X2 = get_image_data(datam2['Name'].values);
y2 = datam2.iloc[:,1];


from sklearn.model_selection import train_test_split

random_state = 42
# For reproducibility
#np.random.seed(random_state);

Xtrain, Xtest, ytrain, ytest = train_test_split(X2, y2, test_size=0.20, shuffle=True, random_state=random_state)

#Xtrain = np.delete(Xtrain, 0, 1)

#Xtrain = pd.DataFrame(Xtrain)

from sklearn import preprocessing
le = preprocessing.LabelEncoder()
ytrain = le.fit_transform(ytrain)




ytest = le.fit_transform(ytest)








from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,Dropout,BatchNormalization,Activation

from tensorflow.keras.layers import Conv2D, MaxPool2D, Flatten


model = Sequential()
model.add(Conv2D(32,3,padding="same", activation="relu", input_shape=((461, 703, 3))))
model.add(MaxPool2D())

model.add(Conv2D(32, 3, padding="same", activation="relu"))
model.add(MaxPool2D())

model.add(Conv2D(64, 3, padding="same", activation="relu"))
model.add(MaxPool2D())
model.add(Dropout(0.4))

model.add(Flatten())
model.add(Dense(128,activation="relu"))
model.add(Dense(2, activation="softmax"))

model.summary()

print(Xtrain[0].shape)


model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])



history = model.fit(Xtrain, 
                ytrain,
                batch_size=16, 
                #validation_data=(Xtest, ytest),
                shuffle=True,
                verbose=1,
                epochs=16)

score = model.evaluate(Xtest, ytest, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])





y_pred = model.predict(Xtest)
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










"""


def normalize(image):
    '''
    Returns the given np.array image rescaled and normalized to be between -.5 and .5
    
    Source: https://www.jeremyjordan.me/batch-normalization/
    '''
    return (image/255. - 0.5)


from tensorflow import keras 

from skimage.transform import rescale, resize, rotate
from skimage.color import rgb2gray

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,Dropout,BatchNormalization,Activation
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D

#from tensorflow.keras.utils import np_utils

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam, RMSprop, SGD
from tensorflow.keras import callbacks
from tensorflow.keras.models import load_model
from tensorflow.keras import models



def create_datagens(
    data, datagen_params, 
    target_shape, batch_size, x_col=X, y_col=y, IMAGE_FILE_ROOT = '../data/bee_imgs/', 
    random_state = None, preprocessing_function = None):
        '''
        Appropriately creates and returns two ImageDataGenerator objects - one for training and one for testing.
        
        ImageDataGenerator objects are responsible for handling image data during model training, by pulling the data
        directly from the image data directory, resizing the image, and applying the appropriate transformations.
        
        The testing ImageDataGenerator object does not apply any transformations.
        '''
        data[y_col] = data[y_col].astype(str) # coercion needed for datagen
        # train/test split
        train, test = train_test_split(
            data, 
            test_size = 1/3, 
            stratify = data.iloc[:,-1], # assumed last column is target variable
            random_state = random_state
            )
        
        # training ImageDataGenerator
        datagen = ImageDataGenerator(
            horizontal_flip  = datagen_params.get("horizontal_flip") or False, 
            vertical_flip    = datagen_params.get("vertical_flip") or False, 
            rotation_range   = datagen_params.get("rotation_range") or False,
            brightness_range = datagen_params.get("brightness_range"),
            preprocessing_function = preprocessing_function
        )

        datagen_iter_train = datagen.flow_from_dataframe(
            train, 
            directory   = IMAGE_FILE_ROOT, 
            x_col       = x_col,
            y_col       = y_col,
            target_size = target_shape, 
            color_mode  = 'rgb', 
            class_mode  = 'binary', 
            batch_size  = batch_size, 
            shuffle     = True,
            seed        = random_state
        )

        # testing ImageDataGenerator
        datagen_test = ImageDataGenerator(preprocessing_function = preprocessing_function)

        datagen_iter_test = datagen_test.flow_from_dataframe(
            test, 
            directory   = IMAGE_FILE_ROOT, 
            x_col       = x_col,
            y_col       = y_col,
            target_size = target_shape, 
            color_mode  = 'rgb', 
            class_mode  = 'binary', 
            batch_size  = 1, 
            shuffle     = False
        )
        
        return datagen_iter_train, datagen_iter_test











def permutate_params(grid_params):
    '''Returns a list of all combinations of unique parameters from the given dictionary'''
    out = [{}]
    
    # loop through each key/val pair
    for param_name, param_list in grid_params.items():
        # shortcircut - no need to permute single items
        if len(param_list) == 1:
            for item in out:
                item[param_name] = param_list[0]
        else:
            temp_out = []
            # for each item in the param, clone entire growing list and add param to each
            for param_val in param_list:
                for item in out:
                    cloned_item = item.copy()
                    cloned_item[param_name] = param_val
                    temp_out.append(cloned_item)
            out = temp_out
    return out





def build_model_from_datagen(
    params = dict(), 
    input_shape = (), 
    datagen_iter_train = None, 
    datagen_iter_val = None, 
    optimizer = "adam",
    file_name = None):
    '''Returns a fitted convolutional neural network with the given parameters and data.'''
    kernel_size = 3
    dropout = .5
    activation_func = "relu"

    conv__filters_1 = params.get('conv__filters_1') or 32
    conv__filters_2 = params.get('conv__filters_2') or 16
    conv__filters_3 = params.get('conv__filters_3') or 32
    density_units_1 = params.get('density_units_1') or 32
    density_units_2 = params.get('density_units_2') or 32
    epochs          = params.get('epochs') or 8
    
    # instantiating model
    model = Sequential([
        # Conv layer #1
        Conv2D(
            filters = conv__filters_1, 
            kernel_size = kernel_size + 4, 
            activation  = activation_func, 
            input_shape = input_shape, #input layer
            padding     = "same"
        ),
        Conv2D(filters = conv__filters_1, kernel_size = kernel_size + 4, activation = activation_func, padding = "same"),
        MaxPooling2D(pool_size=(2,2)),
        Dropout(dropout/2),

        # Conv layer #2
        Conv2D(filters = conv__filters_2, kernel_size = kernel_size + 2, activation=activation_func, padding = "same"),
        Conv2D(filters = conv__filters_2, kernel_size = kernel_size + 2, activation = activation_func, padding = "same"),
        MaxPooling2D(pool_size=(2,2)),
        Dropout(dropout/2),

        # Conv layer #3
        Conv2D(filters = conv__filters_3, kernel_size = kernel_size, activation=activation_func, padding = "same"),
        Conv2D(filters = conv__filters_3, kernel_size = kernel_size, activation = activation_func, padding = "same"),
        MaxPooling2D(pool_size=(2,2)),
        Dropout(dropout/2),

        # Dense layer #1
        Flatten(),
        Dense(density_units_1, activation=activation_func),
        Dropout(dropout),
        
        # Dense layer #2
        Dense(density_units_2, activation=activation_func),
        Dropout(dropout),
        
        # Output layer
        Dense(1, activation='sigmoid')
    ])
    
    # compiling model
    model.compile(
        loss      = 'binary_crossentropy',
        optimizer = optimizer,
        metrics   = ['binary_accuracy']
    )
    
    # fitting model w/ImageDataGenerator
    STEP_SIZE_TRAIN= np.ceil(datagen_iter_train.n/datagen_iter_train.batch_size)
    STEP_SIZE_VALID= np.ceil(datagen_iter_val.n/datagen_iter_val.batch_size)

    # NOTE: the best model is saved to disk via callbacks, and is a retrievable file
    history = model.fit_generator(
        generator           = datagen_iter_train,
        steps_per_epoch     = STEP_SIZE_TRAIN,
        validation_data     = datagen_iter_val,
        validation_steps    = STEP_SIZE_VALID,
        epochs              = epochs,
        callbacks           = [callbacks.ModelCheckpoint(file_name, save_best_only=True, mode='auto', period=1)]
    )
    
    return (model, history)






def gridSearchCNN(
    datagens,
    grid_params, 
    file_name,
    optimizer = "adam",
    random_state = None,
):
    '''
    Iteratively discovers and then returns an optimized convolutional neural network with the given grid_params
    
    Much of the code related to datagen came from:
    https://medium.com/@vijayabhaskar96/tutorial-image-classification-with-keras-flow-from-directory-and-generators-95f75ebe5720
    '''
    # list of all parameter combinations
    all_params = permutate_params(grid_params) 
    
    # establishing variables
    best_model   = None
    best_score   = 0.0 # no accuracy to start
    best_params  = None
    best_history = None
    test_scores  = None
    train_scores = None
    
    datagen_iter_train, datagen_iter_test = datagens
    
    # for each permuted parameter, try fitting a model (NOTE: the best model is saved to disk with file_name)
    for params in all_params:
        model, history = build_model_from_datagen(
            params, 
            input_shape        = datagen_iter_train.image_shape,
            datagen_iter_train = datagen_iter_train,
            datagen_iter_val   = datagen_iter_test,
            optimizer          = optimizer,
            file_name          = file_name
        )

        acc = max(history.history["val_binary_accuracy"])
        
        # only keeping best
        if acc > best_score:
            print("***Good Accurary found: {:.2%}***".format(acc))
            best_score   = acc
            test_scores  = history.history["val_binary_accuracy"]
            train_scores = history.history["binary_accuracy"]
            best_model   = model
            best_params  = params
            best_history = history
    
    # returns metadata of results (NOTE: retrieving best model from hard disk)
    return {
        "best_model"   : load_model(file_name),
        "best_score"   : best_score,
        "best_params"  : best_params,
        "best_history" : best_history,
        "test_scores"  : test_scores,
        "train_scores" : train_scores
    }
    


from skimage.transform import rescale, resize, rotate
from skimage.color import rgb2gray
from sklearn.metrics import confusion_matrix, auc, accuracy_score

def conf_matrix_stats(y_test, preds):
    ''' Return key confusion matrix metrics given true and predicted values'''
    cm = confusion_matrix(y_test, preds)
    TP, FP, FN, TN, = cm[1,1], cm[0,1], cm[1,0], cm[0,0]
    total = (TP + FP + FN + TN)
    acc = (TP + TN ) / total
    miss = 1 - acc
    sens = TP / (TP + FN)
    spec = TN / (TN + FP)
    prec = TP / (TP + FP)
    return {"accuracy": acc, "miss_rate": miss, "sensitivity": sens, "specification": spec, "precision": prec}




MODEL_PATH = "../models"
model_name = "original"
stored_model_path = f"{MODEL_PATH}/{model_name}_model.p"

datagen_params = dict()

datagens = create_datagens(
    Xtrain, 
    datagen_params         = datagen_params,
    batch_size             = 64, # hyperparameter
    target_shape           = (IDEAL_WIDTH, IDEAL_HEIGHT), 
    preprocessing_function = normalize,
    random_state           = random_state
)
"""