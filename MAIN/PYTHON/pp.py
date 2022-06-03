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


datam = pd.read_csv("../Test_Data/fotolar/acnedeneme.csv");
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

datam2 = pd.read_csv("../Test_Data/fotolar/denemem.csv")

X2 = get_image_data(datam2['Name'].values);
y2 = datam2.iloc[:,1];


from sklearn.model_selection import train_test_split

random_state = 42
# For reproducibility
#np.random.seed(random_state);

Xtrain, Xtest, ytrain, ytest = train_test_split(X2, y2, test_size=0.20, shuffle=True, random_state=random_state)
