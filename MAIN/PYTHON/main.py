
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