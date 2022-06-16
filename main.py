
#from keras.optimizers import Adam
from keras.layers import BatchNormalization
from keras.layers.pooling import MaxPooling2D, AveragePooling2D
from keras.layers.core import Flatten, Dense
from keras.utils import np_utils
#from keras.initializers import glorot_uniform
from keras import backend as K
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
from matplotlib import pyplot as plt

from sklearn.model_selection import train_test_split

import cv2
import os
import glob
import pandas as pd

K.image_data_format() == 'channels_first'

folder_benign = "D:/myfolder/M.Tech/Semester 3/Dissertation/lung_colon_image_set/lung_image_sets/lung_n"

benign = []
for filename in os.listdir(folder_benign):
    img = cv2.imread(os.path.join(folder_benign, filename))
    if img is not None:
        img = cv2.resize(img, (256,256))
        benign.append(img)
 
benign_df = pd.DataFrame.from_records(benign)
           
folder_mal_aca = "D:/myfolder/M.Tech/Semester 3/Dissertation/lung_colon_image_set/lung_image_sets/lung_aca"

mal_aca = []
for filename in os.listdir(folder_mal_aca):
    img = cv2.imread(os.path.join(folder_mal_aca, filename))
    if img is not None:
        img = cv2.resize(img, (256,256))
        mal_aca.append(img)
        
mal_aca_df = pd.DataFrame.from_records(mal_aca)

folder_mal_scc = "D:/myfolder/M.Tech/Semester 3/Dissertation/lung_colon_image_set/lung_image_sets/lung_scc"

mal_scc = []
for filename in os.listdir(folder_mal_scc):
    img = cv2.imread(os.path.join(folder_mal_scc, filename))
    if img is not None:
        img = cv2.resize(img, (256,256))
        mal_scc.append(img)

mal_scc_df = pd.DataFrame.from_records(mal_scc)

print(f"Number of images for every class: BENIGN {benign_df.shape[0]}, ADENOCARCINOMAS {mal_aca_df.shape[0]}, SQUAMOS CELL CARCINOMAS {mal_scc_df.shape[0]}.")
print(f"Images shape: {benign[0].shape}.")

indices = [0, 40, 2300]

plt.figure(1, figsize=(15,5))
plt.grid(None)

for n, idx in enumerate(indices):
    plt.subplot(n+1, 3, 1) 
    plt.imshow(benign[idx])
    plt.title('benign')
    plt.subplot(n+1, 3, 2) 
    plt.imshow(mal_aca[idx])
    plt.title('malignant aca')
    plt.subplot(n+1, 3, 3) 
    plt.imshow(mal_scc[idx])
    plt.title('malignant scc')

plt.show()

samples = np.concatenate((benign, mal_aca, mal_scc))
labels = np.array(benign_df.shape[0] * [0] + mal_aca_df.shape[0] * [1] + mal_scc_df.shape[0] * [2])

print(f"Samples shape check: {samples.shape}.")
print(f"Labels shape check: {labels.shape}.")

indices = np.arange(samples.shape[0])
np.random.shuffle(indices)

samples = samples[indices]
labels = labels[indices]

# normalize pictures
samples = samples.astype('float32') / 255

X_train, X_test, y_train, y_test = train_test_split(samples, labels, test_size = 0.2)
X_val, X_test, y_val, y_test = train_test_split(X_test, y_test, test_size = 0.5)

Y_train = np_utils.to_categorical(y_train, 3)
Y_val = np_utils.to_categorical(y_val, 3)
Y_test = np_utils.to_categorical(y_test, 3)

print(Y_train[0])
print(f"Y_train shape after one hot encoding: {Y_val.shape}")
print(f"Y_train shape after one hot encoding: {X_val.shape}")

def base_model(input_shape=(256,256,3), classes=3):
    '''
    CNN base model with 4 hidden layers (3 conv and one fully connected)
    '''
    inputs = layers.Input(shape=input_shape)

    X = layers.Conv2D(32, 3, 3, activation='relu')(inputs)
    X = MaxPooling2D(pool_size=(2,2))(X)

    X = layers.Conv2D(64, 3, 3, activation='relu')(X)
    X = MaxPooling2D(pool_size=(2,2))(X)
    X = layers.Dropout(0.23)(X)

    X = layers.Conv2D(128, 3, 3, activation='relu', padding='same')(X)
    X = MaxPooling2D(pool_size=(2,2), padding='same')(X)
    X = layers.Dropout(0.23)(X)

    X = Flatten()(X)
    X = Dense(256, activation='relu',activity_regularizer=keras.regularizers.l2(0.1))(X)
    outputs = Dense(classes, activation='softmax')(X)

    model = keras.models.Model(inputs, outputs)
    
    return model

model = base_model(input_shape=(256,256,3), classes=3)
model.summary()


opt = keras.optimizers.Adam(lr=0.0001)
model.compile(optimizer=opt, loss='categorical_crossentropy', metrics='accuracy')

history = model.fit(X_train, Y_train, batch_size=64, epochs=200, validation_data=(X_val, Y_val))

keras.models.save_model(model,"D:/My Dissertation/Code/Results/Results - Lung CNN/V2 Results/V2")

plt.style.use('seaborn')
plt.figure(figsize=(16,7))

plt.subplot(1,2,1)
plt.plot(history.history['loss'])  
plt.plot(history.history['val_loss'])  
plt.title('Model loss', fontsize=16)  
plt.ylabel('Loss')  
plt.xlabel('Epoch')  
plt.legend(['train', 'val'])

plt.subplot(1,2,2)
plt.plot(history.history['accuracy'])  
plt.plot(history.history['val_accuracy'])  
plt.title('Model accuracy', fontsize=16)  
plt.ylabel('Accuracy')  
plt.xlabel('Epoch')  
plt.legend(['train', 'val'])

plt.show()

from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sn

y_pred_test = model.predict(X_test, verbose=1)
y_pred_train = model.predict(X_train, verbose=1)

# conversion to one hot encoding
#y_pred_test = (y_pred_test > 0.5).astype("int32")
#y_pred_train = (y_pred_train > 0.5).astype("int32")

# convert it to numerical classes
y_pred_test = np.argmax(y_pred_test, axis=1)
y_pred_train = np.argmax(y_pred_train, axis=1)
y_test_matric = np.argmax(Y_test, axis=1)
score_test = model.evaluate(X_test, Y_test)
print(f"Train accuracy: {history.history['accuracy'][-1]:.3f}")
print(f"Validation accuracy: {history.history['val_accuracy'][-1]:.3f}")
print(f"Test accuracy: {score_test[1]:.3f}")

label_names = ['benign', 'mal_aca', 'mal_scc']

confmat = confusion_matrix(y_test_matric, y_pred_test)
mat = classification_report(y_test_matric, y_pred_test, target_names=label_names)
print(mat)
sn.heatmap(confmat, annot=True, cmap='Blues', cbar=False, 
           xticklabels=label_names, yticklabels=label_names, fmt = 'g')

plt.title("Confusion Matrix")
plt.xlabel("Actual Label")
plt.ylabel("Predicted Label")
plt.show()