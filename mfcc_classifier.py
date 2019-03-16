import warnings
warnings.filterwarnings("ignore")
import os
import pandas as pd
import librosa
import librosa.display
import glob
import tensorflow as tf
import keras
import os
import fnmatch
from os.path import isfile
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten, LSTM
from keras.layers import Convolution2D, MaxPooling2D
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping,ReduceLROnPlateau,ModelCheckpoint,TensorBoard,ProgbarLogger
from keras.utils import np_utils
from sklearn import metrics
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from sklearn.preprocessing import LabelEncoder
import itertools


INPUT_DIR = "heartbeat-sounds/"
# 16 KHz
SAMPLE_RATE = 16000
seed = 1000

# seconds
MAX_SOUND_CLIP_DURATION = 10
CLASSES = ['murmur', 'normal']
MAX_PATIENT = 12
MAX_EPOCHS = 50
BATCH_SIZE = 10


def load_audio_data(folder, file_names, duration=3, sr=16000):
    input_length = sr*duration
    # function to load files and extract features
    # file_names = glob.glob(os.path.join(folder, '*.wav'))
    data = []
    for file_name in file_names:
        try:
            sound_file = folder+file_name
            #print ("load file ", sound_file)
            # use kaiser_fast technique for faster extraction
            X, sr = librosa.load(sound_file, sr=sr, duration=duration, res_type='kaiser_fast')
            dur = librosa.get_duration(y=X, sr=sr)
            # pad audio file same duration
            if round(dur) < duration:
                #print ("fixing audio length :", file_name)
                y = librosa.util.fix_length(X, input_length)

            # normalized raw audio
            # y = audio_norm(y)
            # extract normalized mfcc feature from data
            mfccs = np.mean(librosa.feature.mfcc(y=X, sr=sr, n_mfcc=40).T, axis=0)
        except Exception as e:
            print("Error encountered while parsing file: ", file_name)
        feature = np.array(mfccs).reshape([-1, 1])
        data.append(feature)
    return data


def load_data_and_split():

    if isfile('x_data.npy')==0 and isfile('y_data.npy')==0:
        A_folder = INPUT_DIR + 'set_a/'

        A_normal_files = fnmatch.filter(os.listdir(INPUT_DIR + 'set_a'), 'normal*.wav')
        A_normal_sounds = load_audio_data(folder=A_folder, file_names=A_normal_files, duration=MAX_SOUND_CLIP_DURATION)
        A_normal_labels = [1 for items in A_normal_sounds]


        A_murmur_files = fnmatch.filter(os.listdir(INPUT_DIR + 'set_a'), 'murmur*.wav')
        A_murmur_sounds = load_audio_data(folder=A_folder, file_names=A_murmur_files, duration=MAX_SOUND_CLIP_DURATION)
        A_murmur_labels = [0 for items in A_murmur_files]

        # test files
        A_unlabelledtest_files = fnmatch.filter(os.listdir(INPUT_DIR + 'set_a'), 'Aunlabelledtest*.wav')
        A_unlabelledtest_sounds = load_audio_data(folder=A_folder, file_names=A_unlabelledtest_files,
                                                 duration=MAX_SOUND_CLIP_DURATION)
        A_unlabelledtest_labels = [-1 for items in A_unlabelledtest_sounds]

        print("loaded dataset-a")

        B_folder = INPUT_DIR + 'set_b/'

        B_normal_files = fnmatch.filter(os.listdir(INPUT_DIR + 'set_b'), 'normal*.wav')  # include noisy files
        B_normal_sounds = load_audio_data(folder=B_folder, file_names=B_normal_files, duration=MAX_SOUND_CLIP_DURATION)
        B_normal_labels = [1 for items in B_normal_sounds]

        B_murmur_files = fnmatch.filter(os.listdir(INPUT_DIR + 'set_b'), 'murmur*.wav')  # include noisy files
        B_murmur_sounds = load_audio_data(folder=B_folder, file_names=B_murmur_files, duration=MAX_SOUND_CLIP_DURATION)
        B_murmur_labels = [0 for items in B_murmur_files]

        # test files
        B_unlabelledtest_files = fnmatch.filter(os.listdir(INPUT_DIR + 'set_b'), 'Bunlabelledtest*.wav')
        B_unlabelledtest_sounds = load_audio_data(folder=B_folder, file_names=B_unlabelledtest_files,
                                                 duration=MAX_SOUND_CLIP_DURATION)
        B_unlabelledtest_labels = [-1 for items in B_unlabelledtest_sounds]
        print("loaded dataset-b")

        x_data = np.concatenate((A_normal_sounds, A_murmur_sounds,
                                 B_normal_sounds, B_murmur_sounds))

        y_data = np.concatenate((A_normal_labels, A_murmur_labels,
                                 B_normal_labels, B_murmur_labels))

        np.save('x_data.npy', x_data)
        np.save('y_data.npy', y_data)
        print('Saved the new x_data and y_data')

    else:
        x_data = np.load('x_data.npy')
        y_data = np.load('y_data.npy')
        print('Loaded x_data and y_data from the existing numpy arrays')

    # Add randome_state = seed if the result has to be produced again
    x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, train_size=0.8, random_state=seed, shuffle=True)

    y_train = np.array(keras.utils.to_categorical(y_train, len(CLASSES)))
    y_test = np.array(keras.utils.to_categorical(y_test, len(CLASSES)))

    print("training data shape: ", x_train.shape)
    print("training label shape: ", y_train.shape)
    print("test data shape: ", x_test.shape)
    print("test label shape: ", y_test.shape)

    return x_train, y_train, x_test, y_test


def build_model():
    model = Sequential()
    model.add(LSTM(units=32, dropout=0.05, recurrent_dropout=0.20, return_sequences=True, input_shape=(40, 1)))
    model.add(LSTM(units=16, dropout=0.05, recurrent_dropout=0.20, return_sequences=False))
    model.add(Dense(len(CLASSES), activation='softmax'))
    model.summary()
    return model


def train_net(model, x_train, y_train):

    best_model_file = "./best_model_trained.hdf5"
    # callbacks
    # removed EarlyStopping(patience=MAX_PATIENT)
    callback = [ReduceLROnPlateau(patience=MAX_PATIENT, verbose=1),
                ModelCheckpoint(filepath=best_model_file, monitor='loss', verbose=1, save_best_only=True)]

    model.compile(loss='categorical_crossentropy', optimizer='Adamax', metrics=['acc', 'mse', 'mae', 'mape', 'cosine'])

    print("training started..... please wait.")
    # training

    history = model.fit(x_train, y_train,
                        batch_size=BATCH_SIZE,
                        epochs=MAX_EPOCHS,
                        verbose=0,
                        shuffle=False,
                        callbacks=callback)

    print("training finished!")
    return model


def evaluate_score(model, x_train, y_train, x_test, y_test):
    score = model.evaluate(x_train, y_train, verbose=1)
    print("model train data score       : ", round(score[1] * 100), "%")

    # Score prints the loss along with the metrics as mentioned in model.compile
    score = model.evaluate(x_test, y_test, verbose=1)
    #print(len(score), score)
    print("model test data score        : ", round(score[1] * 100), "%")


def main():

    print('Code Running...')
    x_train, y_train, x_test, y_test = load_data_and_split()
    model = build_model()
    model = train_net(model, x_train, y_train)
    evaluate_score(model, x_train, y_train, x_test, y_test)


if __name__ == "__main__":
	main()