## server.py

import socket
import numpy as np
#from model.tmp_model import Model
from sklearn.preprocessing import MinMaxScaler

import keras
import keras.models as KM
import keras.layers as KL
import keras.backend as K
from keras.applications.inception_v3 import InceptionV3
from keras.layers import Dense, GlobalAveragePooling2D
from keras.models import Model, load_model
from keras.utils.multi_gpu_utils import multi_gpu_model
import librosa
import _thread


scaler = MinMaxScaler(feature_range=(-1, 1))

#################################################################################

def trans_data_mel(array_data) :
    array_data = array_data.astype(np.float32)
    mel = librosa.feature.melspectrogram(y=array_data, sr=44100, n_mels=299)
    db_mel_raw = librosa.power_to_db(mel,ref=np.max)
    z = np.zeros((299,897-db_mel_raw.shape[1]))
    db_mel = np.concatenate((db_mel_raw, z), axis=1)
    db_mel1 = db_mel[:,:299]
    db_mel2 = db_mel[:,299:299*2]
    db_mel3 = db_mel[:,299*2:]
    db_mel_result = np.dstack((db_mel1, db_mel2, db_mel3))
    return db_mel_result


def extract_output(prob_array, threshold=0.15, k=3, flag=True) :
    '''
    labels = ["Air conditioning", "Vehicle horn", "Drill", "Idling", "Jackhammer", "Ambulance", "others"]
    threshold : this is probability threshold - default : 30%
    top_k : top_k default is top3
    '''
    labels = np.array(["Air conditioning", "Vehicle horn", "Drill", "Engine", "Jackhammer", "Siren", "others"])
    if flag == True :
        #thresHold
        index = np.where(prob_array >= threshold)
        human_string = labels[index[0]]
        return list(human_string)
    else :
        #Top_k
        num_top = k
        top_k = prob_array.argsort()[-num_top:][::-1] # array(5,4,6)
        human_string = labels[top_k]

        return list(human_string)

#################################################################################

class CNN_Model(object):
    def __init__(self):
        self.optimizer = keras.optimizers.Adam(lr=0.001)
        self.model = KM.load_model('./model/0035-1.4992.hdf5')
        self.model.compile(loss='categorical_crossentropy',
                      optimizer=self.optimizer,
                      metrics=['accuracy'])


    def run_prediction(self, x_data):
        pred = self.model.predict(x_data)
        print(pred)
        result = extract_output(pred[0],threshold=0.2, k=5, flag=True)
        return result

INSTANCE_model = CNN_Model()
print('model instance is created!')


def predict_func(x_data):
    return INSTANCE_model.model.predict(x_data)

def run_model(x_data):
    input_feature = trans_data_mel(x_data)

    # reshape
    input_img = np.reshape(input_feature, (1,) + input_feature.shape)# shape = (1,229,229,3)

    # run model prediction
    y_data = INSTANCE_model.run_prediction(input_img)

    return y_data

if __name__ == '__main__':
    run_server()
