import os
import librosa
import numpy as np
import cv2

os.getcwd()

data_list = os.listdir('./data')

for i, dl in enumerate(data_list):
    if i % 100 == 0:
        print(i)
    y, sample_rate = librosa.load('./data/{}'.format(dl), sr=44100)
    if len(y) < 400000:
        continue
    mel = librosa.feature.melspectrogram(y=y, sr=44100, n_mels=299)
    db_mel_raw = librosa.power_to_db(mel,ref=np.max)
    z = np.zeros((299,897-db_mel_raw.shape[1]))
    db_mel = np.concatenate((db_mel_raw, z), axis=1)
    db_mel.shape
    db_mel1 = db_mel[:,:299]
    db_mel2 = db_mel[:,299:299*2]
    db_mel3 = db_mel[:,299*2:]
    db_mel_result = np.dstack((db_mel1, db_mel2, db_mel3))
    np.save("./img_data/{}".format(dl.split('.')[0]), db_mel_result)




bb = np.load('./img_data/6xL2uWTpHVk.png')
plt.imshow(bb[:,:,0])
aa.shape
aa[0].shape
db_mel_raw.min()
db_mel_result[:,:,0]-db_mel_raw.min()
(db_mel_result[:,:,0]-db_mel_raw.min()).astype(np.uint8)
plt.imshow((db_mel_result[:,:,0]-db_mel_raw.min()).astype(np.uint8))
plt.imshow(aa[:,:,0])

plt.imshow(aa[:,:,1])

plt.imshow(aa[:,:,2])
