import keras
from keras.applications.inception_v3 import InceptionV3
#from keras.applications.inception_resnet_v2 import InceptionResNetV2
from keras.models import Model, load_model
from keras.layers import Dense, GlobalAveragePooling2D
from keras.optimizers import Adam
from keras.utils import multi_gpu_model
from keras import backend as K

import pandas as pd
import numpy as np
import os
import math
from sklearn.utils import shuffle

os.chdir('../demo/')

################################################################################
################################ HYPER PRAMETER ################################
################################################################################

nb_classes = 7  # number of classes

# input image size
total_img = 24930
batch_size = 32  # batch size
total_epochs = 1000
train_rate = 0.8
learning_rate = 0.001

################################################################################


################################################################################
################################ DIVIDE DATASET ################################
################################################################################

folder_path = "./img_data"
total_data_df = pd.read_csv("./classification/total_data_df.csv")
class_list = ["Air conditioning", "Vehicle horn", "Drill", "Idling", "Jackhammer", "Ambulance"]
total_data_df['classes'] = total_data_df['classes'].apply(lambda x : [int(num) for num in x[1:-1].split(',')])


result = [] # data_name, data_path
for img_name in os.listdir(folder_path):
	data_path = os.path.join(folder_path, img_name)
	data_name = img_name.split(".")[0]
	result.append([data_name, data_path])

# join
exist_file = pd.DataFrame(result, columns=["#YTID", "data_path"])
temp = pd.merge(exist_file, total_data_df)
data_pair = temp.loc[:,['data_path', 'classes']] #
train_data = data_pair.values[:int(total_img*train_rate),:]
val_data = data_pair.values[int(total_img*train_rate):,:]

# Add Urban dataset later
#
#

train_data = np.array(shuffle(train_data))# [[file_directory, target_name],]
val_data = np.array(shuffle(val_data))

print("TOTAL DATA: {}".format(len(data_pair)))
print("TRAIN DATA: {}".format(len(train_data)))
print("TEST DATA: {}".format(len(val_data)))

################################################################################

################################################################################
################################### GENERATOR ##################################
################################################################################

def load_batch(data_pair, batch_size):
	batch_count = 1
	index = 0
	while index < len(data_pair):
		try:
			x_data = np.array(())
			y_data = []
			count = 0

			while count < batch_size and index < len(data_pair):
				pair = data_pair[index]
				features = np.load(pair[0])
				features = features.reshape(-1,features.shape[0],features.shape[1],features.shape[2])
				target = pair[1]
				if x_data.shape[0] == 0:
					x_data = features
				else:
					x_data = np.concatenate((x_data, features), axis=0)

				y_data.append(target) # one-hot encoding

				count += 1
				index += 1

			batch_count += 1
			yield x_data, np.array(y_data)

		except Exception as e:
			print("Error: {}".format(e))

################################################################################

class CustomHistory(keras.callbacks.Callback):
	def init(self):
		self.train_loss = []
		self.val_loss = []
		self.train_acc = []
		self.val_acc = []
		self.lr = []

		# log files for training/validation loss, accuracy
		# self.loss_training_file = open('./loss_training.txt', 'w')
		# self.loss_val_file = open('./loss_val.txt', 'w')
		# self.acc_training_file = open('./acc_training.txt', 'w')
		# self.acc_val_file = open('./acc_val.txt', 'w')

	def on_epoch_end(self, batch, logs={}):
		self.train_loss.append(logs.get('loss'))
		self.val_loss.append(logs.get('val_loss'))
		self.train_acc.append(logs.get('acc'))
		self.val_acc.append(logs.get('val_acc'))
		self.lr.append(step_decay(len(self.train_loss)))

		# log for training/validation loss, accuracy
		# self.loss_training_file.write(str(logs.get('loss')) + '\n')
		# self.loss_val_file.write(str(logs.get('val_loss')) + '\n')
		# self.acc_training_file.write(str(logs.get('acc')) + '\n')
		# self.acc_val_file.write(str(logs.get('val_acc')) + '\n')

class CustomLearningRate(keras.callbacks.LearningRateScheduler):
	def __init__(self, schedule, verbose=0):
		super().__init__(schedule, verbose)
		self.epoch = 0

	def on_epoch_begin(self, epoch, logs=None):
		epoch = self.epoch
		super().on_epoch_begin(epoch, logs)

	def on_epoch_end(self, epoch, logs=None):
		super().on_epoch_end(epoch, logs)
		self.epoch += 1

def step_decay(epoch):
	initial_lrate = learning_rate
	drop = 0.5
	epochs_drop = 40.0
	lrate = initial_lrate * math.pow(drop, math.floor((epoch)/epochs_drop))
	return lrate

class CustomModelCheckpoint(keras.callbacks.ModelCheckpoint):
	def __init__(self, filepath, monitor='val_loss', verbose=0,
				 save_best_only=False, save_weights_only=False,
				 mode='auto', period=1):
		super().__init__(filepath, monitor, verbose, save_best_only, save_weights_only, mode, period)
		self.epoch = 0

	def on_epoch_end(self, epoch, logs=None):
		epoch = self.epoch
		super().on_epoch_end(epoch, logs)
		self.epoch += 1


custom_hist = CustomHistory()
custom_hist.init()
lrCallBack = CustomLearningRate(schedule=step_decay, verbose=1)
mcCallBack = CustomModelCheckpoint(filepath='./checkpoint/{epoch:04d}-{val_loss:.4f}.hdf5',
									monitor='val_loss',
									verbose=1,
									save_best_only=False,
									save_weights_only=False)

callbacks_list = [custom_hist, lrCallBack, mcCallBack]

################################################################################
################################## OUR MODEL ###################################
################################################################################

# img_width = 299
# img_height = 299
# img_channel = 3
# img_shape = (img_width, img_height, img_channel)
#
# base_model = InceptionV3(weights='imagenet', include_top=False, input_shape=img_shape)
# x = base_model.output
# x = GlobalAveragePooling2D()(x)
# x = Dense(1024, activation='relu')(x)
# predictions = Dense(nb_classes, activation='softmax')(x)
#
# # this is the model to train
# model = Model(inputs=base_model.input, outputs=predictions)

model = load_model('./checkpoint_set2/0035-1.4992.hdf5')
old = model.layers[-2]
old.save('./real_v1.hdf5')
model.summary()
# model = multi_gpu_model(model, gpus=2)
# train only the top layers (which were randomly initialized)
# i.e. freeze all convolutional InceptionV2 layers

# compile the model (should be done "after" setting layers to non-trainable)
model.compile(optimizer=Adam(lr=learning_rate), loss='categorical_crossentropy', metrics=['accuracy'])
# with open('ModelSummary.txt', 'w') as fh:
# 	model.summary(print_fn=lambda x: fh.write(x + '\n'))
#
################################################################################

print("Start Training...")

# try:
#for epoch in range(total_epochs):
epoch = 0
while True:
	print('-'*20, 'On epoch: {}'.format(epoch), '-'*20)
	# for x_data, y_label in load_batch(data_path, batch_size, total_data_df):
	# 	history = model.fit(x_data, y_label, verbose=1, epochs=1, validation_split=.2, batch_size=batch_size)

	history = model.fit_generator(generator=load_batch(train_data, batch_size),
									steps_per_epoch=len(train_data)//batch_size,
									verbose=1,
									epochs=1,
									validation_data=load_batch(val_data, batch_size),
									validation_steps=len(val_data)//batch_size,
									shuffle=True,
									callbacks=callbacks_list)


	# scores = model.evaluate_generator(load_batch(val_data, batch_size), len(val_data)//batch_size, verbose=1)
	# print("{}: {}".format(model.metrics_names[1], scores[1]*100))
	#
	# print("Saving checkpoint on epoch {}".format(epoch))
	# model.save('./checkpoint/model_chkp_{}.h5'.format(epoch))
	# print("Checkpoint saved. Continuing...")
	#
	# model.save('model_new.h5')
	epoch += 1

# except Exception as e:
# 	print("Excepted with " + str(e))
# 	print("Saving model...")
# 	model.save('excepted_model.h5')
# 	print("Model saved.")

# save final model
# model.save('model.h5')
# print("Model saved. Finished training.")
