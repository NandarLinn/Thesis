import numpy as np
import os
import time

os.environ["CUDA_VISIBLE_DEVICES"] = '0'

from keras.optimizers import SGD, Adam
from keras.callbacks import ModelCheckpoint

from crnn_model import CRNN
from utils.training import Logger, ModelSnapshot
from crnn_data import GTUtility, InputGenerator, decode, alphabet87 as alphabet

#Load dataset
gt_util_train = GTUtility('dataset/')
gt_util_test = GTUtility('dataset/', test=True)

input_width = 256
input_height = 32
batch_size = 16
input_shape = (input_width, input_height, 1)

model, model_pred = CRNN(input_shape, len(alphabet)+1)
#experiment = 'crnn_lstm_synthtext'

max_string_len = model_pred.output_shape[1]

gen_train = InputGenerator(gt_util_train, batch_size, alphabet, input_shape[:2],
                           grayscale=True, max_string_len=max_string_len)
gen_val = InputGenerator(gt_util_test, batch_size, alphabet, input_shape[:2],
                         grayscale=True, max_string_len=max_string_len)

checkdir = './model'
if not os.path.exists(checkdir):
    os.makedirs(checkdir)

optimizer = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True, clipnorm=5)

# dummy loss, loss is computed in lambda layer
model.compile(loss={'ctc': lambda y_true, y_pred: y_pred}, optimizer=optimizer)
model.summary()

model.fit_generator(generator=gen_train.generate(), # batch_size here?
                    steps_per_epoch=gt_util_train.num_objects // batch_size,
                    epochs=50,
                    validation_data=gen_val.generate(), # batch_size here?
                    validation_steps=gt_util_test.num_objects // batch_size,
                    callbacks=[
                        ModelCheckpoint(checkdir+'/weights.h5', verbose=1, save_weights_only=True),
                        ModelSnapshot(checkdir, 10000),
                        Logger(checkdir)
                    ],
                    initial_epoch=0)

print(" ------------- Training Steps Finished ------------- ")
