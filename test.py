import numpy as np
import os
import editdistance
import time

os.environ["CUDA_VISIBLE_DEVICES"] = '0'

from crnn_model import CRNN
from utils.training import Logger, ModelSnapshot
from crnn_data import GTUtility, InputGenerator, decode, alphabet87 as alphabet

gt_util_test = GTUtility('dataset/', test=True)

input_width = 256
input_height = 32
batch_size = 16
input_shape = (input_width, input_height, 1)

model, model_pred = CRNN(input_shape, len(alphabet)+1)
#experiment = 'crnn_lstm_synthtext'

max_string_len = model_pred.output_shape[1]

gen_val = InputGenerator(gt_util_test, batch_size, alphabet, input_shape[:2],
                         grayscale=True, max_string_len=max_string_len)

model.load_weights('./model/weights.h5')

g = gen_val.generate()
n = 150

mean_ed = 0
mean_ed_norm = 0
mean_character_recogniton_rate = 0
sum_ed = 0
char_count = 0
correct_word_count = 0
lenchar = len(alphabet)
#print("len alphabet", lenchar)

word_recognition_rate = 0

j = 0
while j < n:
    d = next(g)
    res = model_pred.predict(d[0]['image_input'])
    #print("Result length", len(res))
    for i in range(len(res)):
        if not j < n: break
        j += 1

        #print("Result[i] --->", np.argmax(res[i], axis = 1))
 #       best path
        chars = [alphabet[34] if c==lenchar else alphabet[c] for c in np.argmax(res[i], axis=1)]
        gt_str = d[0]['source_str'][i]
        res_str = decode(chars)

        ed = editdistance.eval(gt_str, res_str)
        ed_norm = ed / len(gt_str)
        mean_ed += ed
        mean_ed_norm += ed_norm

        sum_ed += ed
        char_count += len(gt_str)
        if ed == 0.: correct_word_count += 1

        print('%20s %20s %f' %(gt_str, res_str, ed))


mean_ed /= j
mean_ed_norm /= j
character_recogniton_rate = (char_count-sum_ed) / char_count
word_recognition_rate = correct_word_count / j

print('mean editdistance             %0.3f' % (mean_ed))
print('mean normalized editdistance  %0.3f' % (mean_ed_norm))
print('character recogniton rate     %0.3f' % (character_recogniton_rate))
print('word recognition rate         %0.3f' % (word_recognition_rate))
