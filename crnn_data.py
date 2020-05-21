
import numpy as np
import os
import cv2
import string

'''
classes:
1) InputGenerator
2) GTUtility

non-class methods:
1) decode
2) crop_words
'''

# character list for crnn training
#alphabet87 = string.ascii_lowercase + string.ascii_uppercase + string.digits + ' +-*.,:!?%&$~/()[]<>"\'@#_'
# alphabet87+="　4Jナッ受築３LV8ムワ新授補そ他番本鏡※内成0吊除廊畳(－イ５藤タ奥応X開の月号知ア宮D特色G戸検尺6株工目音燃栓P日助９豊2２g面階博B散W図務所E乳袋上合A正質水凡様名ン＝/エ壁式設者仕ク山登モ器市６地防当容計更ャ粧店パ理酸会Ｎ（議町フ整Ｄ士ネ板ラ動直台的CFO流洗Iコ田録e担棚伊,近7M室ス級）村i腰せTいHけ：縮県一陶隠幕カ間不記)愛住８グ建０社ー年第1付し35素立柱ブ掛仮訂ボ展風ト衣多野例ウ１康ル岡脱接方称ｏN消Ｂ火プ寄下メミ移７ＡS西Rシサ事t切テx４化レ9．-リ黒"
alphabet87=" ကခဂဃငစဆဇဈညတထဒဓနဋဌဍဎဏပဖဗဘမယရလဝသဟဠအ_"
# chracter decode
def decode(chars):
    blank_char = '_'
    new = ''
    last = blank_char
    for c in chars:
        if (last == blank_char or last != c) and c != blank_char:
            new += c
        last = c
    return new

# cropping fixed size line image
def crop_words(img, boxes, height, width=None, grayscale=True):
    """
    # Note: make sure that the vertices of all boxes are inside the image
    """

    words = []
    for j in range(len(boxes)):
        h, w = img.shape[:2]

        # box case
        box = np.round(boxes[j] * [w, h, w, h]).astype(np.int32)
        xmin, ymin, xmax, ymax = box
        word_w, word_h = xmax - xmin, ymax - ymin
        word_ar = word_w / word_h
        word_h = int(height)
        word_w = int(round(height * word_ar))

        word = img[ymin:ymax,xmin:xmax,:]
        word = cv2.resize(word, (word_w, word_h), interpolation=cv2.INTER_CUBIC)

        if grayscale:
            word = cv2.cvtColor(word, cv2.COLOR_BGR2GRAY)
            word = cv2.normalize(word, word, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
            word = word[:,:,None]

        word = word.astype(np.float32)

        if width is not None:
            tmp_word = word[:,:width,:]
            word = np.zeros([height, width, tmp_word.shape[2]])
            word[:,slice(0, tmp_word.shape[1]), :] = tmp_word

        words.append(word)
    return words


class InputGenerator(object):
    """Model input generator for cropping bounding boxes."""
    """
    Parameters >>
    	Input: gt_util
    	batch_size:
    	alphabet:
    	input_size
    	grayscale
    	max_string_len
    	concatenate
    """
    def __init__(self, gt_util, batch_size, alphabet, input_size=(255,32),
                grayscale=True, max_string_len=30, concatenate=False):

        self.__dict__.update(locals())

    def generate(self, train=True):
        gt_util = self.gt_util

        alphabet = self.alphabet
        batch_size = self.batch_size
        width, height = self.input_size
        max_string_len = self.max_string_len
        concatenate = self.concatenate
        num_input_channels = 1 if self.grayscale else 3

        inputs = []
        targets = []

        np.random.seed(1337)
        i = gt_util.num_samples
        while True:
            while len(targets) < batch_size:
                if i == gt_util.num_samples:
                    idxs = np.arange(gt_util.num_samples)
                    np.random.shuffle(idxs)
                    i = 0
                    #print('NEW epoch')
                idx = idxs[i]
                i += 1

                img_name = gt_util.image_names[idx]
                img_path = os.path.join(gt_util.image_path, img_name)
                img = cv2.imread(img_path)
                #mean = np.array([104,117,123])
                #img -= mean[np.newaxis, np.newaxis, :]
                boxes = np.copy(gt_util.data[idx][:,:-1])
                texts = np.copy(gt_util.text[idx])

                # drop boxes with vertices outside the image
                mask = np.array([not (np.any(b < 0.) or np.any(b > 1.)) for b in boxes])
                boxes = boxes[mask]
                texts = texts[mask]
                if len(boxes) == 0: continue

                try:
                    words = crop_words(img, boxes, height,
                                       None if concatenate else width,
                                       self.grayscale)
                except Exception as e:
                    import traceback
                    print(traceback.format_exc())
                    print(img_path)
                    continue

                # drop words with width < height here
                mask = np.array([w.shape[1] > w.shape[0] for w in words])
                words = [words[j] for j in range(len(words)) if mask[j]]
                texts = texts[mask]
                if len(words) == 0: continue

                # shuffle words
                idxs_words = np.arange(len(words))
                np.random.shuffle(idxs_words)
                words = [words[j] for j in idxs_words]
                texts = texts[idxs_words]

                # concatenate all instances to one sample
                if concatenate:
                    words_join = []
                    texts_join = []
                    total_w = 0
                    for j in range(len(words)):
                        if j == 0:
                            w = np.random.randint(8)
                        else:
                            w = np.random.randint(4, 32)
                        total_w = total_w + w + words[j].shape[1]
                        if total_w > width:
                            break
                        words_join.append(np.zeros([height, w, words[j].shape[2]]))
                        words_join.append(words[j])
                        texts_join.append(' ')
                        texts_join.append(texts[j])
                    if len(words_join) == 0: continue

                    words_tmp = np.concatenate(words_join, axis=1)
                    words = np.zeros([height, width, words_tmp.shape[2]])
                    words[:,slice(0, words_tmp.shape[1]), :] = words_tmp

                    texts = ''.join(texts_join)

                    inputs.append(words)
                    targets.append(texts)
                else:
                    inputs.extend(words)
                    targets.extend(texts)

            #yield inputs[:batch_size], targets[:batch_size]

            source_str = np.array(targets[:batch_size])

            images = np.ones([batch_size, width, height, num_input_channels])
            labels = -np.ones([batch_size, max_string_len])
            input_length = np.zeros([batch_size, 1])
            label_length = np.zeros([batch_size, 1])
            for j in range(batch_size):
                images[j] = inputs[j].transpose(1,0,2)
                input_length[j,0] = max_string_len
                label_length[j,0] = len(source_str[j])
                for k, c in enumerate(source_str[j][:max_string_len]):
                    if not c in alphabet or c == '_':
                        #print('bad char', c)
                        labels[j][k] = alphabet.index(' ')
                    else:
                        labels[j][k] = alphabet.index(c)

            inputs_dict = {
                'image_input': images,
                'label_input': labels,
                'input_length': input_length, # used by ctc
                'label_length': label_length, # used by ctc
                'source_str': source_str, # used for visualization only
            }
            outputs_dict = {'ctc': np.zeros([batch_size])}  # dummy
            yield inputs_dict, outputs_dict

            inputs = inputs[batch_size:]
            targets = targets[batch_size:]

class GTUtility():
    """
    # Arguments
        data_path: path to ground truth and image data.
        test: Boolean for using training or test set.
    """

    def __str__(self):
        s = ''
        s += '\n'
        s += '%-16s %8i\n' % ('images', self.num_images)
        s += '%-16s %8i\n' % ('objects', self.num_objects)
        s += '%-16s %8.2f\n' % ('per image', self.num_objects/self.num_images)
        s += '%-16s %8i\n' % ('no annotation', self.num_without_annotation)
        return s

    def __init__(self, data_path, test=False):
        self.data_path = data_path
        if test:
            gt_path = os.path.join(data_path, 'test_gt')
            image_path = os.path.join(data_path, 'test_img')
        else:
            gt_path = os.path.join(data_path, 'train_gt')
            image_path = os.path.join(data_path, 'train_img')
        self.gt_path = gt_path
        self.image_path = image_path
        self.sample_count=0
        self.image_names = []
        self.data = []
        self.text = []
        for image_name in os.listdir(image_path):
            #print(image_name)
            img_height, img_width = cv2.imread(os.path.join(image_path, image_name)).shape[:2]
            boxes = []
            text = []
            gt_file_name = 'gt_' + os.path.splitext(image_name)[0] + '.txt'

            with open(os.path.join(gt_path, gt_file_name), 'r') as f:
                for line in f:
                    line_split = line.strip().split('|')
                    assert len(line_split) == 5, "length is %d" % len(line_split)
                    box = [float(v.replace('\ufeff','')) for v in line_split[:4]]

                    box[0] /= img_width
                    box[1] /= img_height
                    box[2] /= img_width
                    box[3] /= img_height

                    box = box + [0] # zero is for dummy label, necessary for crnn training
                    boxes.append(box)
                    text.append(line_split[4])

            boxes = np.asarray(boxes)
            self.data.append(boxes)
            self.text.append(text)
            self.image_names.append(image_name)

        num_without_annotation = 0
        for i in range(len(self.data)):
            if len(self.data[i]) == 0:
                num_without_annotation += 1
            else:
                self.sample_count+=len(self.data[i])

        self.num_without_annotation = num_without_annotation

        self.num_samples = len(self.image_names)
        self.num_images = len(self.data)
        self.num_objects = self.sample_count
