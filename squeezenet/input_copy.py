

import tensorflow as tf, random, numpy as np
from PIL import Image

num_classes=15
labels = ["Cardiomegaly","Emphysema","Effusion","Hernia","No Finding","Infiltration","Mass","Nodule","Pneumothorax","Pleural_Thickening","Atelectasis","Fibrosis","Edema","Consolidation","Pneumonia"]

def preprocess(image_arr, filename):
    image = Image.fromarray(image_arr[:,:,0])
    image=image.resize((256,256), Image.ANTIALIAS)
    width,height = image.size
    augmentation = random.choice(['rotation', 'flip', 'both',"original"])
    if (augmentation == 'rotation'):
            angle = random.choice(range(-10,10))
            image = image.rotate(angle)
    elif (augmentation == 'original'):
            pass
    elif (augmentation == 'flip'):
            image = image.transpose(Image.FLIP_LEFT_RIGHT)
    else:
            angle = random.choice(range(-10,10))
            image = image.rotate(angle)
            image = image.transpose(Image.FLIP_LEFT_RIGHT)
    label = one_hot_labels(filename)
    return np.array(image)/255., label.astype(np.int)


def one_hot_labels(filename):
    filename=filename.decode()
    p=np.zeros(num_classes, int)
    for i in filename.split('|'):
            p[labels.index(i)] = 1
    return p

def input_parser(image_str,filename):
    binary = tf.read_file(image_str)
    image = tf.image.decode_image(binary, channels=1)
    inputs = tf.py_func(preprocess,[image, filename], [tf.double, tf.int64])
    inputs[0] = tf.cast(inputs[0], tf.float32)
    return inputs[0], inputs[1]

def parser_csv(line):
    parsed_line = tf.decode_csv(line, [['string'], ['string']])
    return parsed_line[0], tf.cast(parsed_line[1], dtype=tf.string)


def input_from_csv(csv_file, epochs, batch_size,num_gpus):
    def input_fn():
        dataset = tf.data.TextLineDataset(csv_file).map(parser_csv, num_parallel_calls = num_gpus)
        dataset = dataset.map(input_parser, num_parallel_calls=1)
        dataset = dataset.repeat(epochs).batch(batch_size)
        iterator = dataset.make_one_shot_iterator()
        feats, labs = iterator.get_next()
        return tf.reshape(feats,(-1,256,256,1)), tf.cast(labs,tf.float32)
    return input_fn


'''
init=tf.global_variables_initializer()
sess=tf.Session() #sess.run(init)

ff, ll = sess.run(input_from_csv(csv_file,1,32,2)())
print(ff.shape,ll.shape)
'''
