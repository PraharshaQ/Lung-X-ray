#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 11 16:52:23 2018

@author: praharsha
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep  3 11:46:26 2018

@author: praharsha
"""

import tensorflow as tf, random, numpy as np
from PIL import Image
#from multiprocessing import cpu_count
#import pandas as pd



#HERE,image_arr,image_str are image in array format and path os images in string format respectively.
#here,filename in label of images in sting or array format 

#batch_size=64
num_classes=2
labels=["Defective","Non-Defective"]
#labels = ["Cardiomegaly","Emphysema","Effusion","Hernia","No Finding","Infiltration","Mass","Nodule","Pneumothorax","Pleural_Thickening","Atelectasis","Fibrosis","Edema","Consolidation","Pneumonia"]   
#csv_file="/home/praharsha/Desktop/RnD/Nvidia/chestXray_dataset/data.csv"#it is a file containing two colums.first colum is image paths and second column is labels or class of image.both of them dopesnt have any headings.
#They start with image path and label of first image as first row.
def preprocess(image_arr, filename):#image-arr is array form of images,filename is label whichnis in string form
    #image=Image.open(image_arr)
    #print(image_arr.shape)
    image = Image.fromarray(image_arr[:,:,0])#LHS is image and RHS image is array format of image.The function converts array format of image into image
    image=image.resize((256,256), Image.ANTIALIAS)
    width,height = image.size
    #print(width,height)
    augmentation = random.choice(['rotation', 'flip', 'both',"original"])
    if (augmentation == 'rotation'):
            angle = random.choice(range(-10,10))#gives an angle between 10 to 10
            image = image.rotate(angle)#rotates image
    elif (augmentation == 'original'):
            pass
    elif (augmentation == 'flip'):
            #for i in range(random.choice(range(3))):
            image = image.transpose(Image.FLIP_LEFT_RIGHT)#flips the image
    else:#flip and rotate
            angle = random.choice(range(-10,10))
            image = image.rotate(angle)
            #for i in range(random.choice(range(3))):
            image = image.transpose(Image.FLIP_LEFT_RIGHT)
    
    label = one_hot_labels(filename)
    #label = filename.split('_')[-1][:-4]
    return np.array(image)/255., label.astype(np.int)#returning array format of image and integer format of label

#preprocess("/home/praharsha/Desktop/RnD/Nvidia/chestXray_dataset/images/00000001_001.png","Cardiomegaly|Emphysema")

def one_hot_labels(filename):
    #image_labels_oh=[]
    #for i in filename:
    filename=filename.decode()
    #converting from binary string to string.
    p=np.zeros(num_classes, int)
    #for i in tf.string_split(filename,delimiter='|'):
    p[labels.index(filename)] = 1
    #image_labels_oh.append(p)
    return p                
    #return np.asarray(image_labels_oh)


def input_parser(image_str,filename):
    binary = tf.read_file(image_str)#Reads and outputs the entire contents of the input filename.
    image = tf.image.decode_image(binary, channels=1)#Detects whether an image is a BMP, GIF, JPEG, or PNG, and performs the appropriate operation to convert the input bytes string into a Tensor of type dtype.
    #b = tf.read_file(filename) reads the emtire connents pf the filename
    #tf.image.decode_image(b) outputs the array format of image.
    #the two functions do is,read data from a file(for eg: filename contains location of the images) and convert that image in file to array from.
    inputs = tf.py_func(preprocess,[image, filename], [tf.double, tf.int64])#converting other libraries into tensorflow format
    #Given a python function func, which takes numpy arrays as its arguments and returns numpy arrays as its outputs, wrap this function as an operation in a TensorFlow graph.
    inputs[0] = tf.cast(inputs[0], tf.float32)
    
    #inputs[1]=one_hot_labels(filename)
    #inputs[1] = tf.one_hot(inputs[1], num_classes)
    return inputs[0], inputs[1]

def parser_csv(line):
    parsed_line = tf.decode_csv(line, [['string'], ['string']])
    #print(parsed_line)
    return parsed_line[0], tf.cast(parsed_line[1], dtype=tf.string)


def input_from_csv(csv_file, epochs, batch_size,num_gpus):
    def input_fn():
        #step1:tf.data.TextLineDataset(csv_file) function does is,it take a line from csv_file.i.e,it take imagepath and label of one image.
        #parameter.map(parse_csv) does is,it maps parser_csv funtion to parameter as input.i.e, it runs as parser_csv(parameter).
        #so,output of step1 is two strings.one is image path and the other is label of image
        dataset = tf.data.TextLineDataset(csv_file).map(parser_csv, num_parallel_calls = num_gpus)
        #step2: it maps the input_parser funtion to dataset.ie, input_parser(dataset)
        #it returns array of image and label in string format.
        dataset = dataset.map(input_parser, num_parallel_calls=num_gpus)
        #it repeats with a batch size on whole dataset.
        dataset = dataset.repeat(epochs).batch(batch_size)
        #it only supports iterating once through a dataset, with no need for explicit initialization
        #One-shot iterators handle almost all of the cases that the existing queue-based input pipelines support, but they do not support parameterization.
        iterator = dataset.make_one_shot_iterator()
        #goes to next element.
        feats, labs = iterator.get_next()
        return tf.reshape(feats,(-1,256,256,1)), tf.cast(labs,tf.float32)   
    return input_fn

#init=tf.global_variables_initializer()
#sess=tf.Session() #sess.run(init)
#
#ff, ll = sess.run(input_from_csv("/home/praharsha/Desktop/RnD/Nvidia/class_2/data_2.csv",1,2,1)())   
#ff, ll = sess.run(input_from_csv("/home/praharsha/Desktop/RnD/Nvidia/chestXray_dataset/data.csv",1,2,1)())
#print(ff.shape,ll.shape)

