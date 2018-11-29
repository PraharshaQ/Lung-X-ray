#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep  6 12:18:05 2018

@author: praharsha
"""

import argparse
import tensorflow as tf

from model import model as sq_model
from input_copy import input_from_csv

#os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'#INFO and WARNING not printed
tf.logging.set_verbosity(tf.logging.INFO)#to know info about INFO

def model_fn(features, labels, mode, params):
    '''
    Args:
        features: Input-features - Images (Obtained from Input-Function)
        labels: Class-lables (Obtained from Input-Function)
        mode: Modekeys.TRAIN or EVAL
        params: Hyper-parameters - Args provided
    Returns:
        EstimatorSpec object
    '''
    is_training = (mode == tf.estimator.ModeKeys.TRAIN)
    print("labels: ", labels.shape)
    logits = sq_model(features, is_training, tf.AUTO_REUSE)#tf.AUTO_REUSE-same value is assigned to two or more different variables.
    print("Logits: ", logits.shape)
    #predictions = {"probabilities": tf.round(tf.nn.sigmoid(logits, name="sigmoid_tensor"))}
    predictions = tf.round(tf.nn.sigmoid(logits, name="sigmoid_tensor"))
    
    
    loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=labels,logits=logits))
    print("Loss: ", loss)
    
    eval_metrics={"accuracy":tf.metrics.accuracy(labels=labels,predictions=predictions)}
    
    if mode == tf.estimator.ModeKeys.TRAIN:
        
        optimizer = tf.train.AdamOptimizer(learning_rate=params.learning_rate)
        if params.num_gpus > 1:
            optimizer = tf.contrib.estimator.TowerOptimizer(optimizer)#depriciated.use 
            #tf.contrib.distribute.MirroredStrategy
            
        update_op = tf.get_collection(tf.GraphKeys.UPDATE_OPS)#basically UPDATE_OPS is used for batchnorm parameters to updtae
        train_op = [optimizer.minimize(loss, global_step=tf.train.get_global_step())]#now trian_op is list
        train_op.extend(update_op)#we have created train_op as list because we can add update_op to train_op using extend()
        train_op = tf.group(*train_op)
        return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)
    
    return tf.estimator.EstimatorSpec(mode=mode, loss=loss, eval_metric_ops=eval_metrics)


def _experiment_fn(run_config, hparams):
    '''
    Takes inputs required for model_fn

    Args:
        variable_strategy: CPU / GPU - What to use for paramter-serving
        run_config: Config - Provides num_workers param required by the model_fn
        num_gpus = Number of gpus
        args - Hyperparamters and other argumets required for training and evaluation
    Returns:
        estimator.train_and_evaluate object
    '''
    #not thet u have to change them
    train_input_fn = input_from_csv(hparams.train_data_csv, hparams.num_epochs, hparams.train_batch_size, hparams.num_gpus)
    eval_input_fn = input_from_csv(hparams.eval_data_csv, 1, hparams.eval_batch_size, hparams.num_gpus)
        # Create Specs
    train_spec = tf.estimator.TrainSpec(train_input_fn, max_steps=int(hparams.train_steps))
    eval_spec = tf.estimator.EvalSpec(eval_input_fn, steps=int(hparams.eval_steps), name='estimator-eval', throttle_secs=200)
    
    if hparams.num_gpus > 1:
        model_function = tf.contrib.estimator.replicate_model_fn(model_fn=model_fn,loss_reduction=tf.losses.Reduction.MEAN)
    else:
        model_function = model_fn
        
    classifier = tf.estimator.Estimator(model_fn=model_function, config=run_config, params=hparams)
    return tf.estimator.train_and_evaluate(classifier, train_spec, eval_spec)

def main(model_dir, **hparams):
    '''
    Main-Function:
        Creates config and runs the above _experiment_fn
        '''
    hparams = tf.contrib.training.HParams(**args.__dict__)
    # Session-Config
    sess_config = tf.ConfigProto(allow_soft_placement=True, gpu_options=tf.GPUOptions(force_gpu_compatible=True))
    config = tf.estimator.RunConfig(session_config=sess_config, model_dir=model_dir, keep_checkpoint_max=20,
                                    log_step_count_steps=100, save_checkpoints_steps =100)
    _experiment_fn(config, hparams)
    

# Provide Args and call the main-fn
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
            '--model_dir',
            type=str,
            required=True,
            help='directory where the model will be stored.')
    parser.add_argument(
            '--job_dir',
            type=str,
            required=False,
            help='directory to package trainer')
    parser.add_argument(
            '--train_data_csv',
            type=str,
            required=True,
            help='directory for training files')
    parser.add_argument(
            '--eval_data_csv',
            type=str,
            required=True,
            help='directory for eval files')
    parser.add_argument(
            '--variable_strategy',
            choices=['CPU', 'GPU'],
            type=str,
            default='CPU',
            help='where to locate variable operations')
    parser.add_argument(
            '--num_gpus',
            type=int,
            default=1,
            help='number of GPUs used. Uses only CPU if set to 0')
    parser.add_argument(
            '--num_workers',
            type=int,
            default=1,
            help='number of worker replicas in sync mode')
    parser.add_argument(
            '--num_epochs',
            type=int,
            help='use either epochs or steps')
    parser.add_argument(
            '--train_steps',
            type=int,
            default=2000,
            help='number of steps to use for training')
    parser.add_argument(
            '--eval_steps',
            type=int,
            default=5,
            help='number of steps to use for training')
    parser.add_argument(
            '--train_batch_size',
            type=int,
            default=32,
            help='batch size for training')
    parser.add_argument(
            '--eval_batch_size',
            type=int,
            default=32,
            help='batch size for eval')
    parser.add_argument(
            '--learning_rate',
            type=float,
            default=1e-3,
            help='learning_rate for training')
    parser.add_argument(
            '--sync',
            action='store_true',
            default=False,
            help='batch size for eval')

    args = parser.parse_args()
    main(**vars(args))