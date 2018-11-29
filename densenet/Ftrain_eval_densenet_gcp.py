#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 19 11:28:47 2018

@author: vardhan
"""


#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 18 22:51:00 2018

@author: praharsha
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep  6 12:18:05 2018

@author: praharsha
"""

import argparse
import tensorflow as tf

from model_densenet_5_gcp import dense_net
from input_5 import input_from_csv

tf.logging.set_verbosity(tf.logging.INFO)

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
    is_training = bool(mode == tf.estimator.ModeKeys.TRAIN)
    #mode=tf.estimator.ModeKeys.TRAIN
    #print("labels: ", labels.shape)
    logits = dense_net(features,is_training)
    #print("Logits: ", logits.shape)
    predictions = tf.round(tf.nn.sigmoid(logits, name="sigmoid_tensor"))
    loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=labels,logits=logits))
    #print("Loss: ", loss)
    
    #f_score = tf.contrib.metrics.f1_score(labels, predictions)
    def metric_fn(predictions=predictions,labels=labels,weights=None):
        P,update_op1=tf.contrib.metrics.streaming_precision(predictions,labels)
        R,update_op2=tf.contrib.metrics.streaming_recall(predictions,labels)
        eps=1e-5
        return (2*(P*R)/(P+R+eps),tf.group(update_op1,update_op2))
    
    '''
    labels_bool=tf.cast(labels,tf.bool)
    pred_bool=tf.cast(predictions,tf.bool)
    
    TP = tf.reduce_sum(tf.cast(tf.logical_and(labels_bool,pred_bool),tf.float32))
    TPFP= tf.cast(tf.reduce_sum(predictions),tf.float32)
    TPFN=tf.cast(tf.reduce_sum(labels),tf.float32) 
    
    prec= 1.0*TP/(TPFP)
    rec= 1.0*TP/(TPFN)

    F1 = (2.*prec*rec)/(prec+rec)
    '''
    
    eval_metrics={"F1 Score":metric_fn(predictions,labels)}
    
    #eval_metrics={"accuracy":tf.metrics.accuracy(labels=labels,predictions=predictions)}
    if mode == tf.estimator.ModeKeys.TRAIN:
        optimizer = tf.train.AdamOptimizer(learning_rate=params.learning_rate)
        if params.num_gpus > 1:
            optimizer = tf.contrib.estimator.TowerOptimizer(optimizer)
        update_op = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        train_op = [optimizer.minimize(loss, global_step=tf.train.get_global_step())]
        train_op.extend(update_op)
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
    train_input_fn = input_from_csv(hparams.train_data_csv, hparams.num_epochs, hparams.train_batch_size, hparams.num_gpus)
    eval_input_fn = input_from_csv(hparams.eval_data_csv, 1, hparams.eval_batch_size, hparams.num_gpus)
    train_spec = tf.estimator.TrainSpec(train_input_fn, max_steps=int(hparams.train_steps))
    eval_spec = tf.estimator.EvalSpec(eval_input_fn, steps=int(hparams.eval_steps), name='estimator-eval', throttle_secs=1800)
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
    sess_config = tf.ConfigProto(allow_soft_placement=True, gpu_options=tf.GPUOptions(force_gpu_compatible=True))
    config = tf.estimator.RunConfig(session_config=sess_config, model_dir=model_dir, keep_checkpoint_max=200,
                                    log_step_count_steps=100, save_checkpoints_steps =500)
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
            default='GPU',
            help='where to locate variable operations')
    parser.add_argument(
            '--num_gpus',
            type=int,
            default=2,
            help='number of GPUs used. Uses only CPU if set to 0')
    parser.add_argument(
            '--num_workers',
            type=int,
            default=1,
            help='number of worker replicas in sync mode')
    parser.add_argument(
            '--num_epochs',
            type=int,
            default=1,
            help='use either epochs or steps')
    parser.add_argument(
            '--train_steps',
            type=int,
            default=800000,
            help='number of steps to use for training')
    parser.add_argument(
            '--eval_steps',
            type=int,
            default=50,
            help='number of steps to use for training')
    parser.add_argument(
            '--train_batch_size',
            type=int,
            default=64,
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