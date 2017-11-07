import logging

import cxflow_tensorflow as cxtf
import tensorflow as tf
import tensorflow.contrib.keras as K


class ShotsConvNet(cxtf.BaseModel):
    """Simple 2-layered MLP for majority task."""

    def _create_model(self, **kwargs):
        logging.debug('Constructing placeholders matching the model.inputs')

        sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))

        images = tf.placeholder(dtype=tf.float32, shape=[None, 32, 32, 32, 3], name='images')
        labels = tf.placeholder(dtype=tf.float32, shape=[None, 32], name='labels')

        with tf.variable_scope('conv1'):
            net = images
            net = K.layers.Conv3D(32, 3)(net)
            net = K.layers.MaxPool3D()(net)
        with tf.variable_scope('conv2'):
            #net = tf.expand_dims(images, -1)
            net = K.layers.Conv3D(64, 3)(net)
            net = K.layers.MaxPool3D()(net)
        with tf.variable_scope('conv3'):
            #net = tf.expand_dims(images, -1)
            net = K.layers.Conv3D(128, 3)(net)
            net = K.layers.MaxPool3D()(net)
        with tf.variable_scope('dense4'):
            net = K.layers.Flatten()(net)
            net = K.layers.Dropout(0.5).apply(net, training=self.is_training)
            net = K.layers.Dense(512)(net)
        with tf.variable_scope('dense5'):
            logits = K.layers.Dense(32, activation=None)(net)

        loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=labels, logits=logits)
        tf.identity(loss, name='loss')
        #acc = tf.cast(tf.div(tf.reduce_sum(tf.cast(tf.equal(logits, labels),tf.float32)), 32.0), tf.float32)
        #tf.identity(acc, name='accuracy')
        #tf.tuple(tf.divide(tf.reduce_sum(tf.cast(tf.equal(logits, labels), tf.float32)), 32.0), name='accuracy')
        #accuracy = tf.metrics.accuracy(predictions=logits, labels=labels, name='accuracy')
        #tf.identity(accuracy, name='accuracy')

