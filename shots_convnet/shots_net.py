import logging

import cxflow_tensorflow as cxtf
import tensorflow as tf
import tensorflow.contrib.keras as K
import numpy as np

class ShotsConvNet(cxtf.BaseModel):
    """Simple 2-layered MLP for majority task."""

    def flatten3D(self, inputs: tf.Tensor) -> tf.Tensor:
        """
        Flatten the given ``inputs`` tensor to 3 dimensions.
        :param inputs: >=4d tensor to be flattened
        :return: 3d flatten tensor
        """
        assert len(inputs.get_shape().as_list()) > 3
        return tf.reshape(inputs, [tf.shape(inputs)[0], tf.shape(inputs)[1],  np.prod(inputs.get_shape().as_list()[2:])])

    def _create_model(self, **kwargs):
        logging.debug('Constructing placeholders matching the model.inputs')

        images = tf.placeholder(dtype=tf.float32, shape=[None, 100, 32, 32, 3], name='images')
        labels = tf.placeholder(dtype=tf.int64, shape=[None, 100], name='labels')

        with tf.variable_scope('conv1'):
            net = images
            net = K.layers.Conv3D(8, (5, 3, 3), padding='same')(net)
            net = K.layers.MaxPool3D((1, 2, 2))(net)
            logging.info(net.shape)
        with tf.variable_scope('conv2'):
            net = K.layers.Conv3D(16, (5, 3, 3), padding='same')(net)
            net = K.layers.MaxPool3D((1, 2, 2))(net)
            logging.info(net.shape)
        with tf.variable_scope('conv3'):
            net = K.layers.Conv3D(32, (5, 3, 3), padding='same')(net)
            net = K.layers.MaxPool3D((1, 2, 2))(net)
            logging.info(net.shape)
        with tf.variable_scope('conv4'):
            net = K.layers.Conv3D(64, (5, 3, 3), padding='same')(net)
            net = K.layers.MaxPool3D((1, 2, 2))(net)
            logging.info(net.shape)
        with tf.device('/cpu:0'):
            with tf.variable_scope('fc6'):
                net = self.flatten3D(net)
                logging.info(net.shape)
                net = K.layers.Dense(64)(net)
                logging.info(net.shape)
            with tf.variable_scope('dense6'):
                logits = K.layers.Dense(2, activation=None)(net)
                logging.info(logits.shape)

        loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels, logits=logits)
        tf.identity(loss, name='loss')
        predictions = tf.argmax(logits, 2, name='predictions')
        tf.reduce_mean(tf.cast(tf.equal(predictions, labels), tf.float32), 1, name='frame_accuracy')
        tf.cast(tf.greater_equal(tf.reduce_sum(tf.cast(tf.equal(predictions, labels), tf.float32), 1), 100), dtype=tf.float32, name='sequence_accuracy')
        sts = cxtf.metrics.bin_stats(predictions, labels)
        tf.identity(sts[0], name='f1')
        tf.identity(sts[1], name='precision')
        tf.identity(sts[2], name='recall')
