import logging

import cxflow_tensorflow as cxtf
import tensorflow as tf
import tensorflow.contrib.keras as K
import numpy as np

class ShotsConvNet(cxtf.BaseModel):

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

        images = tf.placeholder(dtype=tf.float32, shape=[None, 10, 32, 32, 3], name='images')
        labels = tf.placeholder(dtype=tf.int64, shape=[None, 10], name='labels')

        with tf.variable_scope('conv1'):
            net = images
            net = K.layers.Conv3D(32, 3, padding='same')(net)
            net = K.layers.MaxPool3D((1, 2, 2))(net)
        with tf.variable_scope('conv2'):
            net = K.layers.Conv3D(64, 3, padding='same')(net)
            net = K.layers.MaxPool3D((1, 2, 2))(net)
        with tf.variable_scope('dense4'):
            net = self.flatten3D(net)
            net = K.layers.Dropout(0.5).apply(net, training=self.is_training)
            net = K.layers.Dense(512)(net)
        with tf.variable_scope('dense5'):
            logits = K.layers.Dense(2, activation=None)(net)

        loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels, logits=logits)
        tf.identity(loss, name='loss')
        predictions = tf.argmax(logits, 2, name='predictions')
        tf.reduce_mean(tf.cast(tf.equal(predictions, labels), tf.float32), 1, name='accuracy')
