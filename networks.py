import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
from keras.layers import Dense, Flatten, Conv2D, Reshape, Conv2DTranspose


class Encoder(tf.keras.Model):
    def __init__(self, z_dim, hidden_dim=256):
        self.z_dim = z_dim
        super(Encoder, self).__init__()
        self.conv1 = Conv2D(filters=32, kernel_size=(4, 4), strides=(2, 2), padding='valid', activation='relu')
        self.conv2 = Conv2D(filters=32, kernel_size=(4, 4), strides=(2, 2), padding='valid', activation='relu')
        self.conv3 = Conv2D(filters=64, kernel_size=(4, 4), strides=(2, 2), padding='valid', activation='relu')
        self.conv4 = Conv2D(filters=128, kernel_size=(4, 4), strides=(2, 2), padding='valid', activation='relu')
        self.conv5 = Conv2D(filters=hidden_dim, kernel_size=(4, 4), strides=(1, 1), padding='valid', activation='relu')
        self.flatten = Flatten()
        self.locs_out = Dense(units=z_dim, activation=tf.nn.relu)
        self.std_out = Dense(units=z_dim, activation=tf.nn.softplus)

    def call(self, x):
        h = tf.pad(x, [[0, 0], [1, 1], [1, 1], [0, 0]], mode='CONSTANT')
        h = self.conv1(h)
        h = tf.pad(h, [[0, 0], [1, 1], [1, 1], [0, 0]], mode='CONSTANT')
        h = self.conv2(h)
        h = tf.pad(h, [[0, 0], [1, 1], [1, 1], [0, 0]], mode='CONSTANT')
        h = self.conv3(h)
        h = tf.pad(h, [[0, 0], [1, 1], [1, 1], [0, 0]], mode='CONSTANT')
        h = self.conv4(h)
        h = self.conv5(h)
        h = self.flatten(h)
        locs = self.locs_out(h)

        scale = self.std_out(h)
        scale = tf.clip_by_value(scale, clip_value_min=1e-3, clip_value_max=1e3)
        # Note if returning sampled z as well, decorate sample_fn with @tf.function so that it gets saved in the
        # computation graph and there is no need to implement it while testing or loading the model
        return locs, scale


class Decoder(tf.keras.Model):
    def __init__(self, hidden_dim=256, *args, **kwargs):
        super(Decoder, self).__init__()
        self.fc1 = Dense(units=hidden_dim, activation=tf.nn.relu)
        self.reshape = Reshape((1, 1, hidden_dim))
        self.conv1t = Conv2DTranspose(filters=128, kernel_size=(4, 4), strides=(1, 1), padding='valid', activation='relu')
        self.conv2t = Conv2DTranspose(filters=64, kernel_size=(4, 4), strides=(2, 2), padding='same', activation='relu')
        self.conv3t = Conv2DTranspose(filters=32, kernel_size=(4, 4), strides=(2, 2), padding='same', activation='relu')
        self.conv4t = Conv2DTranspose(filters=32, kernel_size=(4, 4), strides=(2, 2), padding='same', activation='relu')
        self.conv5t = Conv2DTranspose(filters=3, kernel_size=(4, 4), strides=(2, 2), padding='same', activation='sigmoid')

    def call(self, z):
        h = self.fc1(z)
        h = self.reshape(h)
        h = self.conv1t(h)
        h = self.conv2t(h)
        h = self.conv3t(h)
        h = self.conv4t(h)
        x = self.conv5t(h)
        return x


class MyInferenceLayer(tf.keras.layers.Layer):
    def __init__(self, y_dim):
        super(MyInferenceLayer, self).__init__()
        self.y_dim = y_dim

    def build(self, input_shape):
        # Parameters of the model: W (z_dim*y_dim), b (y_dim,)
        self.kernel = self.add_weight(name='kernel', shape=(input_shape[1], self.y_dim), initializer='random_normal', trainable=True)
        self.bias = self.add_weight(name='bias', shape=(self.y_dim,), initializer='random_normal', trainable=True)

    def call(self, x):
        # x is a tensor of shape (batch_size, z_dim, y_dim). We will broadcast W by multiplying with x, sum along z_dim and add bias
        return tf.reduce_sum(x*self.kernel, axis=1) + self.bias


class Classifier(tf.keras.Model):
    def __init__(self, y_dim):
        super(Classifier, self).__init__()
        # Defining a dense layer will give many-to-one mapping from z to y. CCVAE used one-to-one mapping.
        self.get_logits = MyInferenceLayer(y_dim)

    def call(self, encodes_z, gates):
        gated_z = encodes_z * gates
        logits_y = self.get_logits(gated_z)
        return logits_y


class MyCondGenerationLayer(tf.keras.layers.Layer):
    def __init__(self, z_dim, initializer=None):
        super(MyCondGenerationLayer, self).__init__()
        self.z_dim = z_dim
        self.initializer = initializer

    def build(self, input_shape):
        # Parameters of the model: W (y_dim*z_dim)
        if self.initializer == 'ones':
            self.kernel = self.add_weight(name='kernel', shape=(input_shape[1], self.z_dim), initializer='ones', trainable=True)
        elif self.initializer == 'zeros':
            self.kernel = self.add_weight(name='kernel', shape=(input_shape[1], self.z_dim), initializer='zeros', trainable=True)
        else:
            self.kernel = self.add_weight(name='kernel', shape=(input_shape[1], self.z_dim), initializer='random_normal', trainable=True)

    def call(self, x):
        # x is a tensor of shape (batch_size, y_dim, z_dim). We will broadcast W by multiplying with x, sum along y_dim
        return tf.reduce_sum(x*self.kernel, axis=1)


class Conditional_Prior(tf.keras.Model):
    def __init__(self, z_dim):
        super(Conditional_Prior, self).__init__()
        # Defining a dense layer will give many-to-one mapping from y to z. CCVAE used one-to-one mapping.
        self.loc_true = MyCondGenerationLayer(z_dim, initializer='zeros')
        self.loc_false = MyCondGenerationLayer(z_dim, initializer='zeros')
        self.scale_true = MyCondGenerationLayer(z_dim, initializer='ones')
        self.scale_false = MyCondGenerationLayer(z_dim, initializer='ones')

    def call(self, y, c):
        # Transpose c
        c = tf.transpose(c)
        locs = self.loc_true(y*c) + self.loc_false((1-y)*c)
        scale = self.scale_true(y*c) + self.scale_false((1-y)*c)

        # Apply softplus to scale and clamp to avoid numerical issues
        scale = tf.nn.softplus(scale)
        scale = tf.clip_by_value(scale, 1e-3, 1e3)
        return locs, scale