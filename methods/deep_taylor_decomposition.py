import tensorflow as tf
import tensorflow_datasets as tfds

class DTD:
    def __init__(self, sess, dataset):
        # Tensorflow session.
        self.sess = sess
        # Dataset.
        self.ds = dataset
        
        # Load model parameter for explanation.
        self.X = tf.get_collection('input')[0]
        self.pred = tf.get_collection('output')[0]        
        self.conv_w_1, self.conv_w_2, self.conv_w_3 = \
            tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='conv_params/w')
        self.fc_w_1, self.fc_w_2 = \
            tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='fc_params/w')
        self.activations = tf.get_collection(tf.GraphKeys.ACTIVATIONS)[0]

    
    def backprop_dense_z_plus(self, activation, kernel, relevance):
        W_p = tf.maximum(0., kernel)
        z = tf.matmul(activation, W_p) + 1e-9
        s = relevance / z
        c = tf.matmul(s, tf.transpose(W_p))
        return activation * c
        
        
    def backprop_dense_w_2(self, activation, kernel, relevance):
        W_2 = tf.square(kernel)
        W_sum = tf.reduce_sum(W_2, axis=0)
        W_n = W_2 / W_sum
        return tf.matmul(relevance, tf.transpose(W_n))

        
    def backprop_conv(self, activation, kernel, relevance, strides, padding='SAME'):
        W_p = tf.maximum(0., kernel)
        z = tf.nn.conv2d(activation, W_p, strides, padding) + 1e-10
        s = relevance / z
        c = tf.nn.conv2d_transpose(s, W_p, output_shape=tf.shape(activation), strides=strides, padding=padding)
        return activation * c
        
    
    def backprop_conv_input(self, X, kernel, relevance, strides, padding='SAME', lowest=0., highest=1.):
        W_p = tf.maximum(0., kernel)
        W_n = tf.minimum(0., kernel)

        L = tf.ones_like(X, tf.float32) * lowest
        H = tf.ones_like(X, tf.float32) * highest

        z_o = tf.nn.conv2d(X, kernel, strides, padding)
        z_p = tf.nn.conv2d(L, W_p, strides, padding)
        z_n = tf.nn.conv2d(H, W_n, strides, padding)

        z = z_o - z_p - z_n + 1e-10
        s = relevance / z

        # c_o = tf.nn.conv2d_backprop_input(tf.shape(X), kernel, s, strides, padding)
        # c_p = tf.nn.conv2d_backprop_input(tf.shape(X), W_p, s, strides, padding)
        # c_n = tf.nn.conv2d_backprop_input(tf.shape(X), W_n, s, strides, padding)
        c_o = tf.nn.conv2d_transpose(s, kernel, tf.shape(X), strides, padding)
        c_p = tf.nn.conv2d_transpose(s, W_p, tf.shape(X), strides, padding)
        c_n = tf.nn.conv2d_transpose(s, W_n, tf.shape(X), strides, padding)

        return X * c_o - L * c_p - H * c_n


    def run(self):
        # Initialize a list of relevance.
        Rs = []

        # Assign relevance to the output.
        cond = tf.equal(self.activations[-1], tf.reduce_max(self.activations[-1], axis=-1, keepdims=True))
        relevance = tf.where(cond, self.activations[-1], tf.zeros_like(self.activations[-1]))
        Rs.append(relevance)
        
        # fc_layer_2
        relevance = self.backprop_dense_z_plus(self.activations[-2], self.fc_w_2, Rs[-1])
        Rs.append(relevance)
        
        # fc_layer_1
        relevance = self.backprop_dense_z_plus(self.activations[-3], self.fc_w_1, Rs[-1])
        Rs.append(relevance)
        
        # Flatten
        _, rows, cols, channels = self.activations[-4].get_shape().as_list()
        relevance = tf.reshape(Rs[-1], [-1, rows, cols, channels])
        Rs.append(relevance)
        
        # conv_3
        relevance = self.backprop_conv(self.activations[-5], self.conv_w_3, Rs[-1], [1, 1, 1, 1])
        Rs.append(relevance)
        
        # conv_2
        relevance = self.backprop_conv(self.activations[-6], self.conv_w_2, Rs[-1], [1, 1, 1, 1])
        Rs.append(relevance)
        
        # conv_1
        relevance = self.backprop_conv_input(self.X, self.conv_w_1, Rs[-1], [1, 1, 1, 1])
        Rs.append(relevance)
        
        for idx, data in enumerate(tfds.as_numpy(self.ds)):
            feed_dict = {self.X: data['image']}
            Rs_results = self.sess.run(Rs, feed_dict=feed_dict)
            break

        return data['image'], Rs_results[-1]