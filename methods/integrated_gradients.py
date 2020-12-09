import tensorflow as tf
import tensorflow_datasets as tfds
import numpy as np

class IntegratedGradients:
    def __init__(self, sess, dataset, steps):
        self.sess = sess
        self.ds = dataset
        self.steps = steps
        
        self.X = tf.get_collection('input')[0]
        self.pred = tf.get_collection('output')[0]


    def run(self):
        for idx, data in enumerate(tfds.as_numpy(self.ds)):            
            gradient = tf.gradients(self.pred, self.X)[0]
            
            baseline = tf.zeros_like(self.X)
            scaled_inputs = [baseline + float(i)/self.steps * (self.X - baseline) for i in range(self.steps + 1)]
            scaled_inputs = self.sess.run(scaled_inputs, feed_dict={self.X: data['image']})
            scaled_inputs = np.transpose(scaled_inputs, (1, 0, 2, 3, 4))
            
            scaled_shape = list(scaled_inputs.shape)
            reshaped_shape = [-1] + scaled_shape[2:]
            scaled_inputs = np.reshape(scaled_inputs, reshaped_shape)
            
            grads = self.sess.run(gradient, feed_dict={self.X: scaled_inputs})
            grads = np.reshape(grads, scaled_shape)
            
            integrad = (grads[:, :-1] + grads[:, 1:]) / 2.0
            integrad = np.mean(integrad, axis=1)
            integrad = (self.X - baseline) * integrad
            
            integrad = self.sess.run(integrad, feed_dict={self.X: data['image']})

            break

        return data['image'], integrad
