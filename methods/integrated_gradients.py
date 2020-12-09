import tensorflow as tf
import tensorflow_datasets as tfds
import numpy as np

class IntegratedGradients:
    def __init__(self, sess, dataset, steps):
        # Tensorflow session.
        self.sess = sess
        # Dataset.
        self.ds = dataset
        # Hyperparameter.
        self.steps = steps
        
        # Load model parameter for explnation.
        self.X = tf.get_collection('input')[0]
        self.pred = tf.get_collection('output')[0]


    def run(self):
        for idx, data in enumerate(tfds.as_numpy(self.ds)):            
            # Gradient tensor.
            gradient = tf.gradients(self.pred, self.X)[0]
            
            # Baseline tensor: x'.
            baseline = tf.zeros_like(self.X)
            
            # Data point from the baseline to the input: x' + \alpha * (x - x').
            scaled_inputs = [baseline + float(i)/self.steps * (self.X - baseline) for i in range(self.steps + 1)]
            scaled_inputs = self.sess.run(scaled_inputs, feed_dict={self.X: data['image']})
            # (steps, batch, height, width, channel) -> (batch, steps, height, width, channel)
            scaled_inputs = np.transpose(scaled_inputs, (1, 0, 2, 3, 4))
            
            # (batch, steps, height, width, channel) -> (batch*steps, height, width, channel)
            scaled_shape = list(scaled_inputs.shape)
            reshaped_shape = [-1] + scaled_shape[2:]
            scaled_inputs = np.reshape(scaled_inputs, reshaped_shape)
            
            grads = self.sess.run(gradient, feed_dict={self.X: scaled_inputs})
            # (batch*steps, height, width, channel) -> (batch, steps, height, width, channel)
            grads = np.reshape(grads, scaled_shape)
            
            # Approximation of integral: Riemann sum.
            integrad = (grads[:, :-1] + grads[:, 1:]) / 2.0
            integrad = np.mean(integrad, axis=1)
            
            # (x - x') * \integral.
            integrad = (self.X - baseline) * integrad
            integrad = self.sess.run(integrad, feed_dict={self.X: data['image']})

            break

        return data['image'], integrad
