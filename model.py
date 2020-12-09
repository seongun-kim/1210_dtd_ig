import os
import tensorflow as tf
import tensorflow_datasets as tfds
from pprint import pprint
import numpy as np
from methods.deep_taylor_decomposition import DTD
from methods.integrated_gradients import IntegratedGradients

class Model:
    def __init__(self, sess, batch_size=512, lr=0.001, num_epoch=10, clip_norm=5.0):
        # Model parameters.
        self.sess = sess
        self.dim_input = [None, 28, 28, 1]
        self.dim_output = [None, 10]

        # Hyperparameters.
        self.batch_size = batch_size
        self.lr = lr
        self.num_epoch = num_epoch
        self.clip_norm = clip_norm

        # Build neural network model.
        self._build_model()

        # Build summary for TensorBoard.
        self._build_summary()

        # Load MNIST dataset.
        self._build_dataset()

        # Initializer TF saver.
        self.saver = tf.train.Saver()


    def _build_model(self):
        def _conv2d(X, W, b, strides=[1, 1, 1, 1], name='conv'):
            return tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(X, W, strides=strides, padding='SAME', name=name), b))

        def _init_weights(shape, name):
            return tf.get_variable(name=name, shape=shape, initializer=tf.random_normal_initializer(0., 0.01), trainable=True)

        def _init_bias(shape, name='bias'):
            return tf.get_variable(name=name, shape=shape, initializer=tf.zeros_initializer(), trainable=True)

        self.X = tf.placeholder('float', self.dim_input, name='input')
        self.label = tf.placeholder('float', self.dim_output, name='label')

        with tf.variable_scope('conv_params'):
            self.W_1 = _init_weights([5, 5, 1, 10], name='w1')
            self.W_2 = _init_weights([5, 5, 10, 10], name='w2')
            self.W_3 = _init_weights([5, 5, 10, 10], name='w3')

            self.b_1 = _init_bias([10], name='b1')
            self.b_2 = _init_bias([10], name='b2')
            self.b_3 = _init_bias([10], name='b3')

        with tf.variable_scope('activations', reuse=tf.AUTO_REUSE):
            self.conv_1 = _conv2d(self.X, self.W_1, self.b_1, strides=[1, 1, 1, 1], name='conv_1')
            self.conv_2 = _conv2d(self.conv_1, self.W_2, self.b_2, strides=[1, 1, 1, 1], name='conv_2')
            self.conv_3 = _conv2d(self.conv_2, self.W_3, self.b_3, strides=[1, 1, 1, 1], name='conv_3')

            _, num_rows, num_cols, num_channels = self.conv_3.get_shape().as_list()
            self.flatten = tf.reshape(self.conv_3, [-1, num_rows*num_cols*num_channels])

        with tf.variable_scope('fc_params'):
            self.W_4 = _init_weights([num_rows*num_cols*num_channels, 40], name='w4')
            self.W_5 = _init_weights([40, 10], name='w5')

            self.b_4 = _init_bias([40], name='b4')
            self.b_5 = _init_bias([10], name='b5')

        with tf.variable_scope('activations', reuse=tf.AUTO_REUSE):
            self.fc_1 = tf.nn.relu(tf.matmul(self.flatten, self.W_4, name='dense_1') + self.b_4)
            self.pred = tf.matmul(self.fc_1, self.W_5, name='dense_2') + self.b_5

        tf.add_to_collection(tf.GraphKeys.ACTIVATIONS, [self.conv_1, self.conv_2, self.conv_3, \
                                                        self.flatten, self.fc_1, self.pred])
        tf.add_to_collection('input', self.X)
        tf.add_to_collection('output', self.pred)


    def _build_summary(self):
        self.global_step = tf.train.get_or_create_global_step()
        self.sess.run(self.global_step.initializer)

        train_fname = './logs/train/'
        test_fname = './logs/test/'
        self.train_writer = tf.summary.FileWriter(train_fname, self.sess.graph)
        self.test_writer = tf.summary.FileWriter(test_fname, self.sess.graph)


    def _build_dataset(self):
        def _one_hot(x):
            label = tf.one_hot(x['label'], 10)
            x['label'] = label
            return x

        self.ds_train = tfds.load('mnist', split='train', shuffle_files=True)
        self.ds_train = self.ds_train.map(_one_hot)
        self.ds_test = tfds.load('mnist', split='test', shuffle_files=False)
        self.ds_test = self.ds_test.map(_one_hot)


    def _save(self, save_path):
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        save_path += 'mnist'
        self.saver.save(self.sess, save_path, global_step=self.global_step.eval(session=self.sess), \
                        write_meta_graph=True)
        print('\n---------------')
        print('Checkpoint is saved to {}.\n'.format(save_path))


    def _restore(self, ckpt_path):
        if not hasattr(self, 'saver'):
            self.saver = tf.train.Saver()
        self.saver.restore(self.sess, tf.train.latest_checkpoint(ckpt_path))
        print('\n---------------')
        print('Checkpoint at {} is restored.\n'.format(ckpt_path))


    def train(self):
        # Batch train and test dataset.
        self.ds_train = self.ds_train.batch(self.batch_size)
        self.ds_test = self.ds_test.batch(self.batch_size)

        # Loss operation: MSE loss.
        loss_op = tf.losses.mean_squared_error(self.label, self.pred)
        loss_hist = tf.summary.scalar('loss', loss_op)

        # Accuracy of prediction.
        correct = tf.equal(tf.argmax(self.label, 1), tf.argmax(self.pred, 1))
        accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))
        accuracy_hist = tf.summary.scalar('accuracy', accuracy)

        # Use Adam optimizer
        optimizer = tf.train.AdamOptimizer(learning_rate=self.lr)
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)

        # Update operation: Clip gradient by global norm.
        with tf.control_dependencies(update_ops):
            gvs = optimizer.compute_gradients(loss_op)
            vars = [x[1] for x in gvs]
            gradients = [x[0] for x in gvs]
            clipped, global_norm = tf.clip_by_global_norm(gradients, self.clip_norm)
            train_op = optimizer.apply_gradients(zip(clipped, vars), global_step=self.global_step)
            global_norm_hist = tf.summary.scalar('global_norm', global_norm)

        merged = tf.summary.merge_all()

        # Initializer global variables.
        init_op = tf.global_variables_initializer()
        self.sess.run(init_op)

        # Iterate training epoch.
        for epoch in range(self.num_epoch):
            for idx, data in enumerate(tfds.as_numpy(self.ds_train)):
                feed_dict = {self.X: data['image'], self.label: data['label']}
                loss, _ = self.sess.run([loss_op, train_op], feed_dict=feed_dict)

                summary = self.sess.run(merged, feed_dict=feed_dict)
                self.train_writer.add_summary(summary, self.global_step.eval(session=self.sess))

            for idx, data in enumerate(tfds.as_numpy(self.ds_test)):
                feed_dict = {self.X: data['image'], self.label: data['label']}
                summary = self.sess.run(merged, feed_dict=feed_dict)
                self.test_writer.add_summary(summary, self.global_step.eval(session=self.sess))

            print('\nCheckpoint at epoch {}'.format(epoch+1))
            self._save('./checkpoint/')


    def test(self):
        # Restore the latest model.
        self._restore('./checkpoint/')

        # Accuracy of the prediction.
        correct = tf.equal(tf.argmax(self.label, 1), tf.argmax(self.pred, 1))
        accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))

        # Batch train and test dataset.
        self.ds_test = self.ds_test.batch(self.batch_size)

        # Measure the accuracy.
        correct_list = []
        for idx, data in enumerate(tfds.as_numpy(self.ds_test)):
            feed_dict = {self.X: data['image'], self.label: data['label']}
            acc, corr = self.sess.run([accuracy, correct], feed_dict=feed_dict)
            correct_list.append(corr)
            #print('accuracy: {}'.format(acc))
        correct = np.concatenate(correct_list).ravel()
        print('\n---------------')
        print('Accuracy: {:.2f}\n'.format(np.mean(correct)*100))


    def explain(self, method='dtd', num_visualize=50):
        self._restore('./checkpoint')
        batch_size = num_visualize

        # Make sure the dataset is unbatched before batch.
        iterator = self.ds_test.make_initializable_iterator()
        batch_data = iterator.get_next()
        self.sess.run(iterator.initializer)
        cur_batch_size = self.sess.run(batch_data['image']).shape[0]
        if cur_batch_size != 28:
            self.ds_test = self.ds_test.apply(tf.data.experimental.unbatch())

        # Batch test dataset.
        self.ds_test = self.ds_test.batch(batch_size)

        # Deep Taylor Decomposition
        if method == 'dtd':
            dtd = DTD(self.sess, self.ds_test)
            images, heatmaps = dtd.run()
        
        # Integrated Gradients
        elif method == 'integrated':
            steps = 100
            integrated = IntegratedGradients(self.sess, self.ds_test, steps)
            images, heatmaps = integrated.run()

        return images, heatmaps