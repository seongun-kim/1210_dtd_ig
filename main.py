import os
import tensorflow as tf
import argparse
from model import Model
import utils

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

os.environ['CUDA_VISIBLE_DEVICES'] = '2'

def main():
    parser = argparse.ArgumentParser(description='DTD & Integrated Gradients Tutorials')
    parser.add_argument('--train', action='store_true', help='true if training')
    parser.add_argument('--test', action='store_true', help='true if testing')
    parser.add_argument('--explain', action='store_true', help='true if explaining')
    parser.add_argument('--method', type=str, choices=['dtd', 'integrated'], default='dtd')
    parser.add_argument('--visualize', type=int, default=50, help='number of heatmaps to visualize')

    args = parser.parse_args()

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    model = Model(sess)

    if args.train:
        model.train()
    if args.test:
        model.test()
    if args.explain:
        images, heatmaps = model.explain(method=args.method, num_visualize=args.visualize)
        for i in range(args.visualize):
            fname = 'heatmap_{}'.format(i)
            utils.visualize_heatmap(fname, images[i], heatmaps[i])

if __name__ == '__main__':
    main()