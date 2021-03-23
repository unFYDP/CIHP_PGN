import os
import tensorflow as tf
from PIL import Image
from utils import *

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

N_CLASSES = 20
DATA_DIR = './datasets/CIHP'
LIST_PATH = './datasets/CIHP/list/val.txt'
DATA_ID_LIST = './datasets/CIHP/list/val_id.txt'
with open(DATA_ID_LIST, 'r') as f:
    NUM_STEPS = len(f.readlines())
RESTORE_FROM = './checkpoint/CIHP_pgn'


def main():
    """Create the model and start the evaluation process."""

    # Create queue coordinator.
    coord = tf.train.Coordinator()
    # Load reader.
    with tf.name_scope("create_inputs"):
        reader = ImageReader(DATA_DIR, LIST_PATH, DATA_ID_LIST, None, False, False, False, coord)
        image, label, edge_gt = reader.image, reader.label, reader.edge
        image_rev = tf.reverse(image, tf.stack([1]))
        image_list = reader.image_list

    image_batch = tf.stack([image, image_rev])
    label_batch = tf.expand_dims(label, dim=0)  # Add one batch dimension.
    edge_gt_batch = tf.expand_dims(edge_gt, dim=0)

    # Create network
    with tf.variable_scope('', reuse=False):
        net = PGNModel({'data': image_batch}, is_training=False, n_classes=N_CLASSES)

    # parsing net
    parsing_out1 = net.layers['parsing_fc']
    parsing_out2 = net.layers['parsing_rf_fc']

    # edge net
    edge_out2 = net.layers['edge_rf_fc']

    # combine resize
    parsing_out1 = tf.image.resize_images(parsing_out1, tf.shape(image_batch)[1: 3, ])
    parsing_out2 = tf.image.resize_images(parsing_out2, tf.shape(image_batch)[1: 3, ])

    edge_out2 = tf.image.resize_images(edge_out2, tf.shape(image_batch)[1: 3, ])

    raw_output = tf.reduce_mean(tf.stack([parsing_out1, parsing_out2]), axis=0)
    head_output, tail_output = tf.unstack(raw_output, num=2, axis=0)
    tail_list = tf.unstack(tail_output, num=20, axis=2)
    tail_list_rev = [None] * 20
    for xx in range(14):
        tail_list_rev[xx] = tail_list[xx]
    tail_list_rev[14] = tail_list[15]
    tail_list_rev[15] = tail_list[14]
    tail_list_rev[16] = tail_list[17]
    tail_list_rev[17] = tail_list[16]
    tail_list_rev[18] = tail_list[19]
    tail_list_rev[19] = tail_list[18]
    tail_output_rev = tf.stack(tail_list_rev, axis=2)
    tail_output_rev = tf.reverse(tail_output_rev, tf.stack([1]))

    raw_output_all = tf.reduce_mean(tf.stack([head_output, tail_output_rev]), axis=0)
    raw_output_all = tf.expand_dims(raw_output_all, dim=0)
    raw_output_all = tf.argmax(raw_output_all, axis=3)
    pred_all = tf.expand_dims(raw_output_all, dim=3)  # Create 4-d tensor.

    raw_edge = tf.reduce_mean(tf.stack([edge_out2]), axis=0)
    head_output, tail_output = tf.unstack(raw_edge, num=2, axis=0)
    tail_output_rev = tf.reverse(tail_output, tf.stack([1]))
    raw_edge_all = tf.reduce_mean(tf.stack([head_output, tail_output_rev]), axis=0)
    raw_edge_all = tf.expand_dims(raw_edge_all, dim=0)

    # Which variables to load.
    restore_var = tf.global_variables()
    # Set up tf session and initialize variables.
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    init = tf.global_variables_initializer()

    sess.run(init)
    sess.run(tf.local_variables_initializer())

    # Load weights.
    loader = tf.train.Saver(var_list=restore_var)
    if RESTORE_FROM is not None:
        if load(loader, sess, RESTORE_FROM):
            print(" [*] Load SUCCESS")
        else:
            print(" [!] Load failed...")

    # Start queue threads.
    threads = tf.train.start_queue_runners(coord=coord, sess=sess)

    # Iterate over training steps and output
    parsing_dir = './output'
    if not os.path.exists(parsing_dir):
        os.makedirs(parsing_dir)

    for step in range(NUM_STEPS):
        parsing_ = sess.run(pred_all)
        img_split = image_list[step].split('/')
        img_id = img_split[-1][:-4]

        msk = decode_labels(parsing_, num_classes=N_CLASSES)
        parsing_im = Image.fromarray(msk[0])
        parsing_im.save('{}/{}_vis.png'.format(parsing_dir, img_id))

    coord.request_stop()
    coord.join(threads)


if __name__ == '__main__':
    main()
