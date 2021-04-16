import os
import argparse
import tensorflow as tf
from PIL import Image
from glob import glob
from utils import *

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

N_CLASSES = 20
INPUT_RESIZE = 1080
BATCH_ALT_SIZES = [0.5, 0.75, 1.25, 1.50, 1.75]


def main(input_dir, output_dir, checkpoint_dir):
    # Create queue coordinator.
    coord = tf.train.Coordinator()

    # Load input
    input_files = sorted(glob(os.path.join(input_dir, '*')))
    input_queue = tf.train.slice_input_producer([tf.convert_to_tensor(input_files, dtype=tf.string)], shuffle=False)
    img_contents = tf.io.read_file(input_queue[0])
    img = tf.io.decode_jpeg(img_contents, channels=3)
    # Resize to prevent OOM
    img = tf.image.resize(img, [INPUT_RESIZE, INPUT_RESIZE], preserve_aspect_ratio=True)
    img_r, img_g, img_b = tf.split(value=img, num_or_size_splits=3, axis=2)
    image = tf.cast(tf.concat([img_b, img_g, img_r], 2), dtype=tf.float32)
    # TODO: Subtract by mean (see image_reader)
    image_rev = tf.reverse(image, tf.stack([1]))

    image_batch = tf.stack([image, image_rev])
    h_orig, w_orig = tf.to_float(tf.shape(image_batch)[1]), tf.to_float(tf.shape(image_batch)[2])

    # Additional batches for accuracy
    alt_image_batches = []
    for size in BATCH_ALT_SIZES:
        alt_image_batches.append(tf.image.resize_images(image_batch, tf.stack([tf.to_int32(tf.multiply(h_orig, size)), tf.to_int32(tf.multiply(w_orig, size))])))

    # Create networks
    with tf.variable_scope('', reuse=False):
        orig_net = PGNModel({'data': image_batch}, is_training=False, n_classes=N_CLASSES)

    alt_nets = []
    for batch in alt_image_batches:
        with tf.variable_scope('', reuse=True):
            alt_nets.append(PGNModel({'data': batch}, is_training=False, n_classes=N_CLASSES))

    # parsing nets
    parsing_out1s = [net.layers['parsing_fc'] for net in [orig_net] + alt_nets]
    parsing_out2s = [net.layers['parsing_rf_fc'] for net in [orig_net] + alt_nets]

    # edge nets
    edge_out2s = []
    for size, net in zip([1.] + BATCH_ALT_SIZES, [orig_net] + alt_nets):
        if size >= 1:
            edge_out2s.append(net.layers['edge_rf_fc'])

    # combine resize
    parsing_out1 = tf.reduce_mean(tf.stack([tf.image.resize_images(po, tf.shape(image_batch)[1:3, ]) for po in parsing_out1s]), axis=0)
    parsing_out2 = tf.reduce_mean(tf.stack([tf.image.resize_images(po, tf.shape(image_batch)[1:3, ]) for po in parsing_out2s]), axis=0)

    edge_out2s = tf.reduce_mean(tf.stack([tf.image.resize_images(eo, tf.shape(image_batch)[1:3, ]) for eo in edge_out2s]), axis=0)

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

    if not load(loader, sess, checkpoint_dir):
        raise IOError('Checkpoint loading failed')

    # Start queue threads.
    threads = tf.train.start_queue_runners(coord=coord, sess=sess)

    # Iterate over training steps and output
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for input_file in input_files:
        parsing_ = sess.run(pred_all)
        img_id = os.path.splitext(os.path.basename(input_file))[0]

        msk = decode_labels(parsing_, num_classes=N_CLASSES)
        parsing_im = Image.fromarray(msk[0])
        parsing_im.save('{}/{}.png'.format(output_dir, img_id))

    coord.request_stop()
    coord.join(threads)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--input_dir',
        '-i',
        type=str,
        default='datasets/CIHP/images',
        help='Input images directory')

    parser.add_argument(
        '--output_dir',
        '-o',
        type=str,
        default='output',
        help='Output directory for segmented masks')

    parser.add_argument(
        '--checkpoint_dir',
        '-c',
        type=str,
        default='checkpoint/CIHP_pgn',
        help='Checkpoints directory')

    args = parser.parse_args()
    main(args.input_dir, args.output_dir, args.checkpoint_dir)
