import tensorflow as tf
from nets import inception_resnet_v2
from preprocessing import inception_preprocessing
from scipy.misc import imshow, imread

slim = tf.contrib.slim

batch_size = 1 # 100 # The number of samples in each batch
checkpoint_path = '/home/raulgomez/datasets/WebVision/TF-logs-cluster/4/TF-logs/' # An absolute path to a checkpoint file
model_name = 'InceptionResnetV2'
image_size = 299
num_classes = 1000

with tf.Graph().as_default():
    with slim.arg_scope(inception_resnet_v2.inception_resnet_v2_arg_scope()):

        imgPath = '/home/raulgomez/datasets/WebVision/val_images_256/val000800.jpg'
        testImage_string = tf.gfile.FastGFile(imgPath, 'rb').read()
        testImage = tf.image.decode_jpeg(testImage_string, channels=3)
        processed_image = inception_preprocessing.preprocess_image(testImage, image_size, image_size, is_training=False)
        processed_images = tf.expand_dims(processed_image, 0)

        logits, _ = inception_resnet_v2.inception_resnet_v2(processed_images, num_classes=num_classes, is_training=False)
        probabilities = tf.nn.softmax(logits)
        print(checkpoint_path)
        checkpoint_path = tf.train.latest_checkpoint(checkpoint_path)
        print(checkpoint_path)
        init_fn = slim.assign_from_checkpoint_fn(checkpoint_path, slim.get_model_variables(model_name))

        with tf.Session() as sess:
            init_fn(sess)

            np_image, probabilities = sess.run([processed_images, probabilities])
            probabilities = probabilities[0, 0:]
            sorted_inds = [i[0] for i in sorted(enumerate(-probabilities), key=lambda x: x[1])]

            for i in range(1):
                index = sorted_inds[i]
                print((probabilities[index], index))
        imshow(imread(imgPath))