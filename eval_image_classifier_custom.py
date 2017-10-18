import tensorflow as tf
from nets import inception_resnet_v2
from preprocessing import inception_preprocessing
from scipy.misc import imshow, imread
import numpy as np
from PIL import Image
from get_144_crops import get_144_crops
import os

slim = tf.contrib.slim

batch_size = 144 # 100 # The number of samples in each batch
checkpoint_path = '../../datasets/WebVision/TF-logs-cluster/18/TF-logs/' # An absolute path to a checkpoint file
model_name = 'InceptionResnetV2'
image_size = 299
num_classes = 1000
split = 'val'
test = np.loadtxt('../../datasets/WebVision/info/'+split+'_filelist.txt', dtype=str)
num_crops = 144

output_file_dir = '../../datasets/WebVision/results/144Crops_day18'
if not os.path.exists(output_file_dir):
    os.makedirs(output_file_dir)
output_file_path = output_file_dir + '/'+split+'.txt'
output_file = open(output_file_path, "w")

with tf.Graph().as_default():
    with slim.arg_scope(inception_resnet_v2.inception_resnet_v2_arg_scope()):

        print 'Creating graph  ...'

        crops_input = tf.placeholder(tf.float32, shape=(batch_size, image_size, image_size, 3))
        crops_input_unstacked = tf.unstack(crops_input)

        crops_4_tf = []
        for crop in crops_input_unstacked:
            # testImage = tf.image.decode_jpeg(im.getdata(), channels=3)
            # data_tf = tf.convert_to_tensor(crop)
            processed_image = inception_preprocessing.preprocess_image(crop, image_size, image_size, is_training=False)
            crops_4_tf.append(processed_image)

        batch = tf.stack(crops_4_tf)
        logits, _ = inception_resnet_v2.inception_resnet_v2(batch, num_classes=num_classes, is_training=False)
        probabilities = tf.nn.softmax(logits)
        checkpoint_path = tf.train.latest_checkpoint(checkpoint_path)
        init_fn = slim.assign_from_checkpoint_fn(checkpoint_path, slim.get_model_variables(model_name))

    with tf.Session() as sess:
        init_fn(sess)

        print 'Computing ...'
        count = 0
        im_num = 0
        while im_num < len(test):

            # Load image
            if split == 'test':
                filename = '../../datasets/WebVision/' + split + '_images_256/' + test[im_num]
            else:
                filename = '../../datasets/WebVision/' + split + '_images_256/' + test[im_num][0]
            im = Image.open(filename)
            print(test[im_num])

            # Turn grayscale images to 3 channels
            if (im.size.__len__() == 2):
                im_gray = im
                im = Image.new("RGB", im_gray.size)
                im.paste(im_gray)

            # Get 144 crops
            crops = get_144_crops(im,image_size)
            crops_np = []

            # Transform to np arrays
            for crop in crops:
                crops_np.append(np.array(crop, np.float32)/225)


            probs = sess.run([probabilities], feed_dict={crops_input:crops_np})
            probs = probs[0]

            # Get mean probs over crops
            avgProbs = np.zeros(num_classes)
            for l in range(0,num_classes):
                avgProbs[l] = np.mean(probs[:,l])

            crop_probs = avgProbs
            sorted_inds = [i[0] for i in sorted(enumerate(-avgProbs), key=lambda x: x[1])]
            top5str = ''
            for c in range(5):
                index = sorted_inds[c]
                top5str = top5str + ' ' + str(index)
                # print((avgProbs[index], index))

            output_file.write(test[im_num][0] + top5str + '\n')


            im_num += 1

            # if im_num == 3: break

output_file.close()

print "DONE"
print output_file_dir
