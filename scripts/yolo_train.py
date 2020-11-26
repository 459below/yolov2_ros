#!/usr/bin/env python3

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import numpy as np
import cv2
from tqdm import tqdm
import rospy
from core import YOLO, parse_annotation_xml

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="0,1"

class YoloTrain(object):
    def __init__(self):
        # Either 'tiny_yolo', full_yolo, 'mobile_net, 'squeeze_net', or 'inception3':
        self.backend = rospy.get_param('~backend', default='Full Yolo')
        self.input_size_w = rospy.get_param('~input_size_w', default=416)  # DO NOT change this. 416 is default for YOLO.
        self.input_size_h = rospy.get_param('~input_size_h', default=416)  # DO NOT change this. 416 is default for YOLO.
        
        # The anchors to use. Use the anchor generator and copy these into the config.
        self.anchors = rospy.get_param('~anchors', default=[0.57273, 0.677385, 1.87446, 
                                                  2.06253, 3.33843, 5.47434, 7.88282, 
                                                  3.52778, 9.77052, 9.16828])
        self.weights_path = rospy.get_param('~weights_path', default='../weights/full_yolo.h5')  # Path to the weights.h5 file
        self.weight_file = rospy.get_param('~weight_file')

        # Weight paths
        self.train_annot_folder = rospy.get_param('~train_annot_folder')
        self.train_image_folder = rospy.get_param('~train_image_folder')
        self.saved_weights_name = rospy.get_param('~saved_weights_name')

        # Train configuration
        self.labels = rospy.get_param('~labels')  # Eg: ['trafficcone', 'person', 'dog']
        self.train_times = rospy.get_param('~train_times', default=8)
        self.valid_times = rospy.get_param('~valid_times', default=1)
        self.nb_epochs = rospy.get_param('~nb_epochs', default=50)
        self.learning_rate = rospy.get_param('~learning_rate', default=0.0004)
        self.batch_size = rospy.get_param('~batch_size', default=16)
        self.warmup_epochs = rospy.get_param('~warmup_epochs', default=3)
        self.object_scale = rospy.get_param('~object_scale', default=5.0)
        self.no_object_scale = rospy.get_param('~no_object_scale', default=1.0)
        self.coord_scale = rospy.get_param('~coord_scale', default=1.0)
        self.class_scale = rospy.get_param('~class_scale', default=1.0)
        self.debug = rospy.get_param('~debug', default=True)

        # parse annotations of the training set
        #self.train_imgs, self.train_labels = parse_annotation_xml(
        #    self.train_annot_folder, 
        #    self.train_image_folder, 
        #    self.labels)
        self.train_imgs, self.train_labels = _parse_annotation_ccpd(self.train_image_folder)

        # parse annotations of the validation set, if any, otherwise split the training set
        if 'valid_annot_folder' in rospy.get_param_names():
            self.valid_annot_folder = rospy.get_param('~valid_annot_folder')
            self.valid_image_folder = rospy.get_param('~valid_image_folder')
            self.valid_imgs, self.valid_labels = parse_annotation_xml(
                self.valid_annot_folder, 
                self.valid_image_folder, 
                self.labels)
        else:
            train_valid_split = int(0.8*len(self.train_imgs))
            np.random.shuffle(self.train_imgs)

            self.valid_imgs = self.train_imgs[train_valid_split:]
            self.train_imgs = self.train_imgs[:train_valid_split]

        if len(self.labels) > 0:
            overlap_labels = set(self.labels).intersection(set(self.train_labels.keys()))

            rospy.loginfo('Seen labels: {}'.format(self.train_labels))
            rospy.loginfo('Given labels: {}'.format(self.labels))
            rospy.loginfo('Overlap labels: {}'.format(overlap_labels))

            if len(overlap_labels) < len(self.labels):
                rospy.signal_shutdown('Some labels have no annotations! Please revise the list of labels in the launch file!')
        else:
            rospy.loginfo('No labels are provided. Training on all seen labels.')
            self.labels = self.train_labels.keys()
        
        self.yolo = YOLO(
            backend = self.backend,
            input_size = (self.input_size_h, self.input_size_w), 
            labels = self.labels, 
            anchors = self.anchors
        )

        self.yolo.train(
            train_imgs = self.train_imgs,
            valid_imgs = self.valid_imgs,
            train_times =self.train_times,
            valid_times = self.valid_times,
            nb_epochs = self.nb_epochs, 
            learning_rate = self.learning_rate, 
            batch_size = self.batch_size,
            warmup_epochs = self.warmup_epochs,
            object_scale = self.object_scale,
            no_object_scale = self.no_object_scale,
            coord_scale = self.coord_scale,
            class_scale = self.class_scale,
            saved_weights_name = self.saved_weights_name,
            debug = self.debug)
        
        rospy.signal_shutdown('Completed training.')

def _parse_annotation_ccpd(img_dir):
    # This parser is utilized on CCPD dataset
    all_imgs = []
    seen_labels = {}

    img_files = os.listdir(img_dir)
    for img_file in tqdm(sorted(img_files)):

        img_array = cv2.imread(os.path.join(img_dir, img_file))
        ori_w, ori_h = [float(int(el)) for el in [img_array.shape[1], img_array.shape[0]]]

        iname = img_file.rsplit('/', 1)[-1].rsplit('.', 1)[0].split('-')
        [leftUp, rightDown] = [[int(eel) for eel in el.split('&')] for el in iname[2].split('_')]

        img = {}
        img['filename'] = os.path.join(img_dir, img_file)
        img['width'] = ori_w
        img['height'] = ori_h

        img['object'] = [{
            'name': 'license plate',
            'xmin': int(leftUp[0]),
            'ymin': int(leftUp[1]),
            'xmax': int(rightDown[0]),
            'ymax': int(rightDown[1])
            }]

        all_imgs += [img]

    return all_imgs, seen_labels

if __name__ == '__main__':
    rospy.init_node('yolov2_ros_train')

    try:
        yt = YoloTrain()
    except rospy.ROSInterruptException:
        pass
