#!/usr/bin/env python3

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
from yolov2_ros.srv import *
import rospy
from copy import deepcopy
from core import YOLO
from vision_msgs.msg import Detection2DArray, Detection2D, BoundingBox2D, ObjectHypothesisWithPose
from geometry_msgs.msg import PoseWithCovariance, Pose2D
from std_msgs.msg import Header
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="0"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

class YoloServer(object):
    def __init__(self):
        self.bridge = CvBridge()

        self.n_gpu = rospy.get_param('~n_gpu', default=1)
        self.backend_vehicles = rospy.get_param('~backend_vehicles', default='Full Yolo')                          # Either 'tiny_yolo', full_yolo, 'mobile_net, 'squeeze_net', or 'inception3'
        self.backend_licence_plate = rospy.get_param('~backend_licence_plate', default='Full Yolo')                          # Either 'tiny_yolo', full_yolo, 'mobile_net, 'squeeze_net', or 'inception3'
        self.backend_path = rospy.get_param('~weights_path')                                     # Weights directory
        self.input_size = (rospy.get_param('~input_size_h', default=416),
                           rospy.get_param('~input_size_w', default=416))                        # DO NOT change this. 416 is default for YOLO.
        self.labels = rospy.get_param('/labels')                                                 # Eg: ['trafficcone', 'person', 'dog']
        self.iou_threshold = rospy.get_param('~iou_threshold', default=0.7)
        self.score_threshold = rospy.get_param('~score_threshold', default=0.5)
        self.max_number_detections = rospy.get_param('~max_number_detections', default=5)        # Max number of detections
        self.anchors = rospy.get_param('~anchors', default=[0.57273, 0.677385, 1.87446,          # The anchors to use. Use the anchor generator and copy these into the config.
                                                  2.06253, 3.33843, 5.47434, 7.88282, 
                                                  3.52778, 9.77052, 9.16828])
        self.weights_path = rospy.get_param('~weights_path', default='../weights/full_yolo.h5')   # Path to the weights.h5 file
        self.weight_file_vehicles = rospy.get_param('~weight_file_vehicles')
        self.weight_file_licence_plate = rospy.get_param('~weight_file_licence_plate')

        self.yolo_vehicles = YOLO(
            backend = self.backend_vehicles,
            input_size = self.input_size, 
            labels = self.labels, 
            anchors = self.anchors
        )

        self.yolo_vehicles.load_weights(self.weights_path + '/' + self.weight_file_vehicles)

        self.yolo_licence_plate = YOLO(
            backend = self.backend_licence_plate,
            input_size = self.input_size, 
            labels = ['licence_plate'], 
            anchors = self.anchors
        )

        self.yolo_licence_plate.load_weights(self.weights_path + '/' + self.weight_file_licence_plate)

        rospy.loginfo('YOLO detectors ready...')

        s = rospy.Service('yolo_detect', YoloDetect, self._handle_yolo_detect, buff_size=10000000)

        s.spin()

    def get_all_labels(self, box):
        return box.classes

    def get_xy_score(self, box):
        if box.xy_score == -1:
            box.xy_score = box.c[box.get_label()]

        return box.xy_score

    def get_xy_center(self, box):
        x = ((box.xmax - box.xmin)/2) + box.xmin
        y = ((box.ymax - box.ymin)/2) + box.ymin
        return x, y

    def get_xy_extents(self, box):
        x = box.xmax - box.xmin
        y = box.ymax - box.ymin
        return x, y

    def _handle_yolo_detect(self, req):
        cv_image = None
        detection_array = Detection2DArray()
        detections = []
        boxes = None
        
        try:
            cv_image = self.bridge.imgmsg_to_cv2(req.image, "8UC3")
            # Cropping image to the bottom part for licence plate detection
            cv_image_h = cv_image.shape[0]
            cv_image_w = cv_image.shape[1]
            #cv_image_h_offset = int(cv_image_h/2)
            #cv_image = cv_image[cv_image_h_offset:cv_image_h,0:cv_image_w]

        except CvBridgeError as e:
            rospy.logerr(e)
        try:
            #boxes = self.yolo_vehicles.predict(cv_image, self.iou_threshold, self.score_threshold)
            boxes = self.yolo_licence_plate.predict(cv_image, self.iou_threshold, self.score_threshold)
        except SystemError:
            pass
        #rospy.loginfo('Found {} boxes'.format(len(boxes)))
        for box in boxes:

            ymin_px = int(box.ymin*cv_image_h)
            ymax_px = int(box.ymax*cv_image_h)
            xmin_px = int(box.xmin*cv_image_w)
            xmax_px = int(box.xmax*cv_image_w)

            ymin_norm = box.ymin
            ymax_norm = box.ymax
            xmin_norm = box.xmin
            xmax_norm = box.xmax

#            cv_image_h_offset_norm = cv_image_h_offset/cv_image_h
            cv_image_h_offset_norm = 0

            # Correcting for offset
            box.ymin += cv_image_h_offset_norm
            box.ymax += cv_image_h_offset_norm

            #rospy.loginfo('ymin: %i, ymax: %i, xmin: %i, xmax: %i' % (ymin_px, ymax_px, xmin_px, xmax_px))

#            try:
#                rospy.loginfo(cv_image[ymin_px:ymax_px,xmin_px:xmax_px].shape)
#                lp_boxes = self.yolo_licence_plate.predict(cv_image[ymin_px:ymax_px,xmin_px:xmax_px], self.iou_threshold, self.score_threshold)
#
#                if len(lp_boxes) > 0:
#                    rospy.loginfo('Found a licence plate')
#
#                    #box.ymin += ymin_norm + cv_image_h_offset_norm
#                    #box.ymax += ymin_norm + cv_image_h_offset_norm
#                    box.ymin += ymin_norm
#                    box.ymax += ymin_norm
#                    box.xmin += xmin_norm
#                    box.xmax += xmin_norm
#
#                    box = lp_boxes[0] # The LP detection net should only detect up to 1 object
#                rospy.loginfo('ymin: %i, ymax: %i, xmin: %i, xmax: %i' % (box.ymin, box.ymax, box.xmin, box.xmax))
#            except SystemError:
#                pass

            detection = Detection2D()
            results = []
            bbox = BoundingBox2D()
            center = Pose2D()

            detection.header = Header()
            detection.header.stamp = rospy.get_rostime()
            # detection.source_img = deepcopy(req.image)

            labels = self.get_all_labels(box)
            for i in range(0,len(labels)):
                object_hypothesis = ObjectHypothesisWithPose()
                object_hypothesis.id = i
                object_hypothesis.score = labels[i]
                results.append(object_hypothesis)
            
            detection.results = results

            x, y = self.get_xy_center(box)
            center.x = x
            center.y = y
            center.theta = 0.0
            bbox.center = center

            size_x, size_y = self.get_xy_extents(box)
            bbox.size_x = size_x
            bbox.size_y = size_y

            detection.bbox = bbox
            #if len(lp_boxes) > 0:
            detections.append(detection)

        detection_array.header = Header()
        detection_array.header.stamp = rospy.get_rostime()
        detection_array.detections = detections

        return YoloDetectResponse(detection_array)

if __name__ == '__main__':
    rospy.init_node('yolo_server')
    
    try:
        ys = YoloServer()
    except rospy.ROSInterruptException:
        pass
