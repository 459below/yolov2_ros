import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.applications import InceptionV3
from tensorflow.keras.applications.mobilenet import MobileNet
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.layers import Activation, Conv2D, Input, MaxPooling2D, BatchNormalization, Lambda, LeakyReLU, \
    concatenate

# Setting memory to dynamically grow
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
  try:
    for gpu in gpus:
      tf.config.experimental.set_memory_growth(gpu, True)
  except RuntimeError as e:
    print(e)

base_path = './backend_weights/'  # FIXME :: use environment variables

FULL_YOLO_V3_BACKEND_PATH = base_path + "full_yolo_v3_backend.h5"  # should be hosted on a server
FULL_YOLO_BACKEND_PATH = base_path + "full_yolo_backend.h5"  # should be hosted on a server
TINY_YOLO_BACKEND_PATH = base_path + "tiny_yolo_backend.h5"  # should be hosted on a server
SQUEEZENET_BACKEND_PATH = base_path + "squeezenet_backend.h5"  # should be hosted on a server
MOBILENET_BACKEND_PATH = base_path + "mobilenet_backend.h5"  # should be hosted on a server
INCEPTION3_BACKEND_PATH = base_path + "inception_backend.h5"  # should be hosted on a server
VGG16_BACKEND_PATH = base_path + "vgg16_backend.h5"  # should be hosted on a server
RESNET50_BACKEND_PATH = base_path + "resnet50_backend.h5"  # should be hosted on a server


class BaseFeatureExtractor(object):
    """docstring for ClassName"""

    # to be defined in each subclass
    def __init__(self, input_size):
        raise NotImplementedError("error message")

    # to be defined in each subclass
    def normalize(self, image):
        raise NotImplementedError("error message")

    def get_output_shape(self):
        return self.feature_extractor.output_shape[1:3]

    def extract(self, input_image):
        return self.feature_extractor(input_image)

class FullYoloV3Feature(BaseFeatureExtractor):
    """docstring for ClassName"""

    def __init__(self, input_size):
        input_image = Input(shape=(None, None, 3))
    
        # Layer  0 => 4
        x = self._conv_block(input_image, [{'filter': 32, 'kernel': 3, 'stride': 1, 'bnorm': True, 'leaky': True, 'layer_idx': 0},
                                      {'filter': 64, 'kernel': 3, 'stride': 2, 'bnorm': True, 'leaky': True, 'layer_idx': 1},
                                      {'filter': 32, 'kernel': 1, 'stride': 1, 'bnorm': True, 'leaky': True, 'layer_idx': 2},
                                      {'filter': 64, 'kernel': 3, 'stride': 1, 'bnorm': True, 'leaky': True, 'layer_idx': 3}])
    
        # Layer  5 => 8
        x = self._conv_block(x, [{'filter': 128, 'kernel': 3, 'stride': 2, 'bnorm': True, 'leaky': True, 'layer_idx': 5},
                            {'filter':  64, 'kernel': 1, 'stride': 1, 'bnorm': True, 'leaky': True, 'layer_idx': 6},
                            {'filter': 128, 'kernel': 3, 'stride': 1, 'bnorm': True, 'leaky': True, 'layer_idx': 7}])
    
        # Layer  9 => 11
        x = self._conv_block(x, [{'filter':  64, 'kernel': 1, 'stride': 1, 'bnorm': True, 'leaky': True, 'layer_idx': 9},
                            {'filter': 128, 'kernel': 3, 'stride': 1, 'bnorm': True, 'leaky': True, 'layer_idx': 10}])
    
        # Layer 12 => 15
        x = self._conv_block(x, [{'filter': 256, 'kernel': 3, 'stride': 2, 'bnorm': True, 'leaky': True, 'layer_idx': 12},
                            {'filter': 128, 'kernel': 1, 'stride': 1, 'bnorm': True, 'leaky': True, 'layer_idx': 13},
                            {'filter': 256, 'kernel': 3, 'stride': 1, 'bnorm': True, 'leaky': True, 'layer_idx': 14}])
    
        # Layer 16 => 36
        for i in range(7):
            x = self._conv_block(x, [{'filter': 128, 'kernel': 1, 'stride': 1, 'bnorm': True, 'leaky': True, 'layer_idx': 16+i*3},
                                {'filter': 256, 'kernel': 3, 'stride': 1, 'bnorm': True, 'leaky': True, 'layer_idx': 17+i*3}])
            
        skip_36 = x
            
        # Layer 37 => 40
        x = self._conv_block(x, [{'filter': 512, 'kernel': 3, 'stride': 2, 'bnorm': True, 'leaky': True, 'layer_idx': 37},
                            {'filter': 256, 'kernel': 1, 'stride': 1, 'bnorm': True, 'leaky': True, 'layer_idx': 38},
                            {'filter': 512, 'kernel': 3, 'stride': 1, 'bnorm': True, 'leaky': True, 'layer_idx': 39}])
    
        # Layer 41 => 61
        for i in range(7):
            x = self._conv_block(x, [{'filter': 256, 'kernel': 1, 'stride': 1, 'bnorm': True, 'leaky': True, 'layer_idx': 41+i*3},
                                {'filter': 512, 'kernel': 3, 'stride': 1, 'bnorm': True, 'leaky': True, 'layer_idx': 42+i*3}])
            
        skip_61 = x
            
        # Layer 62 => 65
        x = self._conv_block(x, [{'filter': 1024, 'kernel': 3, 'stride': 2, 'bnorm': True, 'leaky': True, 'layer_idx': 62},
                            {'filter':  512, 'kernel': 1, 'stride': 1, 'bnorm': True, 'leaky': True, 'layer_idx': 63},
                            {'filter': 1024, 'kernel': 3, 'stride': 1, 'bnorm': True, 'leaky': True, 'layer_idx': 64}])
    
        # Layer 66 => 74
        for i in range(3):
            x = self._conv_block(x, [{'filter':  512, 'kernel': 1, 'stride': 1, 'bnorm': True, 'leaky': True, 'layer_idx': 66+i*3},
                                {'filter': 1024, 'kernel': 3, 'stride': 1, 'bnorm': True, 'leaky': True, 'layer_idx': 67+i*3}])
            
        # Layer 75 => 79
        x = self._conv_block(x, [{'filter':  512, 'kernel': 1, 'stride': 1, 'bnorm': True, 'leaky': True, 'layer_idx': 75},
                            {'filter': 1024, 'kernel': 3, 'stride': 1, 'bnorm': True, 'leaky': True, 'layer_idx': 76},
                            {'filter':  512, 'kernel': 1, 'stride': 1, 'bnorm': True, 'leaky': True, 'layer_idx': 77},
                            {'filter': 1024, 'kernel': 3, 'stride': 1, 'bnorm': True, 'leaky': True, 'layer_idx': 78},
                            {'filter':  512, 'kernel': 1, 'stride': 1, 'bnorm': True, 'leaky': True, 'layer_idx': 79}], skip=False)
    
        # Layer 80 => 82
        yolo_82 = self._conv_block(x, [{'filter': 1024, 'kernel': 3, 'stride': 1, 'bnorm': True,  'leaky': True,  'layer_idx': 80},
                                  {'filter':  255, 'kernel': 1, 'stride': 1, 'bnorm': False, 'leaky': False, 'layer_idx': 81}], skip=False)
    
        # Layer 83 => 86
        x = self._conv_block(x, [{'filter': 256, 'kernel': 1, 'stride': 1, 'bnorm': True, 'leaky': True, 'layer_idx': 84}], skip=False)
        x = UpSampling2D(2)(x)
        x = concatenate([x, skip_61])
    
        # Layer 87 => 91
        x = self._conv_block(x, [{'filter': 256, 'kernel': 1, 'stride': 1, 'bnorm': True, 'leaky': True, 'layer_idx': 87},
                            {'filter': 512, 'kernel': 3, 'stride': 1, 'bnorm': True, 'leaky': True, 'layer_idx': 88},
                            {'filter': 256, 'kernel': 1, 'stride': 1, 'bnorm': True, 'leaky': True, 'layer_idx': 89},
                            {'filter': 512, 'kernel': 3, 'stride': 1, 'bnorm': True, 'leaky': True, 'layer_idx': 90},
                            {'filter': 256, 'kernel': 1, 'stride': 1, 'bnorm': True, 'leaky': True, 'layer_idx': 91}], skip=False)
    
        # Layer 92 => 94
        yolo_94 = self._conv_block(x, [{'filter': 512, 'kernel': 3, 'stride': 1, 'bnorm': True,  'leaky': True,  'layer_idx': 92},
                                  {'filter': 255, 'kernel': 1, 'stride': 1, 'bnorm': False, 'leaky': False, 'layer_idx': 93}], skip=False)
    
        # Layer 95 => 98
        x = self._conv_block(x, [{'filter': 128, 'kernel': 1, 'stride': 1, 'bnorm': True, 'leaky': True,   'layer_idx': 96}], skip=False)
        x = UpSampling2D(2)(x)
        x = concatenate([x, skip_36])
    
        # Layer 99 => 106
        yolo_106 = self._conv_block(x, [{'filter': 128, 'kernel': 1, 'stride': 1, 'bnorm': True,  'leaky': True,  'layer_idx': 99},
                                   {'filter': 256, 'kernel': 3, 'stride': 1, 'bnorm': True,  'leaky': True,  'layer_idx': 100},
                                   {'filter': 128, 'kernel': 1, 'stride': 1, 'bnorm': True,  'leaky': True,  'layer_idx': 101},
                                   {'filter': 256, 'kernel': 3, 'stride': 1, 'bnorm': True,  'leaky': True,  'layer_idx': 102},
                                   {'filter': 128, 'kernel': 1, 'stride': 1, 'bnorm': True,  'leaky': True,  'layer_idx': 103},
                                   {'filter': 256, 'kernel': 3, 'stride': 1, 'bnorm': True,  'leaky': True,  'layer_idx': 104},
                                   {'filter': 255, 'kernel': 1, 'stride': 1, 'bnorm': False, 'leaky': False, 'layer_idx': 105}], skip=False)
    
        self.feature_extractor = Model(input_image, [yolo_82, yolo_94, yolo_106], name='Full_YOLOV3_backend')
        try:
            print("Loading pretrained weights: " + FULL_YOLO_V3_BACKEND_PATH)
            self.feature_extractor.load_weights(FULL_YOLO_V3_BACKEND_PATH)
        except:
            print("Unable to load backend weights. Using a fresh model")

    def normalize(self, image):
        return image / 255.

    def _conv_block(inp, convs, skip=True):
        print(type(convs))
        x = inp
        count = 0
        print(len(convs.shape))
        for idx_conv in range(len(convs.shape)):
            conv = convs[idx_conv]
            if count == (len(convs.shape) - 2) and skip:
                skip_connection = x
            count += 1
            
            if conv['stride'] > 1: x = ZeroPadding2D(((1,0),(1,0)))(x) # peculiar padding as darknet prefer left and top
            x = Conv2D(conv['filter'], 
                       conv['kernel'], 
                       strides=conv['stride'], 
                       padding='valid' if conv['stride'] > 1 else 'same', # peculiar padding as darknet prefer left and top
                       name='conv_' + str(conv['layer_idx']), 
                       use_bias=False if conv['bnorm'] else True)(x)
            if conv['bnorm']: x = BatchNormalization(epsilon=0.001, name='bnorm_' + str(conv['layer_idx']))(x)
            if conv['leaky']: x = LeakyReLU(alpha=0.1, name='leaky_' + str(conv['layer_idx']))(x)
    
        return add([skip_connection, x]) if skip else x


class FullYoloFeature(BaseFeatureExtractor):
    """docstring for ClassName"""

    def __init__(self, input_size):
        input_image = Input(shape=input_size)

        # the function to implement the orgnization layer (thanks to github.com/allanzelener/YAD2K)
        def space_to_depth_x2(x):
            return tf.nn.space_to_depth(x, block_size=2)

        # Layer 1
        x = Conv2D(32, (3, 3), strides=(1, 1), padding='same', name='conv_1', use_bias=False)(input_image)
        x = BatchNormalization(name='norm_1')(x)
        x = LeakyReLU(alpha=0.1)(x)
        x = MaxPooling2D(pool_size=(2, 2))(x)

        # Layer 2
        x = Conv2D(64, (3, 3), strides=(1, 1), padding='same', name='conv_2', use_bias=False)(x)
        x = BatchNormalization(name='norm_2')(x)
        x = LeakyReLU(alpha=0.1)(x)
        x = MaxPooling2D(pool_size=(2, 2))(x)

        # Layer 3
        x = Conv2D(128, (3, 3), strides=(1, 1), padding='same', name='conv_3', use_bias=False)(x)
        x = BatchNormalization(name='norm_3')(x)
        x = LeakyReLU(alpha=0.1)(x)

        # Layer 4
        x = Conv2D(64, (1, 1), strides=(1, 1), padding='same', name='conv_4', use_bias=False)(x)
        x = BatchNormalization(name='norm_4')(x)
        x = LeakyReLU(alpha=0.1)(x)

        # Layer 5
        x = Conv2D(128, (3, 3), strides=(1, 1), padding='same', name='conv_5', use_bias=False)(x)
        x = BatchNormalization(name='norm_5')(x)
        x = LeakyReLU(alpha=0.1)(x)
        x = MaxPooling2D(pool_size=(2, 2))(x)

        # Layer 6
        x = Conv2D(256, (3, 3), strides=(1, 1), padding='same', name='conv_6', use_bias=False)(x)
        x = BatchNormalization(name='norm_6')(x)
        x = LeakyReLU(alpha=0.1)(x)

        # Layer 7
        x = Conv2D(128, (1, 1), strides=(1, 1), padding='same', name='conv_7', use_bias=False)(x)
        x = BatchNormalization(name='norm_7')(x)
        x = LeakyReLU(alpha=0.1)(x)

        # Layer 8
        x = Conv2D(256, (3, 3), strides=(1, 1), padding='same', name='conv_8', use_bias=False)(x)
        x = BatchNormalization(name='norm_8')(x)
        x = LeakyReLU(alpha=0.1)(x)
        x = MaxPooling2D(pool_size=(2, 2))(x)

        # Layer 9
        x = Conv2D(512, (3, 3), strides=(1, 1), padding='same', name='conv_9', use_bias=False)(x)
        x = BatchNormalization(name='norm_9')(x)
        x = LeakyReLU(alpha=0.1)(x)

        # Layer 10
        x = Conv2D(256, (1, 1), strides=(1, 1), padding='same', name='conv_10', use_bias=False)(x)
        x = BatchNormalization(name='norm_10')(x)
        x = LeakyReLU(alpha=0.1)(x)

        # Layer 11
        x = Conv2D(512, (3, 3), strides=(1, 1), padding='same', name='conv_11', use_bias=False)(x)
        x = BatchNormalization(name='norm_11')(x)
        x = LeakyReLU(alpha=0.1)(x)

        # Layer 12
        x = Conv2D(256, (1, 1), strides=(1, 1), padding='same', name='conv_12', use_bias=False)(x)
        x = BatchNormalization(name='norm_12')(x)
        x = LeakyReLU(alpha=0.1)(x)

        # Layer 13
        x = Conv2D(512, (3, 3), strides=(1, 1), padding='same', name='conv_13', use_bias=False)(x)
        x = BatchNormalization(name='norm_13')(x)
        x = LeakyReLU(alpha=0.1)(x)

        skip_connection = x

        x = MaxPooling2D(pool_size=(2, 2))(x)

        # Layer 14
        x = Conv2D(1024, (3, 3), strides=(1, 1), padding='same', name='conv_14', use_bias=False)(x)
        x = BatchNormalization(name='norm_14')(x)
        x = LeakyReLU(alpha=0.1)(x)

        # Layer 15
        x = Conv2D(512, (1, 1), strides=(1, 1), padding='same', name='conv_15', use_bias=False)(x)
        x = BatchNormalization(name='norm_15')(x)
        x = LeakyReLU(alpha=0.1)(x)

        # Layer 16
        x = Conv2D(1024, (3, 3), strides=(1, 1), padding='same', name='conv_16', use_bias=False)(x)
        x = BatchNormalization(name='norm_16')(x)
        x = LeakyReLU(alpha=0.1)(x)

        # Layer 17
        x = Conv2D(512, (1, 1), strides=(1, 1), padding='same', name='conv_17', use_bias=False)(x)
        x = BatchNormalization(name='norm_17')(x)
        x = LeakyReLU(alpha=0.1)(x)

        # Layer 18
        x = Conv2D(1024, (3, 3), strides=(1, 1), padding='same', name='conv_18', use_bias=False)(x)
        x = BatchNormalization(name='norm_18')(x)
        x = LeakyReLU(alpha=0.1)(x)

        # Layer 19
        x = Conv2D(1024, (3, 3), strides=(1, 1), padding='same', name='conv_19', use_bias=False)(x)
        x = BatchNormalization(name='norm_19')(x)
        x = LeakyReLU(alpha=0.1)(x)

        # Layer 20
        x = Conv2D(1024, (3, 3), strides=(1, 1), padding='same', name='conv_20', use_bias=False)(x)
        x = BatchNormalization(name='norm_20')(x)
        x = LeakyReLU(alpha=0.1)(x)

        # Layer 21
        skip_connection = Conv2D(64, (1, 1), strides=(1, 1), padding='same', name='conv_21', use_bias=False)(
            skip_connection)
        skip_connection = BatchNormalization(name='norm_21')(skip_connection)
        skip_connection = LeakyReLU(alpha=0.1)(skip_connection)
        skip_connection = Lambda(space_to_depth_x2)(skip_connection)

        x = concatenate([skip_connection, x])

        # Layer 22
        x = Conv2D(1024, (3, 3), strides=(1, 1), padding='same', name='conv_22', use_bias=False)(x)
        x = BatchNormalization(name='norm_22')(x)
        x = LeakyReLU(alpha=0.1)(x)

        self.feature_extractor = Model(input_image, x, name='Full_YOLO_backend')
        if input_size[2] == 3:
            try:
                print("Loading pretrained weights: " + FULL_YOLO_BACKEND_PATH)
                self.feature_extractor.load_weights(FULL_YOLO_BACKEND_PATH)
            except:
                print("Unable to load backend weights. Using a fresh model")
        else:
            print('pre trained weights are available just for RGB network.')

    def normalize(self, image):
        return image / 255.


class TinyYoloFeature(BaseFeatureExtractor):
    """docstring for ClassName"""

    def __init__(self, input_size):
        input_image = Input(shape=input_size)

        # Layer 1
        x = Conv2D(16, (3, 3), strides=(1, 1), padding='same', name='conv_1', use_bias=False)(input_image)
        x = BatchNormalization(name='norm_1')(x)
        x = LeakyReLU(alpha=0.1)(x)
        x = MaxPooling2D(pool_size=(2, 2))(x)

        # Layer 2 - 5
        for i in range(0, 4):
            x = Conv2D(32 * (2 ** i), (3, 3), strides=(1, 1), padding='same', name='conv_' + str(i + 2),
                       use_bias=False)(x)
            x = BatchNormalization(name='norm_' + str(i + 2))(x)
            x = LeakyReLU(alpha=0.1)(x)
            x = MaxPooling2D(pool_size=(2, 2))(x)

        # Layer 6
        x = Conv2D(512, (3, 3), strides=(1, 1), padding='same', name='conv_6', use_bias=False)(x)
        x = BatchNormalization(name='norm_6')(x)
        x = LeakyReLU(alpha=0.1)(x)
        x = MaxPooling2D(pool_size=(2, 2), strides=(1, 1), padding='same')(x)

        # Layer 7 - 8
        for i in range(0, 2):
            x = Conv2D(1024, (3, 3), strides=(1, 1), padding='same', name='conv_' + str(i + 7), use_bias=False)(x)
            x = BatchNormalization(name='norm_' + str(i + 7))(x)
            x = LeakyReLU(alpha=0.1)(x)

        self.feature_extractor = Model(input_image, x, name='Tiny_YOLO_backend')
        if input_size[2] == 3:
            try:
                print("Loading pretrained weights: " + TINY_YOLO_BACKEND_PATH)
                self.feature_extractor.load_weights(TINY_YOLO_BACKEND_PATH)
            except:
                print("Unable to load backend weights. Using a fresh model")
        else:
            print('pre trained weights are available just for RGB network.')

    def normalize(self, image):
        return image / 255.


class MobileNetFeature(BaseFeatureExtractor):
    """docstring for ClassName"""

    def __init__(self, input_size):
        input_image = Input(shape=input_size)

        mobilenet = MobileNet(input_shape=input_size, include_top=False)
        if input_size[2] == 3:
            try:
                print("Loading pretrained weights: " + MOBILENET_BACKEND_PATH)
                mobilenet.load_weights(MOBILENET_BACKEND_PATH)
            except:
                print("Unable to load backend weights. Using a fresh model")
        else:
            print('pre trained weights are available just for RGB network.')

        x = mobilenet(input_image)

        self.feature_extractor = Model(input_image, x, name='MobileNet_backend')

    def normalize(self, image):
        image = image / 255.
        image = image - 0.5
        image = image * 2.

        return image


class SqueezeNetFeature(BaseFeatureExtractor):
    """docstring for ClassName"""

    def __init__(self, input_size):

        # define some auxiliary variables and the fire module
        sq1x1 = "squeeze1x1"
        exp1x1 = "expand1x1"
        exp3x3 = "expand3x3"
        relu = "relu_"

        def fire_module(x, fire_id, squeeze=16, expand=64):
            s_id = 'fire' + str(fire_id) + '/'

            x = Conv2D(squeeze, (1, 1), padding='valid', name=s_id + sq1x1)(x)
            x = Activation('relu', name=s_id + relu + sq1x1)(x)

            left = Conv2D(expand, (1, 1), padding='valid', name=s_id + exp1x1)(x)
            left = Activation('relu', name=s_id + relu + exp1x1)(left)

            right = Conv2D(expand, (3, 3), padding='same', name=s_id + exp3x3)(x)
            right = Activation('relu', name=s_id + relu + exp3x3)(right)

            x = concatenate([left, right], axis=3, name=s_id + 'concat')

            return x

        # define the model of SqueezeNet
        input_image = Input(shape=input_size)

        x = Conv2D(64, (3, 3), strides=(2, 2), padding='valid', name='conv1')(input_image)
        x = Activation('relu', name='relu_conv1')(x)
        x = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), name='pool1')(x)

        x = fire_module(x, fire_id=2, squeeze=16, expand=64)
        x = fire_module(x, fire_id=3, squeeze=16, expand=64)
        x = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), name='pool3')(x)

        x = fire_module(x, fire_id=4, squeeze=32, expand=128)
        x = fire_module(x, fire_id=5, squeeze=32, expand=128)
        x = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), name='pool5')(x)

        x = fire_module(x, fire_id=6, squeeze=48, expand=192)
        x = fire_module(x, fire_id=7, squeeze=48, expand=192)
        x = fire_module(x, fire_id=8, squeeze=64, expand=256)
        x = fire_module(x, fire_id=9, squeeze=64, expand=256)

        self.feature_extractor = Model(input_image, x, name='SqueezeNet_backend')
        if input_size[2] == 3:
            try:
                self.feature_extractor.load_weights(SQUEEZENET_BACKEND_PATH)
            except:
                print("Unable to load backend weights. Using a fresh model")
        else:
            print('pre trained weights are available just for RGB network.')

    def normalize(self, image):
        image = image[..., ::-1]
        image = image.astype('float')

        image[..., 0] -= 103.939
        image[..., 1] -= 116.779
        image[..., 2] -= 123.68

        return image


class Inception3Feature(BaseFeatureExtractor):
    """docstring for ClassName"""

    def __init__(self, input_size):
        input_image = Input(shape=input_size)

        inception = InceptionV3(input_shape=input_size, include_top=False)
        if input_size[2] == 3:
            try:
                inception.load_weights(INCEPTION3_BACKEND_PATH)
            except:
                print("Unable to load backend weights. Using a fresh model")
        else:
            print('pre trained weights are available just for RGB network.')

        x = inception(input_image)

        self.feature_extractor = Model(input_image, x, name='Inception3_backend')

    def normalize(self, image):
        image = image / 255.
        image = image - 0.5
        image = image * 2.

        return image


class VGG16Feature(BaseFeatureExtractor):
    """docstring for ClassName"""

    def __init__(self, input_size):
        vgg16 = VGG16(input_shape=input_size, include_top=False)
        # vgg16.load_weights(VGG16_BACKEND_PATH)

        self.feature_extractor = vgg16

    def normalize(self, image):
        image = image[..., ::-1]
        image = image.astype('float')

        image[..., 0] -= 103.939
        image[..., 1] -= 116.779
        image[..., 2] -= 123.68

        return image


class ResNet50Feature(BaseFeatureExtractor):
    """docstring for ClassName"""

    def __init__(self, input_size):
        resnet50 = ResNet50(input_shape=input_size, include_top=False)
        resnet50.layers.pop()  # remove the average pooling layer
        # resnet50.load_weights(RESNET50_BACKEND_PATH)

        self.feature_extractor = Model(resnet50.layers[0].input, resnet50.layers[-1].output)

    def normalize(self, image):
        image = image[..., ::-1]
        image = image.astype('float')

        image[..., 0] -= 103.939
        image[..., 1] -= 116.779
        image[..., 2] -= 123.68

        return image
