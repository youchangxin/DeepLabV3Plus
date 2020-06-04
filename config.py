# -*- coding: utf-8 -*-
from easydict import EasyDict as edict

__C                             = edict()
# Consumers can get config by: from config import cfg
cfg                             = __C

# Model section
__C.MODEL                       = edict()
__C.MODEL.NUM_CLASSES           = 21


__C.DATASET_DIR                 = 'dataset/VOC'
__C.COLORMAP                    = [[0,0,0],[128,0,0],[0,128,0], [128,128,0], [0,0,128],
                                   [128,0,128],[0,128,128],[128,128,128],[64,0,0],[192,0,0],
                                   [64,128,0],[192,128,0],[64,0,128],[192,0,128],
                                   [64,128,128],[192,128,128],[0,64,0],[128,64,0],
                                   [0,192,0],[128,192,0],[0,64,128]]
__C.CLASSES                     = ['background','aeroplane','bicycle','bird','boat',
                                   'bottle','bus','car','cat','chair','cow','diningtable',
                                   'dog','horse','motorbike','person','potted plant',
                                   'sheep','sofa','train','tv/monitor']
__C.RGB_MEAN                    = ([0.485, 0.456, 0.406])
__C.RGB_STD                     = ([0.229, 0.224, 0.225])


# Train section
__C.TRAIN                       = edict()
__C.TRAIN.IMG_TXT               = "./data/train_image.txt"
__C.TRAIN.LABEL_DIR             = "./data/train_labels"
__C.TRAIN.LOGDIR                = '/log'
__C.TRAIN.BATCH_SIZE            = 4
__C.TRAIN.EPOCHS                = 30
__C.TRAIN.SAVE_EPOCH            = 5
__C.TRAIN.DATA_AUG              = True

# Test section
__C.TEST                        = edict()
__C.TEST.IMG_TXT                = "./data/test_image.txt"
__C.TEST.LABEL_DIR              = "./data/test_labels"
__C.TEST.BATCH_SIZE             = 1