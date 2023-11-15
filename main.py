
import torch, detectron2
TORCH_VERSION = ".".join(torch.__version__.split(".")[:2])
# CUDA_VERSION = torch.__version__.split("+")[-1]
device = torch.device("cuda")
import numpy as np
import cv2
import matplotlib.pyplot as plt
import os
import json
import matplotlib.pyplot as plt
from detectron2.utils.logger import setup_logger
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.structures import BoxMode
# from google.colab.patches import cv2_imshow

setup_logger()

from detectron2.engine import DefaultTrainer
from detectron2.utils.visualizer import ColorMode
import random
import time 
from detectron2.config import get_cfg

import os.path
from os import path

import warnings
warnings.filterwarnings("ignore")

# from google.colab import drive
# drive.mount('/content/gdrive', force_remount=True)

def get_dictionaries(img_dir):
    json_file = os.path.join(img_dir, "result.json")

    with open(json_file) as f:
        imgs_anns = json.load(f)
        images = imgs_anns['images']
        annotations = imgs_anns['annotations']
        dataset_dicts = []
        
        for idx, v in enumerate(images):
          if path.exists(img_dir + images[idx]["file_name"].split('/')[1]):
            record = {}
            filename = img_dir + images[idx]["file_name"].split('/')[1]
            width = images[idx]["width"]
            height = images[idx]["height"]

            record["file_name"] = filename
            record["image_id"] = idx
            record["width"] = width
            record["height"] = height
            
            annos = {}
            for i, anno in enumerate(annotations):
              if(anno["image_id"]==idx):
                annos[i] = annotations[i]
            
            objs = []
            for _, anno in annos.items():
              obj = {
                # "bbox": [np.min(px), np.min(py), np.max(px), np.max(py)],
                # Four values [x_min, y_min, width, height] define a bounding box in COCO format.
                "bbox": [anno['bbox'][0], anno['bbox'][1], anno['bbox'][2]+anno['bbox'][0], anno['bbox'][3]+anno['bbox'][1]],
                "bbox_mode": BoxMode.XYXY_ABS,
                "segmentation": [],
                "category_id": 0,
              }
              objs.append(obj)
            
            record["annotations"] = objs
            dataset_dicts.append(record)

    return dataset_dicts

image_dir = './tag/'

for d in ["train", "val"]:
    # DatasetCatalog.remove("tag_" + d)
    DatasetCatalog.register("tag_" + d, lambda d=d: get_dictionaries(image_dir + d+ '/'))
    MetadataCatalog.get("tag_" + d).set(thing_classes=["circle"])
tag_metadata = MetadataCatalog.get("tag_train")

cfg = get_cfg()
cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_50_FPN_1x.yaml"))
# cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_101_FPN_3x.yaml"))
cfg.DATASETS.TRAIN = ("tag_train",)
cfg.DATASETS.TEST = ()   # no metrics implemented for this dataset
# cfg.DATALOADER.NUM_WORKERS = 2
cfg.DATALOADER.NUM_WORKERS = 0
cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-Detection/faster_rcnn_R_50_FPN_1x.yaml")  # initialize from model zoo
# cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-Detection/faster_rcnn_R_101_FPN_3x.yaml")  # initialize from model zoo
cfg.SOLVER.IMS_PER_BATCH = 2
cfg.SOLVER.BASE_LR = 0.00025
cfg.SOLVER.MAX_ITER = 2
# cfg.SOLVER.MAX_ITER = 500
cfg.SOLVER.STEPS = []        # do not decay learning rate
# cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 1   # faster, and good enough for this toy dataset
cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128   # faster, and good enough for this toy dataset
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1  # only has one class (ballon)

os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
trainer = DefaultTrainer(cfg) 
trainer.resume_or_load(resume=False)
trainer.train()

cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7   # set the testing threshold for this model
cfg.DATASETS.TEST = ("tag_val", )
predictor = DefaultPredictor(cfg)

image_dir = './tag/'

# im = cv2.imread(image_dir+'Capture_balloon.PNG')
# outputs = predictor(im)
# print(outputs)
# v = Visualizer(im[:, :, ::-1], metadata=motorcycle_metadata, scale=0.8)
# v = v.draw_instance_predictions(outputs["instances"].to("cpu"))
# plt.figure(figsize = (14, 10))
# plt.imshow(cv2.cvtColor(v.get_image()[:, :, ::-1], cv2.COLOR_BGR2RGB))
# plt.show()

dataset_dicts = get_dictionaries(image_dir+"val"+'/')
for d in random.sample(dataset_dicts, 20):    
    t = time.time()
    im = cv2.imread(d["file_name"])
    outputs = predictor(im)
    print(outputs)
    print('time')
    print(time.time() - t)
    v = Visualizer(im[:, :, ::-1], metadata=tag_metadata, scale=0.8)
    v = v.draw_instance_predictions(outputs["instances"].to("cpu"))
    plt.figure(figsize = (14, 10))
    plt.imshow(cv2.cvtColor(v.get_image()[:, :, ::-1], cv2.COLOR_BGR2RGB))
    plt.show()
