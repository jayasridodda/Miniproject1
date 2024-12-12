# Miniproject-1
Date:
### AIM:

To Microplastic detection using detectron v2 model for qualitative analysis of oceanic water on the southern coastline of India .


### ALGORITHM:
Step 1:
Detect microplastics in oceanic water samples along India's southern coastline.

Step 2:
Collect and annotate images, split into training, validation, and test sets.

Step 3:
Install Detectron2, dependencies, and configure the system.

Step 3:
Model Configuration, Choose a pre-trained model (e.g., Mask R-CNN) and set hyperparameters.

Step 4:
Data Augmentation by Applying flips, rotations, and noise to enhance model robustness.

Step 5:
Train Model with annotated data and save checkpoints.

Step 6:
Evaluate Performance by Using mAP@50:70 and visualize predictions.

Step 7:
Deploy Model by Saving and integrating into an inference system.

Step 8: 
Report findings and provide visualization.

### PROGRAM:
```
import torch, detectron2
!nvcc --version
TORCH_VERSION = ".".join(torch.__version__.split(".")[:2])
CUDA_VERSION = torch.__version__.split("+")[-1]
print("torch: ", TORCH_VERSION, "; cuda: ", CUDA_VERSION)
print("detectron2:", detectron2.__version__)
```
```
# COMMON LIBRARIES
import os
import cv2
import matplotlib.pyplot as plt
%matplotlib inline

from datetime import datetime
# from google.colab.patches import cv2_imshow

# DATA SET PREPARATION AND LOADING
from detectron2.data.datasets import register_coco_instances
from detectron2.data import DatasetCatalog, MetadataCatalog

# VISUALIZATION
from detectron2.utils.visualizer import Visualizer
from detectron2.utils.visualizer import ColorMode

# CONFIGURATION
from detectron2 import model_zoo
from detectron2.config import get_cfg

# EVALUATION
from detectron2.engine import DefaultPredictor
from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from detectron2.data import build_detection_test_loader

# TRAINING
from detectron2.engine import DefaultTrainer

# LOGGING
import logging
from detectron2.utils.logger import setup_logger

!wget http://images.cocodataset.org/val2017/000000439715.jpg -q -O input.jpg
image = cv2.imread("/kaggle/working/input.jpg")
# cv2.imshow("Testing images", image)

# Using cv2's imshow opens this image in a different window.
# However, the easier and more presentable way to do this is by using matplotlib's inline function
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB)) # converting BGR to RGB for using matplotlib
plt.axis('off')
plt.show()
```
```
cfg = get_cfg()
cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
predictor = DefaultPredictor(cfg)
outputs = predictor(image)

print(outputs["instances"].pred_classes)
print(outputs["instances"].pred_boxes)

visualizer = Visualizer(image[:, :, ::-1], MetadataCatalog.get(cfg.DATASETS.TRAIN[0]), scale=1.2)
out = visualizer.draw_instance_predictions(outputs["instances"].to("cpu"))
# cv2.imshow(out.get_image()[:, :, ::-1])

plt.imshow(cv2.cvtColor(out.get_image()[:, :, ::-1], cv2.COLOR_BGR2RGB)) # converting BGR to RGB for using matplotlib
plt.axis('off')
plt.show()
```
```
!pip install roboflow

from roboflow import Roboflow
rf = Roboflow(api_key="Dlwgwe3psTHRPIuQlsbV")
project = rf.workspace("panats-mp-project").project("microplastic-dataset")
dataset = project.version(19).download("coco")

DATA_SET_NAME = dataset.name.replace(" ", "-")
ANNOTATIONS_FILE_NAME = "_annotations.coco.json"
ANNOTATIONS_FILE_NAME

# TRAIN SET
TRAIN_DATA_SET_NAME = f"{DATA_SET_NAME}-train"
TRAIN_DATA_SET_IMAGES_DIR_PATH = os.path.join(dataset.location, "train")
TRAIN_DATA_SET_ANN_FILE_PATH = os.path.join(dataset.location, "train", ANNOTATIONS_FILE_NAME)

register_coco_instances(
    name=TRAIN_DATA_SET_NAME, 
    metadata={}, 
    json_file=TRAIN_DATA_SET_ANN_FILE_PATH, 
    image_root=TRAIN_DATA_SET_IMAGES_DIR_PATH
)

# # TEST SET
# TEST_DATA_SET_NAME = f"{DATA_SET_NAME}-test"
# TEST_DATA_SET_IMAGES_DIR_PATH = os.path.join(dataset.location, "test")
# TEST_DATA_SET_ANN_FILE_PATH = os.path.join(dataset.location, "test", ANNOTATIONS_FILE_NAME)

# register_coco_instances(
#     name=TEST_DATA_SET_NAME, 
#     metadata={}, 
#     json_file=TEST_DATA_SET_ANN_FILE_PATH, 
#     image_root=TEST_DATA_SET_IMAGES_DIR_PATH
# )

# VALID SET
VALID_DATA_SET_NAME = f"{DATA_SET_NAME}-valid"
VALID_DATA_SET_IMAGES_DIR_PATH = os.path.join(dataset.location, "valid")
VALID_DATA_SET_ANN_FILE_PATH = os.path.join(dataset.location, "valid", ANNOTATIONS_FILE_NAME)

register_coco_instances(
    name=VALID_DATA_SET_NAME, 
    metadata={}, 
    json_file=VALID_DATA_SET_ANN_FILE_PATH, 
    image_root=VALID_DATA_SET_IMAGES_DIR_PATH
)
VALID_DATA_SET_ANN_FILE_PATH, TRAIN_DATA_SET_ANN_FILE_PATH

[
    data_set
    for data_set
    in MetadataCatalog.list()
    if data_set.startswith(DATA_SET_NAME)
]
metadata = MetadataCatalog.get(TRAIN_DATA_SET_NAME)
dataset_train = DatasetCatalog.get(TRAIN_DATA_SET_NAME)

dataset_entry = dataset_train[0]
image = cv2.imread(dataset_entry["file_name"])

visualizer = Visualizer(
    image[:, :, ::-1],
    metadata=metadata, 
    scale=0.8, 
    instance_mode=ColorMode.IMAGE_BW
)

out = visualizer.draw_dataset_dict(dataset_entry)
# cv2.imshow(out.get_image()[:, :, ::-1])

plt.imshow(cv2.cvtColor(out.get_image()[:, :, ::-1], cv2.COLOR_BGR2RGB)) # converting BGR to RGB for using matplotlib
plt.axis('off')
plt.show()
cfg = get_cfg()
cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"))
cfg.DATASETS.TRAIN = (TRAIN_DATA_SET_NAME)
cfg.DATASETS.TEST = ()

# Pre - processing
cfg.INPUT.MIN_SIZE_TRAIN = (800,)  # Minimum size of the input image during training
cfg.INPUT.MAX_SIZE_TRAIN = 1280  # Maximum size of the input image during training

cfg.DATALOADER.NUM_WORKERS = 2

cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml")  # Let training initialize from model zoo
cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 254 
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 4
# cfg.MODEL.ROI_BOX_HEAD.POOLER_RESOLUTION = 18  # Increase if your objects are very small
cfg.MODEL.RPN.PRE_NMS_TOPK_TRAIN = 2000  # Number of proposals to keep before applying NMS during training
# cfg.MODEL.RPN.PRE_NMS_TOPK_TEST = 1000  # Number of proposals to keep before applying NMS during testing
cfg.MODEL.RPN.POST_NMS_TOPK_TRAIN = 1000  # Number of proposals to keep after applying NMS during training
# cfg.MODEL.RPN.POST_NMS_TOPK_TEST = 500  # Number of proposals to keep after applying NMS during testing

cfg.SOLVER.IMS_PER_BATCH = 2
cfg.SOLVER.BASE_LR = 0.001
cfg.SOLVER.MAX_ITER = 2000    
cfg.SOLVER.WEIGHT_DECAY = 0.005

# Set up Detectron2 logger
setup_logger()

trainer = DefaultTrainer(cfg) 
trainer.resume_or_load(resume=False)
trainer.train()

cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7
predictor = DefaultPredictor(cfg)  

evaluator = COCOEvaluator("Microplastic-Dataset-valid", False, output_dir="/kaggle/working/output/")
val_loader = build_detection_test_loader(cfg, "Microplastic-Dataset-valid")
print(inference_on_dataset(trainer.model, val_loader, evaluator))
```







### OUTPUT:
## Input Image:


## Output Image:
![image](https://github.com/user-attachments/assets/d3a61e5e-af6d-44af-844d-ce3bf4b12393)

### RESULT:
