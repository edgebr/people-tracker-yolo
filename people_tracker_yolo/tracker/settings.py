import os

DEEP_SORT_REID_CKPT = os.environ.get('DEEP_SORT_REID_CKPT')
DEEP_SORT_MAX_DIST = float(os.environ.get('DEEP_SORT_MAX_DIST', 0.2))
DEEP_SORT_MIN_CONFIDENCE = float(os.environ.get('DEEP_SORT_MIN_CONFIDENCE', 0.3))
DEEP_SORT_NMS_MAX_OVERLAP = float(os.environ.get('DEEP_SORT_NMS_MAX_OVERLAP', 0.5))
DEEP_SORT_MAX_IOU_DISTANCE = float(os.environ.get('DEEP_SORT_MAX_IOU_DISTANCE', 0.7))
DEEP_SORT_MAX_AGE = int(os.environ.get('DEEP_SORT_MAX_AGE', 70))
DEEP_SORT_N_INIT = int(os.environ.get('DEEP_SORT_N_INIT', 3))
DEEP_SORT_NN_BUDGET = int(os.environ.get('DEEP_SORT_NN_BUDGET', 100))