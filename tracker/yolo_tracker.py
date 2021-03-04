import os

import torch

from detector.utils.general import non_max_suppression
from detector.utils.general import scale_coords, xyxy2xywh
from detector.yolo_detector import YoloDetector

from deep_sort import DeepSort

from tracker import settings


FILE_PATH = os.path.dirname( os.path.abspath(__file__) )
DEEP_SORT_CONF = os.path.join(FILE_PATH, 'deep_sort', 'deep_sort.yaml')
NET_WEIGHTS = os.path.join(FILE_PATH, 'weights', 'yolov5x.pt')

DEEP_SORT_CONF = os.environ.get('DEEP_SORT_CONF', DEEP_SORT_CONF)
NET_WEIGHTS = os.environ.get('NET_WEIGHTS', NET_WEIGHTS)

def bbox_rel(image_width, image_height,  *xyxy):
    """
    Calculates the relative bounding box from absolute pixel values.
    """
    bbox_left = min([xyxy[0].item(), xyxy[2].item()])
    bbox_top = min([xyxy[1].item(), xyxy[3].item()])
    bbox_w = abs(xyxy[0].item() - xyxy[2].item())
    bbox_h = abs(xyxy[1].item() - xyxy[3].item())
    x_c = (bbox_left + bbox_w / 2)
    y_c = (bbox_top + bbox_h / 2)
    w = bbox_w
    h = bbox_h
    return x_c, y_c, w, h


class YoloTracker(YoloDetector):

    def __init__(self):
        super(YoloTracker, self).__init__()

        self.deep_sort = DeepSort(settings.DEEP_SORT_REID_CKPT,
                                  max_dist=settings.DEEP_SORT_MAX_DIST,
                                  min_confidence=settings.DEEP_SORT_MIN_CONFIDENCE,
                                  nms_max_overlap=settings.DEEP_SORT_NMS_MAX_OVERLAP,
                                  max_iou_distance=settings.DEEP_SORT_MAX_IOU_DISTANCE,
                                  max_age=settings.DEEP_SORT_MAX_AGE,
                                  n_init=settings.DEEP_SORT_N_INIT,
                                  nn_budget=settings.DEEP_SORT_NN_BUDGET,
                                  use_cuda=self.device.type != 'cpu')


    def update(self, img):
        im0 = img.copy()

        img = self.preprocessing_image(img)

        pred = self.model(img)[0]
        pred = non_max_suppression(pred, self.conf_thres, self.iou_thres,
                                   classes=self.classes,
                                   agnostic=self.agnostic_nms)

        idx = None
        bb = None

        for i, det in enumerate(pred):
            if det is not None and len(det):
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                bbox_xywh = []
                confs = []

                # Adapt detections to deep sort input format
                for *xyxy, conf, cls in det:
                    img_h, img_w, _ = im0.shape
                    x_c, y_c, bbox_w, bbox_h = bbox_rel(img_w, img_h, *xyxy)
                    obj = [x_c, y_c, bbox_w, bbox_h]
                    bbox_xywh.append(obj)
                    confs.append([conf.item()])

                xywhs = torch.Tensor(bbox_xywh)
                confss = torch.Tensor(confs)
                outputs = self.deep_sort.update(xywhs, confss, im0)
                if len(outputs) > 0:                  
                    bb = xyxy2xywh(outputs[:, :4])
                    idx = outputs[:, -1]

        return idx, bb

