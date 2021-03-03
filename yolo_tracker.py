import sys
import os.path
from utils.general import (
    check_img_size, non_max_suppression, scale_coords, xyxy2xywh)
from utils.torch_utils import select_device, time_synchronized
from utils.parser import get_config
from deep_sort import DeepSort
import argparse
import os
import platform
import shutil
import time
from pathlib import Path
import cv2
import torch
import numpy as np
import torch.backends.cudnn as cudnn
sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))

from person_tracker import PersonTracker

FILE_PATH = os.path.dirname( os.path.abspath(__file__) )

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


def letterbox(img, new_shape=(640, 640), color=(114, 114, 114), auto=True, scaleFill=False, scaleup=True):
    """
    Resize image to a 32-pixel-multiple rectangle https://github.com/ultralytics/yolov3/issues/232
    """
    shape = img.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:  # only scale down, do not scale up (for better test mAP)
        r = min(r, 1.0)

    # Compute padding
    ratio = r, r  # width, height ratios
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding
    if auto:  # minimum rectangle
        dw, dh = np.mod(dw, 64), np.mod(dh, 64)  # wh padding
    elif scaleFill:  # stretch
        dw, dh = 0.0, 0.0
        new_unpad = (new_shape[1], new_shape[0])
        ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]  # width, height ratios

    dw /= 2  # divide padding into 2 sides
    dh /= 2

    if shape[::-1] != new_unpad:  # resize
        img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
    return img, ratio, (dw, dh)


class YoloTracker(PersonTracker):

    def setup(self, weights=os.path.join(FILE_PATH, 'weights', 'yolov5x.pt'), conf_thres=0.4,
              iou_thres=0.5, device='cpu', imgsz=640, classes=[0],
              agnostic_nms=True):
        self.file_path = os.path.dirname( os.path.abspath(__file__) )
        self.conf_thres = conf_thres
        self.iou_thres = iou_thres
        self.classes = classes
        self.agnostic_nms = agnostic_nms
        self.imgsz = imgsz
        cfg = get_config()
        cfg.merge_from_file( os.path.join(FILE_PATH, 'deep_sort', 'deep_sort.yaml') )
        self.deepsort = DeepSort( os.path.join(FILE_PATH, cfg.DEEPSORT.REID_CKPT),
                                 max_dist=cfg.DEEPSORT.MAX_DIST,
                                 min_confidence=cfg.DEEPSORT.MIN_CONFIDENCE,
                                 nms_max_overlap=cfg.DEEPSORT.NMS_MAX_OVERLAP,
                                 max_iou_distance=cfg.DEEPSORT.MAX_IOU_DISTANCE,
                                 max_age=cfg.DEEPSORT.MAX_AGE,
                                 n_init=cfg.DEEPSORT.N_INIT,
                                 nn_budget=cfg.DEEPSORT.NN_BUDGET,
                                 use_cuda=False)
        self.device = select_device(device)
        self.half = self.device.type != 'cpu'
        self.model = torch.load(weights, map_location=self.device)['model'].float()
        self.model.to(self.device).eval()
        if self.half:
            self.model.half()
        img = torch.zeros((1, 3, self.imgsz, self.imgsz), device=self.device)
        _ = self.model(img.half() if self.half else img) if self.device.type != 'cpu' else None
        print('Setup completed')

    def update(self, img):
        im0 = img.copy()

        img = letterbox(img, new_shape=self.imgsz)[0]
        img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
        img = np.ascontiguousarray(img)

        img = torch.from_numpy(img).to(self.device)
        img = img.half() if self.half else img.float()
        img /= 255.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        self.t1 = time_synchronized()

        pred = self.model(img)[0]
        pred = non_max_suppression(pred, self.conf_thres, self.iou_thres,
                                   classes=self.classes,
                                   agnostic=self.agnostic_nms)

        self.t2 = time_synchronized()

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
                outputs = self.deepsort.update(xywhs, confss, im0)
                if len(outputs) > 0:                  
                    bb = xyxy2xywh(outputs[:, :4])
                    idx = outputs[:, -1]

        self.t3 = time_synchronized()

        return idx, bb

    def general_fps(self) -> float:
        return 1.0 / (self.t3 - self.t1)

    def obj_detect_fps(self) -> float:
        return 1.0 / (self.t2 - self.t1)

    def deep_sort_fps(self) -> float:
        return 1.0 / (self.t3 - self.t2)


if __name__ == "__main__":
    import cv2

    tracker = YoloTracker()
    source = 'samples/video_example_mczs_1.mp4'

    cap = cv2.VideoCapture(source)
    assert cap.isOpened(), f'Failed to Open {source}'
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) % 100

    while cap.isOpened():
        ret, img = cap.read()

        cv2.imshow('yolo', img)

        if cv2.waitKey(1) == ord('q'):
            cv2.destroyAllWindows()
            break

        idx, bb = tracker.update(img)

        # print(f'FPS: {tracker.general_fps():.3f}, '
        #       f'{tracker.obj_detect_fps():.3f}, '
        #       f'{tracker.deep_sort_fps():.3f}')
        if idx is not None:
            print(f'types: {type(idx)}, {type(bb)}, 0: {idx[0]}, {bb[0]}')
        print(f'Persons: {len(idx) if idx is not None else 0}')
