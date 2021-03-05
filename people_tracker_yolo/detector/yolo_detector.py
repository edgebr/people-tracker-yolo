import cv2
import torch
import numpy as np

from people_tracker_yolo.detector.utils.general import non_max_suppression, scale_coords
from people_tracker_yolo.detector.utils.torch_utils import select_device
from people_tracker_yolo.detector import settings


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


class YoloDetector:

    def __init__(self):
        self.__setup()

    def __setup(self, weights=settings.NET_WEIGHTS, conf_thres=0.4, iou_thres=0.5,
                device=settings.DEVICE, imgsz=640, classes=[0], agnostic_nms=True):
        self.conf_thres = conf_thres
        self.iou_thres = iou_thres
        self.classes = classes
        self.agnostic_nms = agnostic_nms
        self.imgsz = imgsz

        try:
            self.device = select_device(device)
        except AssertionError:
            device = 'cpu'
            self.device = select_device(device)

        self.half = self.device.type != 'cpu'
        self.model = torch.hub.load('ultralytics/yolov5', 'custom', path_or_model=weights)
        self.model = self.model.autoshape()  # for PIL/cv2/np inputs and NMS
        self.model.to(self.device).eval()
        if self.half:
            self.model.half()
        img = torch.zeros((1, 3, self.imgsz, self.imgsz), device=self.device)
        if self.device.type != 'cpu':
            self.model(img.half() if self.half else img)
        print('Setup completed')


    def preprocessing_image(self, img):
        img = letterbox(img, new_shape=self.imgsz)[0]
        img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
        img = np.ascontiguousarray(img)

        img = torch.from_numpy(img).to(self.device)
        img = img.half() if self.half else img.float()
        img /= 255.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        return img


    def update(self, img):
        im0 = img.copy()

        img = self.preprocessing_image(img)

        pred = self.model(img)[0]
        pred = non_max_suppression(pred, self.conf_thres, self.iou_thres,
                                   classes=self.classes,
                                   agnostic=self.agnostic_nms)

        # x, y, w, h, conf
        bboxes = []

        for i, det in enumerate(pred):
            if det is not None and len(det):
                det[:, :4] = \
                    scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                for x1, y1, x2, y2, conf, _ in det:
                    x = min(x1, x2)
                    y = min(y1, y2)
                    w = abs(x1 - x2)
                    h = abs(y1 - y2)
                    bboxes.append([x.item(), y.item(), w.item(), h.item(), conf.item()])

        return bboxes
