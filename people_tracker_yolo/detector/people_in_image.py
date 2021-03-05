import io
import cv2
import base64
import numpy as np

from PIL import Image as PIL_Image
from people_tracker_yolo.detector.yolo_detector import YoloDetector


class Image(object):

    def __init__(self):
        self._img = None
        self._type = None

    @staticmethod
    def from_file(filepath: str):
        obj = Image()
        obj.img = cv2.imread(filepath)
        obj.type = 'opencv'

        return obj

    @staticmethod
    def from_opencv(img_array):
        obj = Image()
        obj.img = img_array.copy()
        obj.type = 'opencv'

        return obj

    @staticmethod
    def from_base64(content: str):
        obj = Image()
        obj.img = content
        obj.type = 'base64'

        return obj

    def to_base64(self):
        if self._type == 'base64':
            return self._img
        elif self._type == 'opencv':
            res, im_jpeg = cv2.imencode('.jpg', self._img,
                                        [int(cv2.IMWRITE_JPEG_QUALITY), 90])
            if not res:
                raise Exception('Error in jpeg encoding')
            return base64.b64encode(im_jpeg).decode('utf-8')
        else:
            raise Exception('Image not loaded')

    def to_opencv(self):
        if self._type == 'opencv':
            return self._img
        elif self._type == 'base64':
            imgdata = base64.b64decode(self._img)
            return np.array(PIL_Image.open(io.BytesIO(imgdata)))
        else:
            raise Exception('Image not loaded')


class PeopleInImage(object):
    def __init__(self):
        print('[INFO] Loading model...', flush=True)
        self.ai_engine = YoloDetector()
        print('[INFO] Model loaded', flush=True)

    def detect(self, image_b64):
        img = Image.from_base64(image_b64).to_opencv()
        bboxes = self.ai_engine.update(img)
        del img

        result = {
            'count': len(bboxes),
            'persons': [],
        }

        for i in range(result['count']):
            result['persons'].append({
                'bounding_box': {
                    'top_left_x': bboxes[i][0],
                    'top_left_y': bboxes[i][1],
                    'width': bboxes[i][2],
                    'height': bboxes[i][3],
                    'confidence': bboxes[i][4]
                }
            })

        return result

    @staticmethod
    def draw_detections(img, bboxes):
        img = Image.from_opencv(img)
        for bbox in bboxes:
            bbox = bbox['bounding_box']
            x = bbox['top_left_x']
            y = bbox['top_left_y']
            w = bbox['width']
            h = bbox['height']
            cv2.rectangle(
                img.to_opencv(), (x, y), (x + w, y + h), (0, 0, 255), 2
            )

        return img