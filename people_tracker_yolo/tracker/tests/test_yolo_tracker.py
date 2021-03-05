import unittest
import cv2

from people_tracker_yolo.tracker.yolo_tracker import YoloTracker


class TestYoloTracker(unittest.TestCase):

    def test_detect(self):
        tracker = YoloTracker()
        source = 'samples/video_example_mczs_1.mp4'

        cap = cv2.VideoCapture(source)

        idx = None
        bb = None

        while cap.isOpened() and idx is None:
            ret, img = cap.read()
            idx, bb = tracker.update(img)

            if bb is not None:
                for b, i in zip(bb, idx):
                    x, y, w, h = b
                    cv2.rectangle(img, (x, y), (x + w, y + h), color=(0, 0, 255), thickness=2)
                    cv2.putText(img, '{}'.format(i), (x, y + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

                cv2.imwrite(source + '.png', img)

        self.assertIsNotNone(idx)
        self.assertIsNotNone(bb)
        self.assertEqual(4, len(idx))
        self.assertEqual([941, 195, 110, 294], bb[0].tolist())
        self.assertEqual([728, 149, 90, 243], bb[1].tolist())
        self.assertEqual([1042, 141, 96, 230], bb[2].tolist())
        self.assertEqual([854, 116, 44, 118], bb[3].tolist())