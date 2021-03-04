import unittest
import os
import cv2

from tracker.yolo_tracker import YoloTracker


class TestYoloTracker(unittest.TestCase):

    def test_detect(self):
        print('-------------')
        print(os.listdir('.'))

        tracker = YoloTracker()
        source = 'samples/vid02.mp4'

        cap = cv2.VideoCapture(source)

        idx = None
        bb = None

        while cap.isOpened() and idx is None:
            ret, img = cap.read()
            idx, bb = tracker.update(img)

        self.assertIsNotNone(idx)
        self.assertIsNotNone(bb)
        self.assertEqual(2, len(idx))
        print(type(bb[0]))
        self.assertEqual([394, 60, 312, 367], bb[0].tolist())
        self.assertEqual([66, 74, 276, 354], bb[1].tolist())