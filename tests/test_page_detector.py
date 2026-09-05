"""
Unit tests for PageDetector and perspective transformation.
"""

import unittest
import numpy as np
import cv2
from visionvoice.ocr.page_detector import PageDetector
from visionvoice.core.models import PageDetectionStatus


class TestPageDetector(unittest.TestCase):

    def setUp(self):
        self.detector = PageDetector(min_area_ratio=0.10, required_stable_frames=2)

    def test_order_points_clockwise(self):
        pts = np.array([[200, 200], [10, 10], [200, 10], [10, 200]], dtype=np.float32)
        ordered = PageDetector.order_points(pts)
        # Expected: tl=(10,10), tr=(200,10), br=(200,200), bl=(10,200)
        np.testing.assert_array_equal(ordered[0], [10, 10])
        np.testing.assert_array_equal(ordered[1], [200, 10])
        np.testing.assert_array_equal(ordered[2], [200, 200])
        np.testing.assert_array_equal(ordered[3], [10, 200])

    def test_synthetic_page_detection_and_warp(self):
        # Create a synthetic image containing a white page on a dark background
        img = np.zeros((800, 800, 3), dtype=np.uint8)
        # Draw a white rectangle representing a page
        cv2.rectangle(img, (150, 100), (650, 700), (240, 240, 240), -1)
        # Add some text lines inside the page
        cv2.putText(img, "VISIONVOICE TEST PAGE", (200, 200), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)

        # 1st frame analysis
        res1 = self.detector.analyze_frame(img)
        self.assertIn(res1.status, (PageDetectionStatus.PAGE_DETECTED, PageDetectionStatus.PAGE_STABLE))
        self.assertIsNotNone(res1.quadrilateral)

        # 2nd consecutive identical frame should be STABLE
        res2 = self.detector.analyze_frame(img)
        self.assertEqual(res2.status, PageDetectionStatus.PAGE_STABLE)

        # Perform warp
        warped = self.detector.warp_perspective(img, res2.quadrilateral)
        self.assertGreater(warped.shape[0], 400)
        self.assertGreater(warped.shape[1], 300)


if __name__ == "__main__":
    unittest.main()
