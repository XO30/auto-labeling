import unittest
import json
from unittest.mock import patch
from converter.cococonverter import CocoConverter


class CocoConverterTestCase(unittest.TestCase):
    def setUp(self):
        self.converter = CocoConverter()

    def test_generate_random_image_name(self):
        random_name = self.converter.generate_random_image_name()
        self.assertIsInstance(random_name, str)

    def test_img_annotations_to_coco(self):
        info = {"description": "COCO Dataset", "version": "1.0", "year": 2023, "contributor": "Your Name"}
        licenses = [{"id": 1, "name": "License 1", "url": "https://license1.com"}]
        categories = [{"id": 1, "name": "Category 1", "supercategory": "Supercategory 1"}]
        annotations = [[{"segmentation": [0, 0, 100, 0, 100, 100, 0, 100], "bbox": [0, 0, 100, 100], "area": 1000}]]

        coco_data, image_names = self.converter.img_annotations_to_coco(info, licenses, categories, annotations)

        self.assertIsInstance(coco_data, dict)
        self.assertIn("images", coco_data)
        self.assertIn("annotations", coco_data)
        self.assertIsInstance(image_names, list)
        self.assertEqual(len(image_names), 1)
        self.assertEqual(len(coco_data["images"]), 1)
        self.assertEqual(len(coco_data["annotations"]), 1)
        self.assertEqual(coco_data["images"][0]["file_name"], image_names[0])
        self.assertEqual(coco_data["annotations"][0]["image_id"], 1)

    def test_parameters_to_coco(self):
        info = {"description": "COCO Dataset", "version": "1.0", "year": 2023, "contributor": "Your Name"}
        licenses = [{"id": 1, "name": "License 1", "url": "https://license1.com"}]
        categories = [{"id": 1, "name": "Category 1", "supercategory": "Supercategory 1"}]
        parameters = [
            {
                "segmentation": [0, 0, 100, 0, 100, 100, 0, 100],
                "bbox": [0, 0, 100, 100],
                "area": 1000,
                "file_name": "image1.jpg"
            },
            {
                "segmentation": [100, 100, 200, 100, 200, 200, 100, 200],
                "bbox": [100, 100, 100, 100],
                "area": 10000,
                "file_name": "image2.jpg"
            }
        ]

        coco_data = self.converter.parameters_to_coco(info, licenses, categories, parameters)

        self.assertIsInstance(coco_data, dict)
        self.assertIn("images", coco_data)
        self.assertIn("annotations", coco_data)
        self.assertEqual(len(coco_data["images"]), 2)
        self.assertEqual(len(coco_data["annotations"]), 2)
        self.assertEqual(coco_data["images"][0]["file_name"], "image1.jpg")
        self.assertEqual(coco_data["annotations"][0]["image_id"], 1)
        self.assertEqual(coco_data["annotations"][0]["segmentation"], [0, 0, 100, 0, 100, 100, 0, 100])
        self.assertEqual(coco_data["annotations"][1]["image_id"], 2)
        self.assertEqual(coco_data["annotations"][1]["segmentation"], [100, 100, 200, 100, 200, 200, 100, 200])


if __name__ == '__main__':
    unittest.main()
