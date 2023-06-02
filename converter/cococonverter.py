import json
import random
import string


class CocoConverter:
    def __init__(self):
        self.image_names = []

    def generate_random_image_name(self):
        letters = string.ascii_lowercase
        while True:
            random_name = ''.join(random.choice(letters) for _ in range(10))
            if random_name not in self.image_names:
                self.image_names.append(random_name)
                return random_name

    def img_annotations_to_coco(self, info, licenses, categories, annotations):
        coco_data = {
            "info": info,
            "licenses": licenses,
            "categories": categories,
            "images": [],
            "annotations": []
        }

        for image_annotations in annotations:
            image_name = self.generate_random_image_name()
            coco_data["images"].append({
                "id": len(coco_data["images"]) + 1,
                "file_name": image_name
            })

            for annotation in image_annotations:
                annotation["image_id"] = len(coco_data["images"])
                coco_data["annotations"].append(annotation)

        return coco_data, self.image_names

    def parameters_to_coco(self, info, licenses, categories, parameters):
        coco_data = {
            "info": info,
            "licenses": licenses,
            "categories": categories,
            "images": [],
            "annotations": []
        }

        for image_params in parameters:
            coco_data["images"].append({
                "id": len(coco_data["images"]) + 1,
                "file_name": image_params["file_name"]
            })
            annotation = {
                "id": len(coco_data["annotations"]) + 1,
                "image_id": len(coco_data["images"]),
                "segmentation": image_params["segmentation"],
                "bbox": image_params["bbox"],
                "area": image_params["area"]
            }
            coco_data["annotations"].append(annotation)

        return coco_data

    def save_coco_file(self, coco_data, filename):
        with open(filename, 'w') as f:
            json.dump(coco_data, f)
        print(f"Saved COCO file as {filename}")