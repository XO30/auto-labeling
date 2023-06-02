import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import shutil
import json
import random
import string

class ImageCollection:
    def __init__(self, image_path):
        self.image_path = image_path
        self.classes = self._get_classes()
        self.image_name = self._get_image_name(self.classes)
        self.image_count = sum([len(self.image_name[c]) for c in self.classes if c != 0])
        self.images = None
        self.mask = None
        self.coco = None

    def _get_classes(self):
        """
        Get all the classes
        :return: list: list of classes
        """
        self.classes = [f for f in os.listdir(self.image_path) if os.path.isdir(os.path.join(self.image_path, f))]
        return self.classes

    def _get_image_name(self, class_names: list) -> dict:
        """
        Save the image name in a dictionary
        :param class_names: list: list of classes
        :return: dict: dictionary with class name as key and list of image name as value
        """
        image_dict = dict()
        for c in class_names:
            image_dict[c] = [f for f in os.listdir(os.path.join(self.image_path, c)) if os.path.isfile(os.path.join(self.image_path, c, f))]
        return image_dict

    def _get_background_RGB(self) -> np.ndarray:
        """
        Get the average RGB of the background
        :return: np.ndarray: average RGB of the background
        """
        # loop over all the images and average the RGB from all 4 corners
        background = np.zeros((4, 3))
        for c in self.classes:
            if c != '0':
                for i in range(len(self.images[c])):
                    background[0] += self.images[c][i][0, 0, :]
                    background[1] += self.images[c][i][0, -1, :]
                    background[2] += self.images[c][i][-1, 0, :]
                    background[3] += self.images[c][i][-1, -1, :]
        background = background / self.image_count
        return background

    @staticmethod
    def _correct_image(image, background) -> np.ndarray:
        """
        Correct the image with the background
        :param image: np.ndarray: image to be corrected
        :param background: np.ndarray: average RGB of the background
        :return: np.ndarray: corrected image
        """
        # all 4 corners of the image
        corners = [image[0, 0, :], image[0, -1, :], image[-1, 0, :], image[-1, -1, :]]
        # calculate the difference between the corners and the background
        differences = corners - background
        # if the difference is negative, make it 0
        differences[differences < 0] = 0
        # get the average difference
        average_differences = np.mean(differences, axis=0)
        # make difference to int
        average_differences = np.array(average_differences, dtype=np.uint8)
        # correct the image with the average difference
        image_copy = image.copy().astype(np.int16)
        image_copy -= average_differences
        # mke all the negative values to 0
        image_copy[image_copy < 0] = 0
        return image_copy.astype(np.uint8)

    @staticmethod
    def _get_polygon(mask: np.ndarray, id: int) -> list:
        """
        Get the polygon from the mask
        :param mask: np.ndarray: mask of the image
        :param id: int: instance id
        :return: list: polygon of the mask
        """
        contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        contours_size = [len(contour) for contour in contours]
        # get the index of the biggest contour
        biggest_contour_index = np.argmax(contours_size)
        segmentation = list()
        segmentation.append(contours[biggest_contour_index])
        annotations = list()
        if len(segmentation) == 0:
            pass
        else:
            for i, seg in enumerate(segmentation):
                single_annotation = dict()
                single_annotation["segmentation_coords"] = (seg.astype(float).flatten().tolist())
                single_annotation["bbox"] = list(cv2.boundingRect(seg))
                single_annotation["area"] = cv2.contourArea(seg)
                single_annotation["instance_id"] = id
                single_annotation["annotation_id"] = f"{id}_{i}"
                annotations.append(single_annotation)
        return annotations

    @staticmethod
    def _np_encoder(object: object) -> object:
        """
        Encode numpy array to json
        :param object: numpy array
        :return: object: json object
        """
        if isinstance(object, np.generic):
            return object.item()

    def extract_objects(self) -> None:
        objects = dict()
        for c in self.classes:
            if c != '0':
                # expand the mask to 3 channels
                objects[c] = [self.images[c][i] * np.expand_dims(self.mask[c][i], axis=2) for i in range(len(self.images[c]))]
        return objects

    def load_images(self) -> None:
        """
        Load all the images into a dictionary
        :return: None
        """
        self.images = dict()
        for c in self.classes:
            self.images[c] = [cv2.cvtColor(cv2.imread(f), cv2.COLOR_BGR2RGB) for f in [os.path.join(self.image_path, c, f) for f in self.image_name[c]]]
        print('Images loaded')
        return None

    def resize_images(self, size: tuple = (640, 480)) -> None:
        """
        Resize all the images to the same size
        :param size: tuple: size of the image (width, height)
        :return: None
        """
        for c in self.classes:
            for i in range(len(self.images[c])):
                # get the original size
                original_ratio = self.images[c][i].shape[1] / self.images[c][i].shape[0]
                new_ratio = size[0] / size[1]
                # crop image
                if original_ratio > new_ratio:
                    # crop width
                    crop_width = int(self.images[c][i].shape[1] - self.images[c][i].shape[0] * new_ratio)
                    self.images[c][i] = self.images[c][i][:, crop_width // 2: -crop_width // 2, :]
                elif original_ratio < new_ratio:
                    # crop height
                    crop_height = int(self.images[c][i].shape[0] - self.images[c][i].shape[1] / new_ratio)
                    self.images[c][i] = self.images[c][i][crop_height // 2: -crop_height // 2, :, :]
                self.images[c][i] = cv2.resize(self.images[c][i], size)
        print('Images resized')
        return None

    def save(self, filename: str = 'image_collection') -> None:
        """
        Save the images and annotations to a zip file
        :param filename: str: name of the zip file
        :return: None
        """
        if self.images is not None:
            # make directory
            os.makedirs('temp')
            # make for every class a directory
            for c in self.classes:
                os.makedirs(os.path.join('temp', c))
            # save the images
            for c in self.classes:
                for i in range(len(self.images[c])):
                    cv2.imwrite(os.path.join('temp', c, self.image_name[c][i]), cv2.cvtColor(self.images[c][i], cv2.COLOR_RGB2BGR))
            if self.coco is not None:
                json_object = json.dumps(self.coco, default=self._np_encoder)
                # save the json object in the temp folder
                with open(os.path.join('temp', 'annotations.json'), 'w') as outfile:
                    outfile.write(json_object)
            # zip the directory
            shutil.make_archive(filename, 'zip', 'temp')
            # remove the directory
            shutil.rmtree('temp')
            print('Images saved')
        else:
            print('No images to save')

    def plot(self, image_class: int, image: int) -> None:
        """
        Plot the image and the mask
        :param image_class: int: class of the image
        :param image: int: index of the image
        :return: None
        """
        # make a subplot with 3 images
        fig, ax = plt.subplots(1, 2, figsize=(15, 5))
        # plot the original image
        ax[0].imshow(self.images[image_class][image])
        ax[0].set_title('initial image')
        # plot the mask if it exists
        if self.mask is not None:
            ax[1].imshow(self.mask[image_class][image])
            ax[1].set_title('mask')

    def get_mask(self, threshold: int = 80, morph: tuple = (None, None)) -> None:
        """
        Create mask for the objects
        :param threshold: int: threshold for the background
        :param morph: tuple: (function, kernel size) for morphological transformation
        :return: None
        """
        self.mask = dict()
        background = self._get_background_RGB()
        for c in self.classes:
            if c != '0':
                self.mask[c] = [np.zeros_like(self.images[c][i]) for i in range(len(self.images[c]))]
                for i in range(len(self.images[c])):
                    # make a deep copy of the image
                    image = self.images[c][i].copy()
                    image = self._correct_image(image, background)
                    # get the mask
                    mask = np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)
                    # if image - background > threshold, then it is 1 else 0
                    mask[np.sum(np.abs(image - np.mean(background, axis=0)), axis=2) > threshold] = 1
                    # remove noise
                    if morph[0] is not None:
                        mask = morph[0](mask, kernel_size=morph[1])
                    self.mask[c][i] = mask
        print('Mask created')
        return None

    def make_coco_json(self, information: dict = None) -> None:
        if information:
            info = information
        else:
            info = {
                'description': str(),
                'url': str(),
                'version': str(),
                'year': int(),
                'contributor': str(),
                'date_created': str(),
                'categories': [[1, 'example_class']],
                'supercategory': str()
            }

        coco_json = dict()
        coco_json['info'] = dict()
        coco_json['info']['description'] = info['description']
        coco_json['info']['url'] = info['url']
        coco_json['info']['version'] = info['version']
        coco_json['info']['year'] = info['year']
        coco_json['info']['contributor'] = info['contributor']
        coco_json['info']['date_created'] = info['date_created']
        coco_json['licenses'] = dict()
        coco_json['licenses']['url'] = info['url']
        coco_json['licenses']['id'] = 1
        coco_json['licenses']['name'] = info['contributor']
        coco_json['images'] = list()
        coco_json['annotations'] = list()
        coco_json['categories'] = list()
        if information is not None:
            for c in info['categories']:
                coco_json['categories'].append({'id': c[0], 'name': c[1], 'supercategory': ''})
        else:
            for c in self.classes:
                if c != '0':
                    coco_json['categories'].append({'id': self.classes.index(c), 'name': c, 'supercategory': ''})

        for c in self.classes:
            if c != '0':
                for i in range(len(self.images[c])):
                    id = i + int(c) * 1000
                    segmentation = self._get_polygon(self.mask[c][i], id=id)
                    coco_json['images'].append({'id': id, 'width': self.images[c][i].shape[1], 'height': self.images[c][i].shape[0], 'file_name': self.image_name[c][i] , 'license': 1, 'flickr_url': '', 'coco_url': '', 'date_captured': ''})
                    coco_json['annotations'].append({'id': id, 'image_id': id, 'category_id': self.classes.index(c), 'segmentation': [segmentation[0]['segmentation_coords']], 'area': segmentation[0]['area'], 'bbox': segmentation[0]['bbox'], 'iscrowd': 0})
        self.coco = coco_json
        return coco_json


class Augmentation:
    def __init__(self, objects: dict, background: list):
        self.background = background
        self.objects = objects
        self.classes = list(objects.keys())
        self.augmentations = None
        self.image_names = None
        self.coco = None

    def _resize_object(self, object, factor):
        height, width = object.shape[:2]
        new_height, new_width = int(height * factor), int(width * factor)
        resized_object = cv2.resize(object, (new_width, new_height))
        return resized_object

    def _rotate_object(self, object, angle, center=None):
        height, width = object.shape[:2]
        if center is None:
            center = (width / 2, height / 2)
        rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1)
        rotated_object = cv2.warpAffine(object, rotation_matrix, (width, height))
        return rotated_object

    def _adjust_hue(self, image, hue_shift):
        hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        hsv_image[:, :, 0] = (hsv_image[:, :, 0] + hue_shift) % 180
        adjusted_image = cv2.cvtColor(hsv_image, cv2.COLOR_HSV2BGR)
        return adjusted_image

    def _get_polygon(self, object):
        # trun object to binary
        object = cv2.cvtColor(object.copy(), cv2.COLOR_BGR2GRAY)
        object = cv2.threshold(object, 0, 255, cv2.THRESH_BINARY)[1]
        # find contours
        polygon = ImageCollection._get_polygon(object, id=0)
        return polygon

    def _place_objects_on_grid(self, object, width, height):
        grid = np.zeros((height, width, 3), dtype=np.uint8)
        objekt_h, objekt_w = object.shape[:2]
        x = random.randint(0, width - objekt_w)
        y = random.randint(0, height - objekt_h)
        # insert object into grid
        grid[y:y+objekt_h, x:x+objekt_w] = object
        return grid

    def _stack_grids(self, grids):
        stacked_grid = np.zeros_like(grids[0])
        for grid in grids:
            # Find the non-zero pixels in the current grid
            non_zero_pixels = np.where(grid != 0)
            # Update the corresponding pixels in the stacked grid with the current grid
            stacked_grid[non_zero_pixels] = grid[non_zero_pixels]
        return stacked_grid

    def polygons_to_coco(self, polygons, augmentations, classes):
        def get_list_with_random_names_for_augmentations():
            names = list()
            for i in range(len(augmentations)):
                names.append(str(i))
            return names

        info = {
                    'description': 'surgical_tools_dataset',
                    'url': str(),
                    'version': '1.0',
                    'year': 2023,
                    'contributor': 'sts',
                    'date_created': '11.05.2023',
                    'categories': [
                        [1, 'instrument_1'],
                        [2, 'instrument_2'],
                        [3, 'instrument_3'],
                        [4, 'instrument_4'],
                        [5, 'instrument_5'],
                        [6, 'instrument_6'],
                        [7, 'instrument_7']
                    ],
                    'supercategory': 'surgical_instuments'
                }
        coco_json = dict()
        coco_json['info'] = dict()
        coco_json['info']['description'] = info['description']
        coco_json['info']['url'] = info['url']
        coco_json['info']['version'] = info['version']
        coco_json['info']['year'] = info['year']
        coco_json['info']['contributor'] = info['contributor']
        coco_json['info']['date_created'] = info['date_created']
        coco_json['licenses'] = dict()
        coco_json['licenses']['url'] = info['url']
        coco_json['licenses']['id'] = 1
        coco_json['licenses']['name'] = info['contributor']
        coco_json['images'] = list()
        coco_json['annotations'] = list()
        coco_json['categories'] = list()

        for c in info['categories']:
            coco_json['categories'].append({'id': c[0], 'name': c[1], 'supercategory': ''})

        names = get_list_with_random_names_for_augmentations()

        for i in range(len(augmentations)):
            coco_json['images'].append({'id': i, 'width': self.augmentations[i].shape[1], 'height': self.augmentations[i].shape[0], 'file_name': names[i] + '.jpg', 'license': 1, 'flickr_url': '', 'coco_url': '', 'date_captured': ''})

            for j, p in enumerate(polygons[i]):
                coco_json['annotations'].append({'id': j+i*0, 'image_id': i, 'category_id': classes[i][j], 'segmentation': [p[0]['segmentation_coords']], 'area': p[0]['area'], 'bbox': p[0]['bbox'], 'iscrowd': 0})

        return coco_json, names


    def augment(
            self,
            num_augmentation: int,
            max_num_objects: int,
            size: tuple or None = None,
            hue_shift: tuple or None = None,
            rotation: bool = True
    ):
        polygons_of_augmentations = list()
        classes_of_augmentations = list()
        self.augmentations = list()
        for i in range(num_augmentation):
            # choose random background
            background = random.choice(self.background)
            # choose random objects
            num_objects = random.randint(1, max_num_objects)
            objects = list()
            classes = list()
            for j in range(num_objects):
                # randomly choose an class
                object_class = random.choice(self.classes)
                classes.append(object_class)
                # randomly choose an object of the class
                object = random.choice(self.objects[object_class])
                objects.append(object)
            # resize objects but resize all objects with the same factor
            if size is not None:
                factor = random.uniform(size[0], size[1])
                objects = [self._resize_object(o, factor) for o in objects]
            # rotate objects
            if rotation:
                objects = [self._rotate_object(o, random.randint(0, 360)) for o in objects]
            # place objects on grid
            grids = [self._place_objects_on_grid(o, background.shape[1], background.shape[0]) for o in objects]
            # get polygons
            polygons = [self._get_polygon(g) for g in grids]
            # stack grids
            stacked_grids = self._stack_grids(grids)
            # put the stacked grid on the background so where the grid is black the background is visible
            stacked_grids[stacked_grids == 0] = background[stacked_grids == 0]
            # change hue
            if hue_shift is not None:
                stacked_grids = self._adjust_hue(stacked_grids, random.randint(hue_shift[0], hue_shift[1]))
            # return stacked grid and polygons
            self.augmentations.append(stacked_grids)
            polygons_of_augmentations.append(polygons)
            classes_of_augmentations.append(classes)
            self.coco, self.image_names = self.polygons_to_coco(polygons_of_augmentations, self.augmentations, classes_of_augmentations)
        return self.coco, self.image_names, self.augmentations

    def save(self, filename: str = 'augmentations') -> None:
        """
        Save the augmentations and annotations to a zip file
        :param filename: str: name of the zip file
        :return: None
        """
        if self.augmentations is not None:
            # make directory
            os.makedirs('temp')
            # save the images
            for i, img in enumerate(self.augmentations):
                cv2.imwrite(os.path.join('temp', self.image_names[i]) + '.jpg', cv2.cvtColor(self.augmentations[i], cv2.COLOR_BGR2RGB))
            if self.coco is not None:
                json_object = json.dumps(self.coco, default=ImageCollection._np_encoder)
                # save the json object in the temp folder
                with open(os.path.join('temp', 'annotations.json'), 'w') as outfile:
                    outfile.write(json_object)
            # zip the directory
            shutil.make_archive(filename, 'zip', 'temp')
            # remove the directory
            shutil.rmtree('temp')
            print('Images saved')
        else:
            print('No images to save')

def opening(img: np.ndarray, kernel_size: int = 3) -> np.ndarray:
     """
     Remove noise from the image
     :param img: np.ndarray: image
     :param kernel_size: int: size of the kernel
     :return: np.ndarray: image with noise removed
     """
     kernel = np.ones((kernel_size, kernel_size), np.uint8)
     return cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)

def closing(img: np.ndarray, kernel_size: int = 3) -> np.ndarray:
     """
     Remove noise from the image
     :param img: np.ndarray: image
     :param kernel_size: int: size of the kernel
     :return: np.ndarray: image with noise removed
     """
     kernel = np.ones((kernel_size, kernel_size), np.uint8)
     return cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)