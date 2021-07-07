import os
import cv2
import copy
import json
import random
from pathlib import Path
from torchvision.datasets import CocoDetection
# from copy_paste import copy_paste_class
import albumentations as A
import numpy as np
from datetime import datetime
from pycocotools import mask as maskUtils

min_keypoints_per_image = 10

def _count_visible_keypoints(anno):
    return sum(sum(1 for v in ann["keypoints"][2::3] if v > 0) for ann in anno)

def _has_only_empty_bbox(anno):
    return all(any(o <= 1 for o in obj["bbox"][2:]) for obj in anno)

def has_valid_annotation(anno):
    # if it's empty, there is no annotation
    if len(anno) == 0:
        return False
    # if all boxes have close to zero area, there is no annotation
    if _has_only_empty_bbox(anno):
        return False
    # keypoints task have a slight different critera for considering
    # if an annotation is valid
    if "keypoints" not in anno[0]:
        return True
    # for keypoint detection tasks, only consider valid images those
    # containing at least min_keypoints_per_image
    if _count_visible_keypoints(anno) >= min_keypoints_per_image:
        return True

    return False

#@copy_paste_class
class CocoDetectionCP():
    def __init__(
        self,
        sceneRoot,  # Image files (empty scene)
        objectRoot,  # Image files (contains annotation objects in scene)
        annFile,  # Annotations file
        transforms,  # List of transforms for objs
        transformScene  # List of transforms for the scenes
    ):
        # super(CocoDetectionCP, self).__init__(
        #     sceneRoot, objectRoot, annFile, None, None, transforms
        # )
        self.sceneRoot = sceneRoot
        self.objectRoot = objectRoot
        self.transforms = transforms
        self.transformScenes = transformScene
        self.c = CocoDetection(objectRoot, annFile, transforms)
        self.json_base = self.init_json()
        self.albu_images = []  # store of the cv2 edited images


        # Get filenames in the root directory

        # filter images without detection annotations
        ids = []
        for img_id in self.c.ids:
            ann_ids = self.c.coco.getAnnIds(imgIds=img_id, iscrowd=None)
            # anno = self.c.coco.loadAnns(ann_ids)
            # if has_valid_annotation(anno):
            #    ids.append(img_id)
            ids.append(img_id)
        self.ids = ids

        self.scene_names = [filename for
                            filename in os.listdir(self.sceneRoot)]

    def __len__(self):
        # Override the len function.
        # return: number of scene images.
        return len(self.scene_names)

    def _split_transforms(self):
        # This determines if there is a 'CopyPaste' transform in the
        # list of transforms, and will split the transforms such that
        # the CopyPaste transform (tf) will occur last.

        # Determine the index of CopyPaste tf (if it exists)
        split_index = None
        for ix, tf in enumerate(list(self.transforms.transforms)):
            if tf.get_class_fullname() == 'copypaste.CopyPaste':
                split_index = ix

        if split_index is not None:
            tfs = list(self.transforms.transforms)
            pre_copy = tfs[:split_index]
            copy_paste = tfs[split_index]
            post_copy = tfs[split_index+1:]

            #replicate the other augmentation parameters
            bbox_params = None
            keypoint_params = None
            paste_additional_targets = {}
            if 'bboxes' in self.transforms.processors:
                bbox_params = self.transforms.processors['bboxes'].params
                paste_additional_targets['paste_bboxes'] = 'bboxes'
                if self.transforms.processors['bboxes'].params.label_fields:
                    msg = "Copy-paste does not support bbox label_fields! "
                    msg += "Expected bbox format is (a, b, c, d, label_field)"
                    raise Exception(msg)
            if 'keypoints' in self.transforms.processors:
                keypoint_params = self.transforms.processors[
                    'keypoints'].params
                paste_additional_targets['paste_keypoints'] = 'keypoints'
                if keypoint_params.label_fields:
                    raise Exception('Copy-paste does not support keypoint label fields!')

            if self.transforms.additional_targets:
                raise Exception('Copy-paste does not support additional_targets!')

            #recreate transforms
            self.transforms = A.Compose(pre_copy, bbox_params,
                                         keypoint_params, additional_targets=None)
            self.post_transforms = A.Compose(post_copy, bbox_params,
                                              keypoint_params, additional_targets=None)
            self.copy_paste = A.Compose(
                [copy_paste], bbox_params, keypoint_params, additional_targets=paste_additional_targets
            )
        else:
            self.copy_paste = None
            self.post_transforms = None

    def __getitem__(self, idx):
        #split transforms if it hasn't been done already
        if not hasattr(self, 'post_transforms'):
            self._split_transforms()

        # Scene image data
        scene_id = idx
        scene_data = self.load_example(idx, scene=True, scene_index=scene_id)

        if self.copy_paste is not None:
            paste_idx = random.randint(0, self.c.__len__() - 1)
            # paste_idx = 5  # hardcode for testing
            paste_img_data = self.load_example(paste_idx)
            for k in list(paste_img_data.keys()):
                paste_img_data['paste_' + k] = paste_img_data[k]
                del paste_img_data[k]

            combine_data = self.copy_paste(**scene_data, **paste_img_data)
            combine_data = self.post_transforms(**combine_data)
            combine_data['paste_index'] = paste_idx

            # Get COCO ann format data
            licenses = copy.deepcopy(self.c.coco.dataset['licenses'])
            categories = [copy.deepcopy(self.c.coco.cats[1])]
            im_meta = {
                "id": scene_id,
                "width": combine_data['image'].shape[1],
                "height": combine_data['image'].shape[0],
                #"file_name": self.scene_names[scene_id],
                "file_name": 'combine_' + str(scene_id),
                "license": None,
                "flickr_url": "",
                "coco_url": None,
                "date_captured": str(datetime.now())
            }
            # images = [im_meta]
            self.add_img_json(im_meta)

            # attempt to get RLE counts for each of the transformed masks
            # in the combined data:
            new_anns = []
            rle_masks = []

            # There are as many annotations as there are pasted masks
            for ix, paste_mask in enumerate(combine_data['masks']):
                # to Fortran contiguous
                # See: https://github.com/cocodataset/cocoapi/issues/91
                contig_mask = np.asfortranarray(paste_mask)
                paste_rle = maskUtils.encode(contig_mask)
                # In Python3
                # See: https://github.com/cocodataset/cocoapi/issues/70
                paste_rle['counts'] = paste_rle['counts'].decode('ascii')

                rle_masks.append(paste_rle)

                # areas
                area = int(maskUtils.area(paste_rle))
                bbox = maskUtils.toBbox(paste_rle)
                bbox = bbox.tolist()  # ndarray to list
                bbox = [int(i) for i in bbox]

                new_ann = {
                    "id": ix,
                    "image_id": scene_id,
                    "category_id": categories[0]['id'],  # 1 == beading
                    "segmentation": paste_rle,
                    "area": area,
                    "bbox": bbox,
                    "iscrowd": 0  # always for individual instance seg.
                }
                # new_anns.append(new_ann)
                self.add_anno_json(new_ann)

            # Annotation for the new combined scene and target masks/bboxes.
            # combine_ann = {
            #     "licenses": licenses,
            #     "categories": categories,
            #     "images": images,
            #     "annotations": new_anns,
            # }
            #
            # combine_data['annotation'] = combine_ann


        # Required annotation file format |-> source
        # info (dict) |-> original ann
        # licenses [] |-> original ann
        # categories [(dict)] |-> original ann
        # images [(dict)]: scene image ann
        #       - get following info after transforms. Keys are str
        #       - id:
        #       - width: px
        #       - height: px
        #       - file_name: str
        #       - license: null
        #       - flickr_url: ""
        #       - coco_url: null
        #       - date_captured: system.date()
        # annotations [(dict)]: original ann after transforms
        #       - id: unique
        #       - image_id: (corresponding image id, scene)
        #       - category_id: 1 (beading)
        #       - segmentation (dict):
        #           - size: after tf
        #           - counts (str): RLE of mask
        #           - area (num): after tf
        #           - bbox [len(4)]
        #           - iscrowd: 0 (for all)
        self.add_image(combine_data['image'])
        return combine_data

    def load_example(self, index, scene=False, scene_index=5):
        """
        Load an example with annotations.
        index:: (int) the index of the image from ids
        scene:: (bool) to load from an empty scene or not.
        """
        img_id = self.c.ids[index]
        ann_ids = self.c.coco.getAnnIds(imgIds=img_id)
        target = self.c.coco.loadAnns(ann_ids)

        if scene:
            scene_path = self.scene_names[scene_index]
            total_path = os.path.abspath(os.path.join(self.sceneRoot, scene_path))
        else:
            path = self.c.coco.loadImgs(img_id)[0]['file_name']
            total_path = os.path.abspath(os.path.join(self.c.root, path))
        image = cv2.imread(total_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        #convert all of the target segmentations to masks
        #bboxes are expected to be (y1, x1, y2, x2, category_id)
        masks = []
        bboxes = []
        if not scene:
            for ix, obj in enumerate(target):
                masks.append(self.c.coco.annToMask(obj))
                bboxes.append(obj['bbox'] + [obj['category_id']] + [ix])

        #pack outputs into a dict
        output = {
            'image': image,
            'masks': masks,
            'bboxes': bboxes
        }

        if scene:
            return self.transformScenes(**output)
        # Otherwise for objects
        return self.transforms(**output)

    def init_json(self):
        """
        Initialise the JSON file for the combined result.
        @return: Object
        """
        return {
            "licenses": copy.deepcopy(self.c.coco.dataset['licenses']),
            "categories": [copy.deepcopy(self.c.coco.cats[1])],
            "images": [],
            "annotations": []
        }

    def write_json(self, f_name):
        """
        Writes the current json_base object to a json file on the system.
        @param f_name: string of the filename to write.
        @return:
        """
        with open(f_name, 'w') as j_file:
            json.dump(self.json_base, j_file, indent=4)

    def add_img_json(self, img):
        """
        Add an COCO image object to the json_base object.

        @param img: object for json image
        @return: None
        """
        if not self.json_base['images'] is None:
            self.json_base['images'].append(img)
        else:
            raise Exception('JSON not initialised.')

    def add_anno_json(self, anno):
        """
        Add a COCO-style annotation to the json_base object.
        @param anno: COCO annotation object.
        @return: None
        """
        if not self.json_base['annotations'] is None:
            self.json_base['annotations'].append(anno)
        else:
            raise Exception('JSON not initialised.')

    def download_img(self, img, f_name, out_dir='./'):
        """
        Download a cv2 image to the local file system.
        @param img: (cv2) image array
        @param f_name: (str) filename
        @param out_dir: (str) directory to store
        @return:
        """
        out_path = os.path.join(out_dir, Path(f_name).name)
        cv2.imwrite(out_path, img)

    def add_image(self, img):
        """
        Add a cv2 image to the class store. This does not store on the file-
        system.

        @param img: cv2 image array
        @return: None
        """
        self.albu_images.append(img)
