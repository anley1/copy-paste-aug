import os
import cv2
from torchvision.datasets import CocoDetection
from copy_paste import copy_paste_class

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

@copy_paste_class
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


        # Get filenames in the root directory

        # filter images without detection annotations
        ids = []
        for img_id in self.c.ids:
            ann_ids = self.c.coco.getAnnIds(imgIds=img_id, iscrowd=None)
            anno = self.c.coco.loadAnns(ann_ids)
            if has_valid_annotation(anno):
                ids.append(img_id)
        self.ids = ids

        self.scene_names = [filename for
                            filename in os.listdir(self.sceneRoot)]

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