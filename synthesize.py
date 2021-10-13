from copy_paste import CopyPaste
from coco import CocoDetectionCP
import albumentations as A

transformScene = A.Compose([
        #A.RandomScale(scale_limit=(-0.9, 1), p=1), #LargeScaleJitter from scale of 0.1 to 2
        A.PadIfNeeded(256, 256, border_mode=0), #pads with image in the center, not the top left like the paper
        # A.Resize(534, 889, always_apply=True, p=1),
        A.Resize(800, 1333, always_apply=True, p=1),
    ]
)


transform = A.Compose([
        A.RandomScale(scale_limit=(-0.9, 1), p=1), #LargeScaleJitter from scale of 0.1 to 2
        A.PadIfNeeded(256, 256, border_mode=0), #pads with image in the center, not the top left like the paper
        A.Flip(p=0.5),
        A.Resize(800, 1333, always_apply=True, p=1),
        #A.Solarize(always_apply=True, p=1.0, threshold=(128, 128)),
        # A.RandomCrop(534, 889),
        #A.Resize(800, 1333, always_apply=True, p=1),
        # pct_objects is the percentage of objects to paste over
        CopyPaste(blend=True, sigma=1, pct_objects_paste=1, p=1., gray=False)
    ], bbox_params=A.BboxParams(format="coco")
)


#data = CocoDetectionCP(
#    '../agilent-repos/mmdetection/data/bead_cropped_detection/images',
#    '../agilent-repos/mmdetection/data/custom/object-classes.json',
#    transform
#)
data = CocoDetectionCP(
    '../Swin-Transformer-Object-Detection/data/flooding_high_cropped',
    '../Swin-Transformer-Object-Detection/data/w_nebulizer_save_4',
    #'../Swin-Transformer-Object-Detection/data/beading_basler',
    #'../Swin-Transformer-Object-Detection/data/basler_bead_non_cropped.json',
    '../Swin-Transformer-Object-Detection/data/w_nebulizer_train_iter1.json',
    transform,
    transformScene,
    gray=False
)

# Synthesize a new dataset using all of the scene images, and the available
# annotations from the objects images (randomly).
data.synthesize(pct=0.03,
                out_dir="data/out_02_pct_w_nebulizer/",
                json_name="data/out_02_pct_w_nebulizer.json")

