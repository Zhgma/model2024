_base_ = '/mnt/data/Xia/mmdetection-2.25.2/configs/mask_rcnn/mask_rcnn_r50_fpn_2x_coco.py'

# We also need to change the num_classes in head to match the dataset's annotation
model = dict(
    roi_head=dict(
        bbox_head=dict(num_classes=1),
        mask_head=dict(num_classes=1)))

# Modify dataset related settings
dataset_type = 'COCODataset'
classes = ('c',)
data = dict(
    train=dict(
        img_prefix='/mnt/data/Xia/2_234_5_1/segmentation/coco/train2017/',
        classes=classes,
        ann_file='/mnt/data/Xia/2_234_5_1/segmentation/coco/annotations/instances_train2017.json'),
    val=dict(
        img_prefix='/mnt/data/Xia/2_234_5_1/segmentation/coco/val2017/',
        classes=classes,
        ann_file='/mnt/data/Xia/2_234_5_1/segmentation/coco/annotations/instances_val2017.json'),
    test=dict(
        img_prefix='/mnt/data/Xia/2_234_5_1/segmentation/coco/val2017/',
        classes=classes,
        ann_file='/mnt/data/Xia/2_234_5_1/segmentation/coco/annotations/instances_val2017.json'))

# We can use the pre-trained Mask RCNN model to obtain higher performance
#load_from = 'checkpoints/mask_rcnn_r50_caffe_fpn_mstrain-poly_3x_coco_bbox_mAP-0.408__segm_mAP-0.37_20200504_163245-42aa3d00.pth'