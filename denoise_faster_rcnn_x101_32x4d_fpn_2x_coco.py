_base_ = '/mnt/data/Xia/mmdetection-2.25.2/configs/faster_rcnn/faster_rcnn_x101_32x4d_fpn_2x_coco.py'

# We also need to change the num_classes in head to match the dataset's annotation
model = dict(
    roi_head=dict(
        bbox_head=dict(num_classes=1)))

# Modify dataset related settings
dataset_type = 'COCODataset'
#classes = ('n','pcc',)
classes = ('c',)
data = dict(
    train=dict(
        img_prefix='/mnt/data/Xia/2_234_5_1/denoise/train2017/',
        classes=classes,
        ann_file='/mnt/data/Xia/2_234_5_1/denoise/annotations/instances_train2017.json'),
    val=dict(
        img_prefix='/mnt/data/Xia/2_234_5_1/denoise/val2017/',
        classes=classes,
        ann_file='/mnt/data/Xia/2_234_5_1/denoise/annotations/instances_val2017.json'),
    test=dict(
        img_prefix='/mnt/data/Xia/2_234_5_1/denoise/test2017/',
        classes=classes,
        ann_file='/mnt/data/Xia/2_234_5_1/denoise/annotations/instances_test2017.json'))

# We can use the pre-trained Mask RCNN model to obtain higher performance
load_from = 'resnext101_32x4d-a5af3160.pth'