from mmdet.apis import init_detector, inference_detector, show_result

config_file = 'configs/faster_rcnn_r50_fpn_1x_test.py'
checkpoint_file = '/home/zhangyu/mmdetection/work_dirs/faster_rcnn_r50_fpn_1x/latest.pth'

# build the model from a config file and a checkpoint file
model = init_detector(config_file, checkpoint_file, device='cuda:0')

# test a single image and show the results
img = 'test.jpg'  # or img = mmcv.imread(img), which will only load it once
result = inference_detector(model, img)
show_result(img, result, model.CLASSES, out_file='test_dec.jpg',show=False)