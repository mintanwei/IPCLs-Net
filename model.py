import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor, TwoMLPHead
from torchvision.ops.poolers import MultiScaleRoIAlign, MultiScaleRoIAlignWithKNN, MultiScaleRoIAlignWithDistance

def rcnn_model(nob3=False, box_score_thresh=0.3, box_nms_thresh=0.3, box_detections_per_img=250, knn_features=True, k=4):
    box_roi_pool = None
    box_head = None
    if knn_features:
        box_roi_pool = MultiScaleRoIAlignWithKNN(
            featmap_names=[0, 1, 2, 3],
            output_size=7,
            sampling_ratio=2,
            knn=k)

        out_channels = 256 * k
        resolution = box_roi_pool.output_size[0]
        representation_size = 1024

        box_head = TwoMLPHead(
            out_channels * resolution ** 2,
            representation_size)

    # load a model pre-trained pre-trained on COCO
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(
        pretrained=False, box_detections_per_img=box_detections_per_img,
        box_score_thresh=box_score_thresh, box_nms_thresh=box_nms_thresh, box_roi_pool=box_roi_pool, box_head=box_head)

    # replace the classifier with a new one, that has
    # num_classes which is user-defined
    if nob3:
        num_classes = 4
    else:
        num_classes = 5  # 4 class (A/B1/B2/B3) + background

    # get number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    # replace the pre-trained head with a new one
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    return model

def rcnn_distance_model(box_score_thresh=0.3, box_nms_thresh=0.3, box_detections_per_img=250, k=4, p=2):
    box_roi_pool = MultiScaleRoIAlignWithDistance(
        featmap_names=[0, 1, 2, 3],
        output_size=7,
        sampling_ratio=2,
        k=k,
        p=p)

    out_channels = 256
    resolution = box_roi_pool.output_size[0]
    representation_size = 1024

    box_head = TwoMLPHead(
        out_channels * resolution ** 2,
        representation_size)

    # load a model pre-trained pre-trained on COCO
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(
        pretrained=False, box_detections_per_img=box_detections_per_img,
        box_score_thresh=box_score_thresh, box_nms_thresh=box_nms_thresh, box_roi_pool=box_roi_pool, box_head=box_head)

    # replace the classifier with a new one, that has
    # num_classes which is user-defined
    num_classes = 5  # 4 class (A/B1/B2/B3) + background

    # get number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    # replace the pre-trained head with a new one
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    return model

if __name__ == '__main__':
    print(rcnn_model)