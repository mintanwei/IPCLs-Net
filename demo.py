import cv2
import numpy
import random
from torchvision import transforms
import torch
# from engine import evaluate


def random_color():
    b = random.randint(0, 255)
    g = random.randint(0, 255)
    r = random.randint(0, 255)

    return (b, g, r)


def compute_iou(box1, box2):
    # box [x1, y1, x2, y2]
    box1 = box1.to(torch.device("cpu"))
    box2 = box2.to(torch.device("cpu"))
    rec1 = (box1[1], box1[0], box1[3], box1[2])
    rec2 = (box2[1], box2[0], box2[3], box2[2])

    # computing area of each rectangles
    S_rec1 = (rec1[2] - rec1[0]) * (rec1[3] - rec1[1])
    S_rec2 = (rec2[2] - rec2[0]) * (rec2[3] - rec2[1])

    # computing the sum_area
    sum_area = S_rec1 + S_rec2

    # find the each edge of intersect rectangle
    left_line = max(rec1[1], rec2[1])
    right_line = min(rec1[3], rec2[3])
    top_line = max(rec1[0], rec2[0])
    bottom_line = min(rec1[2], rec2[2])

    # judge if there is an intersect
    if left_line >= right_line or top_line >= bottom_line:
        return 0
    else:
        intersect = (right_line - left_line) * (bottom_line - top_line)
        return (intersect / (sum_area - intersect)) * 1.0



def stat(labels, type):
    classes = ['gt', 'A', 'B1', 'B2', 'B3']
    stat_dict = {'A': 0, 'B1': 0, 'B2': 0, 'B3': 0}
    prop_dict = {}
    for label in labels:
        stat_dict[classes[label]] += 1

    for k, v in stat_dict.items():
        prop_dict[k] = v / len(labels)

    print("[%s] A: %d (%.3f), B1: %d (%.3f), B2: %d (%.3f), B3: %d (%.3f)" %
          (type,
           stat_dict['A'], prop_dict['A'],
           stat_dict['B1'], prop_dict['B1'],
           stat_dict['B2'], prop_dict['B2'],
           stat_dict['B3'], prop_dict['B3'])
          )


def visualize(model, data_loader, device):
    model.eval()
    classes = ['gt', 'A', 'B1', 'B2', 'B3']
    colors = [(0, 0, 0), (0, 255, 0), (0, 0, 255), (255, 0, 255), (255, 255, 255)]
    # images, targets = next(iter(data_loader))
    precision_correct = [0, 0, 0, 0, 0]
    precision_total = [0, 0, 0, 0, 0]
    recall_correct = [0, 0, 0, 0, 0]
    recall_total = [0, 0, 0, 0, 0]
    sensitivity = [0, 0, 0, 0, 0]

    for count, (images, targets) in enumerate(data_loader):
        gt_boxes = targets[0]["boxes"]
        gt_labels = targets[0]["labels"]
        PIL_img = transforms.ToPILImage()(images[0])
        gt_img = cv2.cvtColor(numpy.asarray(PIL_img), cv2.COLOR_RGB2BGR)
        rs_img = cv2.cvtColor(numpy.asarray(PIL_img), cv2.COLOR_RGB2BGR)
        abs_img = cv2.cvtColor(numpy.asarray(PIL_img), cv2.COLOR_RGB2BGR)

        cv2.imwrite('results/%d.png' % count, gt_img)

        images = list(image.to(device) for image in images)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        # images = images[0]
        # targets = targets[0]

        # print(images)
        # print(targets)

        out = model(images, targets)
        # print(out)
        boxes = out[0]['boxes']
        labels = out[0]['labels']
        scores = out[0]['scores']

        # stat(gt_labels, 'GT')
        # stat(labels, 'PD')

        for idx in range(gt_boxes.shape[0]):
            x1, y1, x2, y2 = gt_boxes[idx][0], gt_boxes[idx][1], gt_boxes[idx][2], gt_boxes[idx][3]
            name = classes[gt_labels[idx].item()]
            cv2.rectangle(gt_img, (x1, y1), (x2, y2), colors[gt_labels[idx].item()], thickness=2)
            cv2.putText(gt_img, text=name, org=(x1, y1 + 15), fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                        fontScale=0.5, thickness=1, lineType=cv2.LINE_AA, color=colors[gt_labels[idx].item()])

        for idx in range(boxes.shape[0]):
            x1, y1, x2, y2 = boxes[idx][0], boxes[idx][1], boxes[idx][2], boxes[idx][3]
            name = classes[labels[idx].item()]
            cv2.rectangle(rs_img, (x1, y1), (x2, y2), colors[labels[idx].item()], thickness=2)
            cv2.putText(rs_img, text=name + " %.3f" % scores[idx], org=(x1, y1 + 15), fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                        fontScale=0.5, thickness=1, lineType=cv2.LINE_AA, color=colors[labels[idx].item()])

        # cv2.imshow('result', src_img)
        # cv2.waitKey()
        # cv2.destroyAllWindows()

        # print()
        # print("图%d" % (count + 1))
        precision_correct_add, precision_total_add, recall_correct_add, recall_total_add, unlabeled_index = statistics(gt_boxes, gt_labels, boxes, labels)
        for cls_id in range(1, 5):
            precision_correct[cls_id] += precision_correct_add[cls_id]
            precision_total[cls_id] += precision_total_add[cls_id]
            recall_correct[cls_id] += recall_correct_add[cls_id]
            recall_total[cls_id] += recall_total_add[cls_id]

        for idx in unlabeled_index:
            x1, y1, x2, y2 = boxes[idx][0], boxes[idx][1], boxes[idx][2], boxes[idx][3]
            name = classes[labels[idx].item()]
            cv2.rectangle(abs_img, (x1, y1), (x2, y2), colors[labels[idx].item()], thickness=2)
            cv2.putText(abs_img, text=name + " %.3f" % scores[idx], org=(x1, y1 + 15), fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                        fontScale=0.5, thickness=1, lineType=cv2.LINE_AA, color=colors[labels[idx].item()])

        cv2.imwrite('results/%d_gt.png' % count, gt_img)
        cv2.imwrite('results/%d_pd.png' % count, rs_img)
        # cv2.imwrite('results/%d_abs.png' % count, abs_img)

    ep = 1e-5
    TP = [0, 0, 0, 0, 0]
    FP = [0, 0, 0, 0, 0]
    FN = [0, 0, 0, 0, 0]

    import numpy as np
    matrix = np.zeros([5,6])

    for cls_id in range(1, 5):
        TP[cls_id] = precision_correct[cls_id]
        FP[cls_id] = precision_total[cls_id] - precision_correct[cls_id]
        FN[cls_id] = recall_total[cls_id] - recall_correct[cls_id]
        sensitivity[cls_id] = (TP[cls_id] + ep) / (TP[cls_id] + FN[cls_id] + ep)
        print("Class %s" % classes[cls_id])
        print("TP:%d, FP:%d, FN:%d, Sensitivity=%.3f, Precision=%.3f, Recall=%.3f" %
              (TP[cls_id], FP[cls_id], FN[cls_id], sensitivity[cls_id],
               precision_correct[cls_id] / (precision_total[cls_id] + ep), recall_correct[cls_id] / (recall_total[cls_id] + ep)))
        matrix[cls_id - 1] = [TP[cls_id], FP[cls_id], FN[cls_id], sensitivity[cls_id],
               precision_correct[cls_id] / (precision_total[cls_id] + ep), recall_correct[cls_id] / (recall_total[cls_id] + ep)]


    print("Overall")
    print("TP:%d, FP:%d, FN:%d, Sensitivity=%.3f, Precision=%.3f, Recall=%.3f" %
          (sum(TP), sum(FP), sum(FN), sum(TP) / (sum(TP) + sum(FN)),
           sum(precision_correct) / sum(precision_total), sum(recall_correct) / sum(recall_total)))
    matrix[4] = [sum(TP), sum(FP), sum(FN), sum(TP) / (sum(TP) + sum(FN)),
           sum(precision_correct) / sum(precision_total), sum(recall_correct) / sum(recall_total)]

    return matrix

def statistics(gt_boxes, gt_labels, pd_boxes, pd_labels):
    # print(gt_boxes)
    # print(gt_labels)
    # print(pd_boxes)
    # print(pd_labels)
    classes = ['gt', 'A', 'B1', 'B2', 'B3']

    # calculate recall

    gt_boxes_count = gt_boxes.shape[0]
    pd_boxes_count = pd_boxes.shape[0]
    recall_total = [0, 0, 0, 0, 0]
    recall_correct = [0, 0, 0, 0, 0]
    for idx in range(gt_boxes_count):
        all_ious = {}
        for pd_idx in range(pd_boxes_count):
            iou = compute_iou(gt_boxes[idx], pd_boxes[pd_idx])
            if iou > 0:
                all_ious[pd_idx] = iou

        gt_class = gt_labels[idx].item()
        recall_total[gt_class] += 1

        if len(all_ious) > 0:
            max_iou_idx = sorted(all_ious.items(), key=lambda x: x[1], reverse=True)[0][0]
            max_iou_pd_class = pd_labels[max_iou_idx].item()
            if max_iou_pd_class == gt_class:
                recall_correct[gt_class] += 1

    # print(recall_total)
    # print(recall_correct)

    # calculate precision
    precision_total = [0, 0, 0, 0, 0]
    precision_correct = [0, 0, 0, 0, 0]
    unlabeled_index = []
    for idx in range(pd_boxes_count):
        all_ious = {}
        for gt_idx in range(gt_boxes_count):
            iou = compute_iou(pd_boxes[idx], gt_boxes[gt_idx])
            if iou > 0:
                all_ious[gt_idx] = iou

        pd_class = pd_labels[idx].item()
        precision_total[pd_class] += 1

        if len(all_ious) > 0:
            max_iou_idx = sorted(all_ious.items(), key=lambda x: x[1], reverse=True)[0][0]
            max_iou_gt_class = gt_labels[max_iou_idx].item()
            if max_iou_gt_class == pd_class:
                precision_correct[pd_class] += 1
        else:
            unlabeled_index.append(idx)

    # print(precision_total)
    # print(precision_correct)
    # print("Ground Truth (Total):")
    # print("A:%d, B1:%d, B2:%d, B3:%d" % (recall_total[1], recall_total[2], recall_total[3], recall_total[4]))
    # print("Prediction (Total):")
    # print(
    #     "A:%d, B1:%d, B2:%d, B3:%d" % (precision_total[1], precision_total[2], precision_total[3], precision_total[4]))
    # print("Recall:")
    ep = 1e-5
    # for cls_id in [1,2,3,4]:
    #     print(classes[cls_id] + ": " + "%d/%d(%.3f)" % (recall_correct[cls_id], recall_total[cls_id], (recall_correct[cls_id] + ep) /(recall_total[cls_id] + ep)))
    # print("Overall: " + "%d/%d(%.3f)" % (sum(recall_correct), sum(recall_total), sum(recall_correct) / sum(recall_total)))
    # print("Precision:")
    # for cls_id in [1, 2, 3, 4]:
    #     print(classes[cls_id] + ": " + "%d/%d(%.3f)" % (
    #     precision_correct[cls_id], precision_total[cls_id], (precision_correct[cls_id] + ep) / (precision_total[cls_id] + ep)))
    # print(
    #     "Overall: " + "%d/%d(%.3f)" % (sum(precision_correct), sum(precision_total), sum(precision_correct) / sum(precision_total)))

    return precision_correct, precision_total, recall_correct, recall_total, unlabeled_index

if __name__ == '__main__':
    import os
    import torch
    from main import get_transform
    from dataset import NBIDataset
    from torch.utils.data import DataLoader, Subset
    import utils
    import torchvision
    from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    # our dataset has five classes - 4 class (A/B1/B2/B3) + background
    nob3 = False
    if nob3:
        num_classes = 4
    else:
        num_classes = 5

    # use our dataset and defined transformations
    root = "./NBI_dataset"
    dataset_train = NBIDataset(os.path.join(root, "train"), get_transform(train=True),nob3)
    dataset_test = NBIDataset(os.path.join(root, "test"), get_transform(train=False),nob3)

    # define training and validation data loaders
    data_loader_train = DataLoader(dataset_train, batch_size=1, shuffle=True, collate_fn=utils.collate_fn)
    data_loader_test = DataLoader(dataset_test, batch_size=1, shuffle=False, collate_fn=utils.collate_fn)

    # box_score_threshs = [0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5]
    # box_nms_threshs = [0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5]
    box_score_threshs = [0.3]
    box_nms_threshs = [0.3]
    results = []

    # result_matrix = numpy.zeros([10,5,6])
    # for id in range(0, 10):
    #     rcnn_model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True, box_detections_per_img=250,
    #                                                                       box_score_thresh=0.1, box_nms_thresh=0.1)
    #
    #     # get number of input features for the classifier
    #     in_features = rcnn_model.roi_heads.box_predictor.cls_score.in_features
    #     # replace the pre-trained head with a new one
    #     rcnn_model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    #
    #     model = rcnn_model
    #     model.to(device)
    #
    #     model_path = os.path.join("./models", 'model_epoch-50-repeat-%d.pt' % id)
    #     net_state_dict = torch.load(model_path)
    #     model.load_state_dict(net_state_dict)
    #
    #     result_matrix[id] = visualize(model, data_loader_test, device)
    #
    # print(numpy.mean(result_matrix, axis=0))
    # print(numpy.std(result_matrix, axis=0))

    for bst in box_score_threshs:
        for bnt in box_nms_threshs:
            rcnn_model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True, box_detections_per_img=250,
                                                                              box_score_thresh=bst, box_nms_thresh=bnt)

            # get number of input features for the classifier
            in_features = rcnn_model.roi_heads.box_predictor.cls_score.in_features
            # replace the pre-trained head with a new one
            rcnn_model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

            model = rcnn_model
            model.to(device)

            model_path = os.path.join("./models", 'model_epoch-20-ce.pt')
            net_state_dict = torch.load(model_path)
            model.load_state_dict(net_state_dict)

            # ap, ar = evaluate(model, data_loader_test, device=device)
            # results.append("score_thresh:%.2f, box_thresh:%.2f, AP:%.3f, AR:%.3f, F1:%.3f" % (bst, bnt, ap, ar, 2*ap*ar/(ap+ar)))
            # print("ap:%.3f, ar:%.3f" % (ap, ar))
            visualize(model, data_loader_test, device)

    # for string in results:
    #     print(string)