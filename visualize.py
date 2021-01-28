import cv2
import numpy
from torchvision import transforms
import torch
import numpy as np


def compute_iou(box1, box2):
    # box [x1, y1, x2, y2]
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


def find_intersection(set_1, set_2):
    """
    Find the intersection of every box combination between two sets of boxes that are in boundary coordinates.

    :param set_1: set 1, a tensor of dimensions (n1, 4)
    :param set_2: set 2, a tensor of dimensions (n2, 4)
    :return: intersection of each of the boxes in set 1 with respect to each of the boxes in set 2, a tensor of dimensions (n1, n2)
    """

    # PyTorch auto-broadcasts singleton dimensions
    lower_bounds = torch.max(set_1[:, :2].unsqueeze(1), set_2[:, :2].unsqueeze(0))  # (n1, n2, 2)
    upper_bounds = torch.min(set_1[:, 2:].unsqueeze(1), set_2[:, 2:].unsqueeze(0))  # (n1, n2, 2)
    intersection_dims = torch.clamp(upper_bounds - lower_bounds, min=0)  # (n1, n2, 2)
    return intersection_dims[:, :, 0] * intersection_dims[:, :, 1]  # (n1, n2)


def find_jaccard_overlap(set_1, set_2):
    """
    Find the Jaccard Overlap (IoU) of every box combination between two sets of boxes that are in boundary coordinates.

    :param set_1: set 1, a tensor of dimensions (n1, 4)
    :param set_2: set 2, a tensor of dimensions (n2, 4)
    :return: Jaccard Overlap of each of the boxes in set 1 with respect to each of the boxes in set 2, a tensor of dimensions (n1, n2)
    """

    # Find intersections
    intersection = find_intersection(set_1, set_2)  # (n1, n2)

    # Find areas of each box in both sets
    areas_set_1 = (set_1[:, 2] - set_1[:, 0]) * (set_1[:, 3] - set_1[:, 1])  # (n1)
    areas_set_2 = (set_2[:, 2] - set_2[:, 0]) * (set_2[:, 3] - set_2[:, 1])  # (n2)

    # Find the union
    # PyTorch auto-broadcasts singleton dimensions
    union = areas_set_1.unsqueeze(1) + areas_set_2.unsqueeze(0) - intersection  # (n1, n2)

    return intersection / union  # (n1, n2)


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
    return stat_dict

@torch.no_grad()
def visualize(model, data_loader, device, generate_result=True, k=0):
    model.eval()
    classes = ['gt', 'A', 'B1', 'B2', 'B3']
    colors = [(0, 0, 0), (0, 255, 0), (0, 0, 255), (255, 0, 255), (255, 255, 255)]
    # images, targets = next(iter(data_loader))
    precision_correct = [0, 0, 0, 0, 0]
    precision_total = [0, 0, 0, 0, 0]
    recall_correct = [0, 0, 0, 0, 0]
    recall_total = [0, 0, 0, 0, 0]
    sensitivity = [0, 0, 0, 0, 0]
    precision = [0, 0, 0, 0, 0]
    recall = [0, 0, 0, 0, 0]
    f2 = [0, 0, 0, 0, 0]
    print_str = ""

    stat_dict_gt = {'A': 0, 'B1': 0, 'B2': 0, 'B3': 0}
    stat_dict_lb = {'A': 0, 'B1': 0, 'B2': 0, 'B3': 0}

    # model generates all results & save the results
    for count, (images, targets) in enumerate(data_loader):

        gt_boxes = targets[0]["boxes"].to(device)
        gt_labels = targets[0]["labels"].to(device)

        if generate_result:
            PIL_img = transforms.ToPILImage()(images[0])
            gt_img = cv2.cvtColor(numpy.asarray(PIL_img), cv2.COLOR_RGB2BGR)
            rs_img = cv2.cvtColor(numpy.asarray(PIL_img), cv2.COLOR_RGB2BGR)
            abs_img = cv2.cvtColor(numpy.asarray(PIL_img), cv2.COLOR_RGB2BGR)

            cv2.imwrite('results/%d.png' % count, gt_img)

        images = list(image.to(device) for image in images)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        out = model(images, targets)

        boxes = out[0]['boxes']
        labels = out[0]['labels']
        scores = out[0]['scores']

        print(count)
        for key, v in stat(gt_labels, 'GT').items():
            stat_dict_gt[key] += v

        for key, v in stat(labels, 'PD').items():
            stat_dict_lb[key] += v

        if generate_result:
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

        precision_correct_add, precision_total_add, recall_correct_add, recall_total_add, unlabeled_index = statistics(gt_boxes, gt_labels, boxes, labels)

        for cls_id in range(1, 5):
            precision_correct[cls_id] += precision_correct_add[cls_id]
            precision_total[cls_id] += precision_total_add[cls_id]
            recall_correct[cls_id] += recall_correct_add[cls_id]
            recall_total[cls_id] += recall_total_add[cls_id]

        if generate_result:
            for idx in unlabeled_index:
                x1, y1, x2, y2 = boxes[idx][0], boxes[idx][1], boxes[idx][2], boxes[idx][3]
                name = classes[labels[idx].item()]
                cv2.rectangle(abs_img, (x1, y1), (x2, y2), colors[labels[idx].item()], thickness=2)
                cv2.putText(abs_img, text=name + " %.3f" % scores[idx], org=(x1, y1 + 15), fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                            fontScale=0.5, thickness=1, lineType=cv2.LINE_AA, color=colors[labels[idx].item()])

            cv2.imwrite('results/%d_gt.png' % count, gt_img)
            if k == 0:
                cv2.imwrite('results/%d_pd.png' % count, rs_img)
            else:
                cv2.imwrite('results/%d_pd%d.png' % (count, k), rs_img)
            # cv2.imwrite('results/%d_abs.png' % count, abs_img)

    # print the whole dataset
    prop_dict_gt = {}
    prop_dict_lb = {}
    for key, v in stat_dict_gt.items():
        prop_dict_gt[key] = v / sum(stat_dict_gt.values())

    for key, v in stat_dict_lb.items():
        prop_dict_lb[key] = v / sum(stat_dict_lb.values())

    print()
    print("[%s] A: %d (%.3f), B1: %d (%.3f), B2: %d (%.3f), B3: %d (%.3f)" %
          ("GT",
           stat_dict_gt['A'], prop_dict_gt['A'],
           stat_dict_gt['B1'], prop_dict_gt['B1'],
           stat_dict_gt['B2'], prop_dict_gt['B2'],
           stat_dict_gt['B3'], prop_dict_gt['B3'])
          )
    print("[%s] A: %d (%.3f), B1: %d (%.3f), B2: %d (%.3f), B3: %d (%.3f)" %
          ("PD",
           stat_dict_lb['A'], prop_dict_lb['A'],
           stat_dict_lb['B1'], prop_dict_lb['B1'],
           stat_dict_lb['B2'], prop_dict_lb['B2'],
           stat_dict_lb['B3'], prop_dict_lb['B3'])
          )

    # calculating metrics
    ep = 1e-5
    TP = [0, 0, 0, 0, 0]
    FP = [0, 0, 0, 0, 0]
    FN = [0, 0, 0, 0, 0]

    matrix = np.zeros([5,7])

    for cls_id in range(1, 5):
        TP[cls_id] = precision_correct[cls_id]
        FP[cls_id] = precision_total[cls_id] - precision_correct[cls_id]
        FN[cls_id] = recall_total[cls_id] - recall_correct[cls_id]
        sensitivity[cls_id] = TP[cls_id] / (TP[cls_id] + FN[cls_id] + ep)
        precision[cls_id] = precision_correct[cls_id] / (precision_total[cls_id] + ep)
        recall[cls_id] = recall_correct[cls_id] / (recall_total[cls_id] + ep)
        f2[cls_id] = (1 + 4) * (precision[cls_id] * recall[cls_id]) / (4 * precision[cls_id] + recall[cls_id] + ep)
        print_str += "Class %s\n" % classes[cls_id]
        print_str += "TP:%d, FP:%d, FN:%d, Sensitivity=%.3f, Precision=%.3f, Recall=%.3f, F2=%.3f\n" % \
                     (TP[cls_id], FP[cls_id], FN[cls_id], sensitivity[cls_id], precision[cls_id], recall[cls_id], f2[cls_id])
        matrix[cls_id - 1] = [TP[cls_id], FP[cls_id], FN[cls_id], sensitivity[cls_id], precision[cls_id], recall[cls_id], f2[cls_id]]


    print_str += "Overall\n"
    print_str += "TP:%d, FP:%d, FN:%d, Sensitivity=%.3f, Precision=%.3f, Recall=%.3f, F2=%.3f\n" % \
                 (sum(TP), sum(FP), sum(FN), sum(TP) / (sum(TP) + sum(FN)),
                  sum(precision_correct) / sum(precision_total), sum(recall_correct) / sum(recall_total),
                  5 * sum(precision_correct) / sum(precision_total) * sum(recall_correct) / sum(recall_total) / (4 * sum(precision_correct) / sum(precision_total) + sum(recall_correct) / sum(recall_total))
                  )
    matrix[4] = [sum(TP), sum(FP), sum(FN), sum(TP) / (sum(TP) + sum(FN)),
           sum(precision_correct) / sum(precision_total), sum(recall_correct) / sum(recall_total),
                 5 * sum(precision_correct) / sum(precision_total) * sum(recall_correct) / sum(recall_total) / (
                 4 * sum(precision_correct) / sum(precision_total) + sum(recall_correct) / sum(recall_total))]

    print(print_str)

    return print_str, matrix


def statistics(gt_boxes, gt_labels, pd_boxes, pd_labels):
    gt_boxes_count = gt_boxes.shape[0]
    pd_boxes_count = pd_boxes.shape[0]
    recall_total = [0, 0, 0, 0, 0]
    recall_correct = [0, 0, 0, 0, 0]

    iou_matrix = find_jaccard_overlap(gt_boxes, pd_boxes)
    max_iou, max_iou_index = torch.max(iou_matrix, 1)
    # print(max_iou, max_iou_index)

    for idx in range(gt_boxes_count):
        gt_class = gt_labels[idx].item()
        recall_total[gt_class] += 1

        if max_iou[idx] > 0:
            max_iou_idx = max_iou_index[idx]
            max_iou_pd_class = pd_labels[max_iou_idx].item()
            if max_iou_pd_class == gt_class:
                recall_correct[gt_class] += 1

    precision_total = [0, 0, 0, 0, 0]
    precision_correct = [0, 0, 0, 0, 0]
    unlabeled_index = []

    iou_matrix = find_jaccard_overlap(pd_boxes, gt_boxes)
    max_iou, max_iou_index = torch.max(iou_matrix, 1)

    for idx in range(pd_boxes_count):
        pd_class = pd_labels[idx].item()
        precision_total[pd_class] += 1

        if max_iou[idx] > 0:
            max_iou_idx = max_iou_index[idx]
            max_iou_gt_class = gt_labels[max_iou_idx].item()
            if max_iou_gt_class == pd_class:
                precision_correct[pd_class] += 1
        else:
            unlabeled_index.append(idx)


    return precision_correct, precision_total, recall_correct, recall_total, unlabeled_index