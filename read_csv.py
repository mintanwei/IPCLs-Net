import csv

# return annotation dicts {img_name: {"bbx": [N * 4], "labels": [N]}}
def csv_to_label_and_bbx(csv_file_path, nob3=False, img_names=None):
    class_ids = {"A": 1, "B1": 2, "B2": 3, "B3": 4}
    stats = {"A": 0, "B1": 0, "B2": 0, "B3": 0}
    anno_dicts = {}
    with open(csv_file_path, newline='') as csvfile:
        rows = csv.reader(csvfile)
        rows = list(rows)
        for row in rows[1:]:
            anno_dict = {}
            img_name = row[0]
            if img_names is not None:
                if img_name not in img_names:
                    continue
            raw_boxes = row[1:]
            bounding_boxes = []
            labels = []
            for box in raw_boxes:
                if box == '':
                    continue
                dic_box = eval(box)
                label = dic_box["label"]
                y = dic_box["y"]
                x = dic_box["x"]
                height = dic_box["height"]
                width = dic_box["width"]
                xmin = x
                xmax = x + width
                ymin = y
                ymax = y + height

                if nob3 and label == "B3":
                   label = "B2"

                stats[label] += 1
                label = class_ids[label]
                bounding_boxes.append([xmin, ymin, xmax, ymax])
                labels.append(label)
            anno_dict["bbx"] = bounding_boxes
            anno_dict["labels"] = labels
            anno_dicts[img_name] = anno_dict

    print(stats)
    return anno_dicts


def caibo100(root, save_dir):
    import os
    from PIL import Image
    import numpy as np
    csv_file_path = os.path.join(root, "annotations.csv")
    anno_dicts = csv_to_label_and_bbx(csv_file_path)
    img_names = list(sorted(os.listdir(os.path.join(root, "images_full"))))
    # answer = {}
    classes = ["A", "B1", "B2", "B3"]

    # save_dir = os.path.join(root, "caibo100")
    # save_dir = "./NBI_patch_dataset"
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
        os.mkdir(os.path.join(save_dir, "A"))
        os.mkdir(os.path.join(save_dir, "B1"))
        os.mkdir(os.path.join(save_dir, "B2"))
        os.mkdir(os.path.join(save_dir, "B3"))

    for img_id, img_name in enumerate(img_names):
        img_path = os.path.join(root, "images_full", img_name)
        img = Image.open(img_path).convert("RGB")
        annotations = anno_dicts[img_name]
        boxes = annotations['bbx']
        labels = annotations['labels']

        for box_id, box in enumerate(boxes):
            patch = img.crop(tuple(box))
            label = labels[box_id]
            patch_name = "%d_%d.png" % (img_id, box_id)
            patch.save(os.path.join(save_dir, classes[label - 1], patch_name))
            # answer[patch_name] = label

    # np.save(os.path.join(save_dir, "ans"), answer)


def random_copy(src, dst, num):
    import os
    from random import shuffle
    import shutil
    random_patches = list(os.listdir(src))
    shuffle(random_patches)
    # print(random_patches)
    for patch in random_patches[:num]:
        shutil.copy(os.path.join(src, patch), dst)

def caibo_pr(dir, ans_dict):
    import os
    import numpy as np
    class_ids = {"A": 1, "B1": 2, "B2": 3, "B3": 4, "patch_200": 5}
    confusion_matrix = np.zeros((4, 5))
    for class_name in class_ids:
        class_dir = os.path.join(dir, class_name)
        class_image_names = list(os.listdir(class_dir))
        for class_image_name in class_image_names:
            confusion_matrix[ans_dict[class_image_name] - 1][class_ids[class_name] - 1] += 1
    print(confusion_matrix)

    classes = ["A", "B1", "B2", "B3"]
    precision_array = np.zeros(4)
    recall_array = np.zeros(4)
    for index, class_name in enumerate(classes):
        recall_array[index] = confusion_matrix[index][index] / (np.sum(confusion_matrix, axis=1)[index] + 1e-8)
        precision_array[index] = confusion_matrix[index][index] / (np.sum(confusion_matrix, axis=0)[index] + 1e-8)

    # print(precision_array)
    # print(recall_array)
    print("Precision:")
    for index, class_name in enumerate(classes):
        print(class_name, precision_array[index])
    print("Recall:")
    for index, class_name in enumerate(classes):
        print(class_name, recall_array[index])


def caibo_stats(train_ans_path, test_ans_path, caibo_200_result_path):
    import numpy as np
    train_ans_dict = np.load(train_ans_path, allow_pickle=True).item()
    test_ans_dict = np.load(test_ans_path, allow_pickle=True).item()
    print("训练集:")
    caibo_pr("./ZMCB200+200/Part_I", train_ans_dict)
    print("测试集:")
    caibo_pr("./ZMCB200+200/Part_II", test_ans_dict)

if __name__ == '__main__':
    # caibo100("./NBI_dataset/train")
    # random_copy("E:/NBI_Detection/NBI_dataset/test/caibo_1395", "I:/食管血管检测/ZMCB200+200/Part_II/patch_200", 200)
    # random_copy("E:/NBI_Detection/NBI_dataset/train/caibo_8815", "I:/食管血管检测/ZMCB200+200/Part_I/patch_200", 200)
    # print(csv_to_label_and_bbx("./NBI_dataset/train/annotations.csv"))
    # caibo_stats("I:/食管血管检测/caibo_8815/ans.npy", "I:/食管血管检测/caibo_1395/ans.npy", "./ZMCB200+200/")
    caibo100("./NBI_dataset/train", "./NBI_patch_dataset/train_full")
