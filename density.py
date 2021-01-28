from PIL import Image
from read_csv import csv_to_label_and_bbx
import os
import numpy as np
from PIL import Image
import scipy.io as io
from itertools import islice
from tqdm import tqdm
from matplotlib import pyplot as plt
from scipy.ndimage.filters import gaussian_filter
import scipy
import pickle
from model import rcnn_distance_model

def generate_gaussian_kernels(out_kernels_path='gaussian_kernels.pkl', round_decimals=3, sigma_threshold=4, sigma_min=0,
                              sigma_max=100, num_sigmas=801):
    """
    Computing gaussian filter kernel for sigmas in linspace(sigma_min, sigma_max, num_sigmas) and saving
    them to dict.
    """
    kernels_dict = dict()
    sigma_space = np.linspace(sigma_min, sigma_max, num_sigmas)
    for sigma in tqdm(sigma_space):
        sigma = np.round(sigma, decimals=round_decimals)
        kernel_size = np.ceil(sigma * sigma_threshold).astype(np.int)

        img_shape = (kernel_size * 2 + 1, kernel_size * 2 + 1)
        img_center = (img_shape[0] // 2, img_shape[1] // 2)

        arr = np.zeros(img_shape)
        arr[img_center] = 1

        arr = scipy.ndimage.filters.gaussian_filter(arr, sigma, mode='constant')
        kernel = arr / arr.sum()
        kernels_dict[sigma] = kernel

    print(f'Computed {len(sigma_space)} gaussian kernels. Saving them to {out_kernels_path}')
    print(kernels_dict)

    with open(out_kernels_path, 'wb') as f:
        pickle.dump(kernels_dict, f)

precomputed_kernels_path = 'gaussian_kernels.pkl'

# uncomment to generate and save dict with kernel sizes
# generate_gaussian_kernels(precomputed_kernels_path, round_decimals=3, sigma_threshold=4, sigma_min=0, sigma_max=100, num_sigmas=801)

with open(precomputed_kernels_path, 'rb') as f:
    kernels_dict = pickle.load(f)
    # kernels_dict = SortedDict(kernels_dict)

def gaussian_filter_density(non_zero_points, map_h, map_w, kernels_dict=None, min_sigma=2, const_sigma=15):
    gt_count = len(non_zero_points)
    density_map = np.zeros((map_h, map_w), dtype=np.float32)

    for i in range(gt_count):
        point_y, point_x, size = non_zero_points[i]
        sigma = min(size * 0.25, 100)
        kernel = kernels_dict[sigma]
        full_kernel_size = kernel.shape[0]
        kernel_size = full_kernel_size // 2

        min_img_x = max(0, point_x-kernel_size)
        min_img_y = max(0, point_y-kernel_size)
        max_img_x = min(point_x+kernel_size+1, map_h - 1)
        max_img_y = min(point_y+kernel_size+1, map_w - 1)

        kernel_x_min = kernel_size - point_x if point_x <= kernel_size else 0
        kernel_y_min = kernel_size - point_y if point_y <= kernel_size else 0
        kernel_x_max = kernel_x_min + max_img_x - min_img_x
        kernel_y_max = kernel_y_min + max_img_y - min_img_y

        density_map[min_img_x:max_img_x, min_img_y:max_img_y] += kernel[kernel_x_min:kernel_x_max, kernel_y_min:kernel_y_max]
    return density_map


if __name__ == '__main__':
    root = "./NBI_new_dataset/test/"
    label_and_bbx = csv_to_label_and_bbx("./NBI_new_dataset/annotations_all.csv")
    all_images = list(sorted(os.listdir(root)))
    save_path = "./density_maps_predict"
    if not os.path.exists(save_path):
        os.mkdir(save_path)


    for image_name in all_images:
        image_path = os.path.join(root, image_name)
        boxes = label_and_bbx[image_name]["bbx"]
        labels = label_and_bbx[image_name]["labels"]
        img = Image.open(image_path)
        img_array = np.array(img, dtype=np.float32)
        img_cls = np.array(img, dtype=np.float32)
        # img.show()

        # print(box) # x1 y1 x2 y2

        cls_points = [[],[],[],[]] # A, B1, B2, B3

        for i, box in enumerate(boxes):
            x1, y1, x2, y2 = tuple(box)
            center_x = (x1 + x2) // 2
            center_y = (y1 + y2) // 2
            w = x2 - x1
            h = y2 - y1
            size = max(w, h)
            cls_points[labels[i]-1].append((center_x, center_y, size))


        w = img.size[0]
        h = img.size[1]
        density_map = np.zeros((5, h, w), dtype=np.float32)
        density_map[0] = 1e-10
        dm = []

        colors = [(0, 0, 0), (0, 1, 0), (1, 0, 0), (1, 0 ,1), (1, 1, 1)] # neg, A, b1-3
        for i, points in enumerate(cls_points):
            if len(points) > 0:
                density_map[i + 1] = gaussian_filter_density(points, map_h=h, map_w=w, kernels_dict=kernels_dict)
                # multiplier = 255 / np.max(density_map[i + 1])
                # color = density_map[i + 1] * multiplier
                log_den = np.log(density_map[i + 1] + 1e-12)
                # print(log_den)
                max_log = np.max(log_den) # -> 255
                min_log = np.max(log_den) - np.log(1000) # -> 0
                color = (log_den - min_log) / (max_log - min_log) * 90
                print(color)
                color = np.clip(color, 0, 255)
                for c in range(3):
                    img_array[:,:,c] += color * colors[i + 1][c]

                # zero_padding = np.zeros((h, w, 2), dtype=np.float32)
                # arr = np.concatenate([color.reshape(h, w, 1), zero_padding], axis=2)
                # print(arr)
                # map = Image.fromarray(np.uint8(arr))
                # map.show()
                # dm.append(map)

        np.save(os.path.join(save_path, image_name[:-4]), density_map[1:])

        img_array = np.clip(img_array, 0, 255)
        final = Image.fromarray(np.uint8(img_array))
        final.save(os.path.join(save_path, image_name))
        # final.show()

        # cls = np.argmax(density_map, axis=0)
        # print(cls)
        # for x in range(h):
        #     for y in range(w):
        #         for c in range(3):
        #             img_cls[x, y, c] += colors[cls[x, y]][c] * 50
        # img_cls = np.clip(img_cls, 0, 255)
        # cls_res = Image.fromarray(np.uint8(img_cls))
        # cls_res.show()
