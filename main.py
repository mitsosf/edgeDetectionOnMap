import math
import os
import time
from concurrent.futures.thread import ThreadPoolExecutor
from matplotlib import pyplot as plt

import cv2
import numpy as np

images_path = 'data_jp2/'
rescaled_images_path = 'data_jp2_rescaled/'
images = {}
scale_percent = 0.2
min_road_length = 4000


def parallel_rescale(image_list):
    for image_name in image_list:
        if image_name != '.gitignore':
            start_time = time.time()
            image = cv2.imread(images_path + image_name, cv2.IMREAD_GRAYSCALE)
            width = int(image.shape[1] * scale_percent)
            height = int(image.shape[0] * scale_percent)
            dim = (width, height)

            # Resize image
            image_rescaled = cv2.resize(image, dim, interpolation=cv2.INTER_AREA)
            diff = time.time() - start_time
            print('Read and rescaled image ' + str(image_name) + ' in ' + str(diff) + ' seconds')
            image_id = image_name.split('.')[0]

            images[image_id[:3]].append((image_id[3:], image_rescaled))
            # cv2.imwrite(rescaled_images_path + image_name, image_rescaled)


def rescale_images():
    print('Rescaling images')

    executor = ThreadPoolExecutor(max_workers=4)

    executor.submit(parallel_rescale, os.listdir(images_path)[:25])
    executor.submit(parallel_rescale, os.listdir(images_path)[25:50])
    executor.submit(parallel_rescale, os.listdir(images_path)[50:70])
    executor.submit(parallel_rescale, os.listdir(images_path)[70:])

    executor.shutdown(wait=True)


def stitch_images(images_list):
    vertical_lines = list()
    for vertical_line in images_list.values():
        line_images = ()
        for image in vertical_line:
            line_images = line_images + (image[1],)
        vertical_lines.append(np.vstack(line_images))

    row_images = ()
    for image in vertical_lines:
        row_images = row_images + (image,)

    return np.hstack(row_images)


def initialize_list(image_list):
    for image_name in os.listdir(images_path):
        if image_name != '.gitignore' and image_name[:3] not in images.keys():
            images[image_name[:3]] = list()


def sort_images():
    sorted_keys = sorted(images)

    sorted_keys.sort(key=lambda image: int(image) if image[:1] != '0' else int('1' + image))

    final_dict = {}
    for key in sorted_keys:
        images[key].sort(key=lambda tup: tup[0], reverse=True)
        final_dict[key] = images[key]

    return final_dict


def hough_transform(image):
    rgb_image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    gaussian = cv2.GaussianBlur(image, (11, 11), 0)
    canny = cv2.Canny(gaussian, 100, 150, L2gradient=True)
    # return canny
    lines = cv2.HoughLinesP(canny, 1, np.pi / 180, 60, minLineLength=400, maxLineGap=45)
    print('Detected ' + str(len(lines)) + ' roads.')
    stats = {
        '0.30': 0,
        '0.35': 0,
        '0.40': 0,
        '0.45': 0,
        '0.50': 0,
        '0.55': 0,
        '0.60': 0,
        '0.65': 0,
    }
    for line in lines:
        x1, y1, x2, y2 = line[0]
        angle = math.atan2(y2/y1, x2/x1)
        for stat in stats.keys():
            if angle - float(stat) < 0.5:
                stats[stat] += 1
                break

        cv2.line(rgb_image, (x1, y1), (x2, y2), (0, 0, 255), 2)

    print(stats)

    # return rotate_image(rgb_image, math.degrees(0.3))
    return rgb_image


def rotate_image(image, angle):
    image_center = tuple(np.array(image.shape[1::-1]) / 2)
    rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
    result = cv2.warpAffine(image, rot_mat, image.shape[1::-1], flags=cv2.INTER_LINEAR)
    return result


def main():
    # Enables jp2 support

    os.environ['OPENCV_IO_ENABLE_JASPER'] = 'true'
    initialize_list(images)
    start_time = time.time()
    # rescale_images()
    # image_info = sort_images()
    # result = stitch_images(image_info)
    # cv2.imwrite(rescaled_images_path + 'res.jp2', result)
    result = cv2.imread(rescaled_images_path + 'res_scaled10.jp2', cv2.IMREAD_GRAYSCALE)
    hough_transformed = hough_transform(result)

    processing_end_time = time.time() - start_time
    print('Initialized, resized and stitched images in ' + str(processing_end_time) + ' seconds')
    cv2.namedWindow('Hough map', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Hough map', 900, 1000)
    cv2.imshow('Hough map', hough_transformed)
    print('Rendered map in ' + str(time.time() - processing_end_time) + 'seconds')
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
    exit()
