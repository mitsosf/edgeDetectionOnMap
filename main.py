import os
import time
from concurrent.futures.thread import ThreadPoolExecutor

import cv2
import numpy as np

images_path = 'data_jp2/'
rescaled_images_path = 'data_jp2_rescaled/'
images = {}


def parallel_rescale(image_list):
    for image_name in image_list:
        if image_name != '.gitignore':
            start_time = time.time()
            image = cv2.imread(images_path + image_name, cv2.IMREAD_GRAYSCALE)
            scale_percent = 2
            width = int(image.shape[1] * scale_percent / 100)
            height = int(image.shape[0] * scale_percent / 100)
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


def main():
    # Enables jp2 support

    os.environ['OPENCV_IO_ENABLE_JASPER'] = 'true'
    initialize_list(images)
    start_time = time.time()
    rescale_images()
    image_info = sort_images()
    result = stitch_images(image_info)
    print('Initialized, resized and stitched images in ' + str((time.time() - start_time)) + ' seconds')
    cv2.imwrite(rescaled_images_path + 'res.jp2', result)
    cv2.imshow('Reconstructed map', result)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
    exit()
