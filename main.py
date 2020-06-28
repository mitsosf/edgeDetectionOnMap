import math
import os
import time
import matplotlib.pylab as plt
from concurrent.futures.thread import ThreadPoolExecutor

import cv2
import numpy as np

images_path = 'data_jp2/'
rescaled_images_path = 'data_jp2_rescaled/'
images = {}
scale_percentage = 0.1
min_road_length = 4000


def parallel_rescale(image_list):
    for image_name in image_list:
        if image_name != '.gitignore':
            start_time = time.time()
            image = cv2.imread(images_path + image_name, cv2.IMREAD_GRAYSCALE)
            width = int(image.shape[1] * scale_percentage)
            height = int(image.shape[0] * scale_percentage)
            dim = (width, height)

            # Resize image
            image_rescaled = cv2.resize(image, dim, interpolation=cv2.INTER_AREA)
            if image_name == '990222.jp2':
                cv2.imwrite(rescaled_images_path + 'natural_history_museum_10.jp2', image_rescaled)
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


def auto_canny(image, sigma=0.33):
    # compute the median of the single channel pixel intensities
    v = np.median(image)

    # apply automatic Canny edge detection using the computed median
    lower = int(max(0, (1.0 - sigma) * v))
    upper = int(min(255, (1.0 + sigma) * v))
    edged = cv2.Canny(image, lower, upper, L2gradient=True)

    # return the edged image
    return edged


def hough_transform(image, perpendicular_lines=False):
    rgb_image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

    kernel = np.ones((5, 5), np.uint8)
    erosion = cv2.erode(image, kernel, iterations=2)
    dilation = cv2.dilate(erosion, kernel, iterations=1)

    gaussian = cv2.GaussianBlur(dilation, (11, 11), 0)
    canny = cv2.Canny(dilation, 100, 200, L2gradient=True)
    # cv2.namedWindow('Canny', cv2.WINDOW_NORMAL)
    # cv2.resizeWindow('Canny', 900, 1000)
    # cv2.imshow('Canny', canny)
    lines = cv2.HoughLinesP(canny, 1, np.pi / 180, 100, minLineLength=min_road_length * scale_percentage, maxLineGap=40)
    # for line in lines:

    line_counter = 0
    for line in lines:
        x1, y1, x2, y2 = line[0]

        if perpendicular_lines:
            deltaY = y2 - y1
            deltaX = x2 - x1
            angle_in_degrees = math.atan2(deltaY, deltaX) * 180 / np.pi
            if -3 < angle_in_degrees < 3 or 87 < angle_in_degrees < 93 \
                    or 177 < angle_in_degrees < 183 or 267 < angle_in_degrees < 273:
                line_counter += 1
                cv2.line(rgb_image, (x1, y1), (x2, y2), (0, 0, 255), 2)
        else:
            line_counter += 1
            cv2.line(rgb_image, (x1, y1), (x2, y2), (0, 0, 255), 2)

    first_line = lines[0]
    x1, y1, x2, y2 = first_line[0]
    deltaY = y2 - y1
    deltaX = x2 - x1
    angleInDegrees = math.atan2(deltaY, deltaX) * 180 / np.pi

    return rgb_image, angleInDegrees, line_counter


def rotate_image(image, angle):
    image_center = tuple(np.array(image.shape[1::-1]) / 2)
    rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
    result = cv2.warpAffine(image, rot_mat, image.shape[1::-1], flags=cv2.INTER_LINEAR)
    return result


def calculate_area_in_pixels(image):
    contours, _ = cv2.findContours(image, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    structure = contours[1]
    return round(cv2.contourArea(structure) * scale_percentage * 100)


def draw_rectangle(image, coordinates):
    rgb_image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

    return cv2.rectangle(image, coordinates[0], coordinates[1], (0, 0, 255), -1)


def calculate_relative_area(source_actual, source_pixels, target_pixels):
    return target_pixels * source_actual / source_pixels


def main():
    # Enables jp2 support

    os.environ['OPENCV_IO_ENABLE_JASPER'] = 'true'
    initialize_list(images)
    start_time = time.time()
    # rescale_images()
    # image_info = sort_images()
    # result = stitch_images(image_info)
    # cv2.imwrite(rescaled_images_path + 'res.jp2', result)

    processing_end_time = time.time() - start_time
    print('Initialized, resized and stitched images in ' + str(processing_end_time) + ' seconds')

    result = cv2.imread(rescaled_images_path + 'res_scaled10.jp2', cv2.IMREAD_GRAYSCALE)

    hough_transformed, angle, _ = hough_transform(result)
    rotated_image = rotate_image(result, angle)
    hough_rotated, _, detected_lines = hough_transform(rotated_image, perpendicular_lines=True)
    print('Detected ' + str(detected_lines) + ' roads after performing the Hough Transform on the rotated map')

    central_park_augmented = draw_rectangle(rotated_image, ((1230, 2060), (3920, 2600)))
    central_park_area_in_pixels = calculate_area_in_pixels(central_park_augmented)
    print('The area of Central Park is around ' + str(central_park_area_in_pixels) + ' pixels in the original image')

    museum = cv2.imread(rescaled_images_path + 'natural_history_museum_10.jp2', cv2.IMREAD_GRAYSCALE)
    museum = rotate_image(museum, angle)
    museum_augmented = draw_rectangle(museum, ((200, 180), (400, 340)))
    museum_area_in_pixels = calculate_area_in_pixels(museum_augmented)
    print('The area of the National History Museum is around ' + str(museum_area_in_pixels) + 'pixels'
                                                                                              ' in the original image')

    museum_actual_area = 82386.68
    central_park_area = calculate_relative_area(museum_actual_area, museum_area_in_pixels, central_park_area_in_pixels)
    print('If the museuim\'s actual area is ' + str(museum_actual_area) + ' then the Central Park has an area of:')
    print(str(central_park_area) + ' m^2')

    cv2.namedWindow('Hough map', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Hough map', 900, 1000)
    cv2.imshow('Hough map', central_park_augmented)
    print('Rendered map in ' + str(time.time() - processing_end_time) + 'seconds')
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
    exit()
