import os
import time
import cv2
import matplotlib.pyplot as plt

images_path = 'data_jp2/'
rescaled_images_path = 'data_jp2_rescaled/'


def rescale_images():
    print('Rescaling images')
    images = list()
    for image_name in os.listdir(images_path):
        if image_name != '.gitignore':
            start_time = time.time()

            image = cv2.imread(images_path + image_name, 0)
            scale_percent = 60
            width = int(image.shape[1] * scale_percent / 100)
            height = int(image.shape[0] * scale_percent / 100)
            dim = (width, height)

            # Resize image
            image_rescaled = cv2.resize(image, dim, interpolation=cv2.INTER_AREA)
            images.append(image_rescaled)
            diff = time.time() - start_time
            print('Rescaled image ' + str(image_name) + ' in ' + str(diff) + ' seconds')
            cv2.imwrite(rescaled_images_path + image_name, image_rescaled)

    return 1

    fig, axes = plt.subplots(nrows=1, ncols=2)

    # ax = axes.ravel()
    #
    # ax[0].imshow(image, cmap='gray')
    # ax[0].set_title("Original image")
    #
    # ax[1].imshow(image_rescaled, cmap='gray')
    # ax[1].set_title("Rescaled image (aliasing)")
    # plt.show()


def main():
    # Enables jp2 support
    os.environ['OPENCV_IO_ENABLE_JASPER'] = 'true'
    rescale_images()


if __name__ == '__main__':
    main()
    exit()
