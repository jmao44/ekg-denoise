import os
import cv2
import numpy as np

from matplotlib import pyplot as plt
from scipy import ndimage
from PIL import Image

def main():
    image_path = 'images/test07.jpg'
    save_path = 'output_images/output' + image_path[-6:-4] + '.png'
    image = cv2.imread(image_path, flags=cv2.IMREAD_GRAYSCALE)
    height, width = image.shape
    print('INPUT image is of size: {} x {}.'.format(height, width))
    image = image[50 : height - 120, 50 : width - 20]
    ret, image = cv2.threshold(image, 50, 255, cv2.THRESH_BINARY)
    structure = np.array([[1, 1, 1],
                          [1, 1, 1],
                          [1, 1, 1]], np.uint8)
    labeled_image, cc_num = ndimage.label(image, structure=structure)
    cc = ndimage.find_objects(labeled_image)
    print('There are {} connected components.'.format(cc_num))
    cc_areas = ndimage.sum(image, labeled_image, range(cc_num + 1))
    area_mask = cc_areas < 40000
    labeled_image[area_mask[labeled_image]] = 0
    plt.imsave(save_path, labeled_image, cmap='binary')
    print('Image is saved to: {}'.format(save_path))

if __name__ == '__main__':
    main()