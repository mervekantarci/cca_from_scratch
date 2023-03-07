from matplotlib import pyplot as plt
import numpy as np
import cv2
import argparse
import os


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-file', required=True, action="store", type=str,
                        help='path to input file')
    parser.add_argument('--save', required=False, action="store_true", default=False,
                        help='processed image will be saved')
    parser.add_argument('--no_display', required=False, action="store_true", default=False,
                        help='processed image will not be displayed')

    args_ = parser.parse_args()
    return args_


def preprocess(img, threshold):
    # resizing did not improve the results
    # img = cv2.resize(img, (0, 0), fx=0.8, fy=0.8)
    # convert to grayscale
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # convert to binary form
    img[img < threshold + 1] = 1
    img[img > threshold] = 0

    return img


def clean(img, kernel_size, show=True, save=False, filepath=None):
    # below can be used for erosion experiments
    # img_eroded = cv2.erode(img, np.ones((2, 2)))
    # dilated results are preferred, kernel size and number of iterations decided experimentally
    img_dilated = cv2.dilate(img, kernel=np.ones(kernel_size, np.uint8), iterations=1)

    if save:
        filename, file_extension = os.path.splitext(filepath)
        save_path = filename + "_processed" + file_extension
        cv2.imwrite(save_path, img_dilated * 255)

    if show:
        plt.imshow(img_dilated, cmap="gray")
        plt.show()

    return img_dilated


def label_connected_components(img):
    # zero pad image with 1 row & column from each side to ease array access
    padded_img = np.zeros((img.shape[0] + 2, img.shape[1] + 2))
    padded_img[1:-1, 1:-1] = img
    # this will store labels, 0 for background
    labeled_img = np.zeros(padded_img.shape)
    label = 0
    for i in range(1, img.shape[0] + 1):
        for j in range(1, img.shape[1] + 1):
            if padded_img[i, j] == 0 or labeled_img[i, j] != 0:
                continue
            label = label + 1  # set a new label
            labeled_img[i, j] = label
            # first pixel does not need to go through previous pixels, only the later ones
            connections = [(i+1,j-1),(i+1,j),(i+1,j+1),(i,j+1)]
            i_comp = 0
            while i_comp < len(connections):   # while neighbor check is not completed
                # traverse all component indices
                c_i, c_j = connections[i_comp][0], connections[i_comp][1]
                if labeled_img[c_i, c_j] == 0 and padded_img[c_i, c_j] != 0:
                    labeled_img[c_i, c_j] = label
                    # add all neighbors, foreground pixel check will be done in later iterations
                    connections.extend([(c_i+x, c_j+y) for x in range(-1,2,1) for y in range(-1,2,1) if x != 0 or y != 0])
                i_comp = i_comp + 1

    # -1 because of zero (background) labels
    num_components = np.unique(labeled_img).shape[0] - 1
    return num_components


if __name__ == '__main__':
    args = get_args()
    print(args)

    im = cv2.imread(args.file)
    im = preprocess(im, threshold=128)
    # set dilation & erosion kernel
    im = clean(im, kernel_size=(2, 2), show=not args.no_display, save=args.save, filepath=args.file)
    number_components = label_connected_components(im)
    print("Number of components: " + str(number_components))



