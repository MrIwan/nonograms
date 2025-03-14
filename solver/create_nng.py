
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np

def create_nng_from_image(path: str, size: int, treshhold: int = 120, show_image: bool = False):
    img_colord = Image.open(path)
    img_colord.thumbnail((size, size))  # resizes image in-place
    img_gray = img_colord.convert('L')

    if show_image:
        img_colord_plot = plt.imshow(img_colord)
        img_plot = plt.imshow(img_gray, cmap='gray')

    img = np.asarray(img_gray)
    array = np.zeros((size, size), dtype=int)

    for i, line in enumerate(img):
        for j, pixel in enumerate(line):
            if pixel > 120:
                array[i][j] = 0
            else:
                array[i][j] = 1

    if show_image:        
        imgplot = plt.imshow(array, cmap='binary')

    # create restrictions array
    row_restrictions = []
    col_restrictions = []

    for i in range(size):
        row_restrictions.append([])
        col_restrictions.append([])
        for j in range(size):
            if array[i][j] == 1:
                if len(row_restrictions[i]) == 0:
                    row_restrictions[i].append(1)
                elif array[i][j-1] == 1:
                    row_restrictions[i][len(row_restrictions[i]) - 1] += 1
                else:
                    row_restrictions[i].append(1)

            if array[j][i] == 1:
                if len(col_restrictions[i]) == 0:
                    col_restrictions[i].append(1)
                elif array[j-1][i] == 1:
                    col_restrictions[i][len(col_restrictions[i]) - 1] += 1
                else:
                    col_restrictions[i].append(1)

    return row_restrictions, col_restrictions

    