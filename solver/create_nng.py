
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np

def create_restrictions_from_array(array: np.ndarray):
    rows = len(array)
    colums = len(array[0])

    # create restrictions array
    row_restrictions = []
    col_restrictions = []

    for i in range(rows):
        row_restrictions.append([])
        for j in range(colums):
            if array[i][j] == 1:
                if len(row_restrictions[i]) == 0:
                    row_restrictions[i].append(1)
                elif array[i][j-1] == 1:
                    row_restrictions[i][len(row_restrictions[i]) - 1] += 1
                else:
                    row_restrictions[i].append(1)
    
    for i in range(colums):
        col_restrictions.append([])
        for j in range(rows):
            if array[j][i] == 1:
                if len(col_restrictions[i]) == 0:
                    col_restrictions[i].append(1)
                elif array[i][j - 1] == 1:
                    col_restrictions[i][len(col_restrictions[i]) - 1] += 1
                else:
                    col_restrictions[i].append(1)

    return row_restrictions, col_restrictions

def create_nng_from_image(path: str, rows: int, colums: int, treshhold: int = 120, show_image: bool = False):
    img_colord = Image.open(path)
    img_colord.thumbnail((rows, colums))  # resizes image in-place
    img_gray = img_colord.convert('L')

    if show_image:
        img_colord_plot = plt.imshow(img_colord)
        img_plot = plt.imshow(img_gray, cmap='gray')

    img = np.asarray(img_gray)
    array = np.zeros((rows, colums), dtype=int)

    treshhold = np.average(img)

    for i, line in enumerate(img):
        for j, pixel in enumerate(line):
            if pixel > treshhold:
                array[i][j] = 0
            else:
                array[i][j] = 1

    if show_image:        
        imgplot = plt.imshow(array, cmap='binary')

    # create restrictions array
    row_restrictions, col_restrictions = create_restrictions_from_array(img)

    return row_restrictions, col_restrictions

    