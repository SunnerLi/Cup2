from collections import OrderedDict
from config import *
import numpy as np
import cv2

# Mapping object (Auto-generated)
obj_index_2_name = {index: name for name, index in obj_name_2_index.iteritems()}

# Other variable (Auto-generated)
kind = len(obj_name_2_index)
grid_height = None
grid_width = None 

def isFullIntersec(item1_p1, item1_p2, item2_p1, item2_p2):
    """
        Judge if the full intersection is occuring

        Arg:    item1_p1    - The point 1 of the item1
                item1_p2    - The point 2 of the item1
                item2_p1    - The point 1 of the item2
                item2_p2    - The point 2 of the item2
    """
    if item1_p1[0] >= item2_p1[0] and \
        item1_p1[1] >= item2_p1[1] and \
        item1_p2[0] <= item2_p2[0] and \
        item1_p2[0] <= item2_p2[0]:
        return True
    return False 

def getObjInfoList(dat_name):
    with open(dat_name, 'r') as f:
        number_of_obj = int(f.readline())
        info = f.readlines()
        res = []
        for i in range(number_of_obj):
            one_obj_dict = OrderedDict()
            one_obj_dict['object'] = obj_name_2_index[info[i*5][:-1]]
            one_obj_dict['p1'] = (int(info[i*5+1]), int(info[i*5+2]))
            one_obj_dict['p2'] = (int(info[i*5+3]), int(info[i*5+4]))
            res.append(one_obj_dict)
    info = res
    return info

def drawObjBBox(img, dat_name):
    info = getObjInfoList(dat_name)
    res = np.copy(img)
    for object_dict in info:
        cv2.rectangle(res, object_dict['p1'], object_dict['p2'], 
            obj_index_2_color_tuple[object_dict['object']], thickness=5)
    return res

def drawGridBox(img, dat_name):
    global grid_height
    global grid_width

    # Read data and Generate grid variable
    info = getObjInfoList(dat_name)
    res = np.copy(img)
    grid_height = np.shape(img)[0] / grid_height_num
    grid_width = np.shape(img)[1] / grid_width_num 

    # Get intersection score for each class
    scores = np.zeros([kind, grid_height_num, grid_width_num])
    for j in range(grid_height_num):
        for k in range(grid_width_num):
            grid_p1 = (k * grid_width, j * grid_height)
            grid_p2 = (k * grid_width + grid_width, j * grid_height + grid_height)

            # Check if the grid have intersection
            for m in range(len(info)):
                int_p1 = (
                    max(grid_p1[0], info[m]['p1'][0]), max(grid_p1[1], info[m]['p1'][1])
                )
                int_p2 = (
                    min(grid_p2[0], info[m]['p2'][0]), min(grid_p2[1], info[m]['p2'][1])
                )
                if int_p1[0] < int_p2[0] and int_p1[1] < int_p2[1]:
                    roi_area = (int_p2[0] - int_p1[0]) * (int_p2[1] - int_p1[1])
                    grid_area = (grid_p2[0] - grid_p1[0]) * (grid_p2[1] - grid_p1[1])
                    rate = float(roi_area) / grid_area
                    scores[info[m]['object']][j][k] += rate
                    cv2.rectangle(res, grid_p1, grid_p2, 
                        obj_index_2_response_color_tuple[info[m]['object']], thickness=2)
    return res


def encodeByFile(img, dat_name):
    """
        Encode the image as the vector by the description file
    """
    global grid_height
    global grid_width

    # Read data and Generate grid variable
    info = getObjInfoList(dat_name)
    grid_height = np.shape(img)[0] / grid_height_num
    grid_width = np.shape(img)[1] / grid_width_num 

    # Get intersection score for each class
    scores = np.zeros([kind, grid_height_num, grid_width_num])
    for j in range(grid_height_num):
        for k in range(grid_width_num):
            grid_p1 = (k * grid_width, j * grid_height)
            grid_p2 = (k * grid_width + grid_width, j * grid_height + grid_height)

            # Check if the grid have intersection
            for m in range(len(info)):
                int_p1 = (
                    max(grid_p1[0], info[m]['p1'][0]), max(grid_p1[1], info[m]['p1'][1])
                )
                int_p2 = (
                    min(grid_p2[0], info[m]['p2'][0]), min(grid_p2[1], info[m]['p2'][1])
                )
                if int_p1[0] < int_p2[0] and int_p1[1] < int_p2[1]:
                    roi_area = (int_p2[0] - int_p1[0]) * (int_p2[1] - int_p1[1])
                    grid_area = (grid_p2[0] - grid_p1[0]) * (grid_p2[1] - grid_p1[1])
                    rate = float(roi_area) / grid_area
                    # print "item: ", obj_index_2_name[info[m]['object']], "inter rate: ", rate
                    scores[info[m]['object']][j][k] = min(scores[info[m]['object']][j][k] + rate, 1)

    # Plot the ROI binary map
    has_response_map = np.zeros([grid_height_num, grid_width_num])
    for i in range(grid_height_num):
        for j in range(grid_width_num):
            for k in range(kind):
                if scores[k][i][j] != 0.0:
                    has_response_map[i][j] = 1
                    break

    # Punish the big object
    for i in range(grid_height_num):
        for j in range(grid_width_num):
            if has_response_map[i][j] == 1:
                grid_p1 = (j * grid_width, i * grid_height)
                grid_p2 = (j * grid_width + grid_width, i * grid_height + grid_height)

                for k in range(len(info)):
                    item1_int_p1 = (
                        max(grid_p1[0], info[k]['p1'][0]), 
                        max(grid_p1[1], info[k]['p1'][1])
                    )
                    item1_int_p2 = (
                        min(grid_p2[0], info[k]['p2'][0]), 
                        min(grid_p2[1], info[k]['p2'][1])
                    )
                    if item1_int_p1[0] < item1_int_p2[0] and item1_int_p1[1] < item1_int_p2[1]:
                        for m in range(k+1, len(info)):
                            if info[k]['object'] != info[m]['object']:
                                item2_int_p1 = (
                                    max(grid_p1[0], info[m]['p1'][0]), 
                                    max(grid_p1[1], info[m]['p1'][1])
                                )
                                item2_int_p2 = (
                                    min(grid_p2[0], info[m]['p2'][0]), 
                                    min(grid_p2[1], info[m]['p2'][1])
                                )

                                # Check if there's full collabrated
                                if isFullIntersec(
                                    item1_int_p1, item1_int_p2,
                                    item2_int_p1, item2_int_p2):
                                    scores[info[m]['object']][i][j] = 0.0
    return np.reshape(scores, [-1])

def decodeByVector(img, vector):
    global grid_height
    global grid_width

    # Generate grid variable and form the vector to the original shape
    grid_height = np.shape(img)[0] / grid_height_num
    grid_width = np.shape(img)[1] / grid_width_num
    scores = np.reshape(vector, [kind, grid_height_num, grid_width_num])
    res = np.copy(img)

    # Plot the ROI binary map
    has_response_map = np.zeros([grid_height_num, grid_width_num])
    for i in range(grid_height_num):
        for j in range(grid_width_num):
            for k in range(kind):
                if scores[k][i][j] != 0.0:
                    has_response_map[i][j] = 1
                    break

    # Plot the response grid
    for i in range(grid_height_num):
        for j in range(grid_width_num):
            if has_response_map[i][j] == 1:
                # Determine the critical point
                grid_p1 = (j * grid_width, i * grid_height)
                grid_p2 = (j * grid_width + grid_width, i * grid_height + grid_height)
                line2_p = (j * grid_width, i * grid_height + 20)

                # Draw the information
                _class = np.argmax(scores, axis=0)[i][j]
                cv2.rectangle(res, grid_p1, grid_p2, 
                    obj_index_2_response_color_tuple[_class], thickness=5)
                cv2.putText(res, obj_index_2_name[_class], grid_p1, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0))
                cv2.putText(res, str(round(scores[_class][i][j], 2)), line2_p, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0))
    return res

if __name__ == '__main__':
    # Read image and encode
    img = cv2.imread('frame000.bmp')
    vector = encodeByFile(img, 'frame000.dat')

    # Decode and draw on the image
    res = decodeByVector(
        drawGridBox(img, 'frame000.dat'), 
        vector
    )

    # Show
    cv2.imshow('show', res)
    cv2.waitKey(0)