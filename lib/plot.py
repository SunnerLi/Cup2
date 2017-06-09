from collections import Counter, defaultdict
from config import *
import numpy as np
import time
import cv2

# Mapping object (Auto-generated)
obj_index_2_name = {index: name for name, index in obj_name_2_index.iteritems()}

# Other variable (Auto-generated)
kind = len(obj_name_2_index)
grid_height = None
grid_width = None 

# Dilation kernel
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))

def binaryEdgeMapToRed(img):
    """
        Change the laplacian edge image into red image

        Arg:    img - The laplacian edge image
        Ret:    The red edge image
    """
    red = np.zeros([np.shape(img)[0], np.shape(img)[1], 3])
    red[..., 2] = img
    return red

def coverEdge(origin_img, edge_img):
    """
        Cover the original image with laplacian edge image

        * Notice: Since this function used bitwise operator to split the edge region.
                  As the result, the region will not work if the pixel is white.

        Arg:    origin_img  - The original image
                edge_img    - The edge image after doing the laplacian process
        Ret:    The image with edge covered
    """
    res = np.copy(origin_img)    
    edge_map_inv = cv2.bitwise_not(edge_img)
    img_bg = cv2.bitwise_and(res, res, mask=edge_map_inv)
    img_fg = cv2.bitwise_and(binaryEdgeMapToRed(edge_img), binaryEdgeMapToRed(edge_img), mask=edge_img)
    res = cv2.add(img_bg.astype(np.uint8), img_fg.astype(np.uint8))
    return res

def mergeSegmentAndScoringRes(img, result_segment, result_scoring):
    """
        Merge the segment and scoring result into the original image

        Arg:    img             - The original image
                result_segment  - The predict result after conducting the UNet
                result_scoring  - The predict result after conducting the scoring net
        Ret:    The image with merge result
    """
    # Copy image to prevent revised the original one
    res_img = np.copy(img)

    # Do the connected component
    result_segment = cv2.dilate(result_segment, kernel)
    result_segment = result_segment.astype(np.uint8)
    num_segment, label_map, component_info_list, centroids = cv2.connectedComponentsWithStats(
        result_segment, 4, cv2.CV_32S) 

    # Generate grid variable and form the vector to the original shape
    grid_height = np.shape(img)[0] / grid_height_num
    grid_width = np.shape(img)[1] / grid_width_num
    scores = np.reshape(result_scoring, [kind, grid_height_num, grid_width_num])

    # Plot the ROI binary map
    has_response_map = np.zeros([grid_height_num, grid_width_num])
    for i in range(grid_height_num):
        for j in range(grid_width_num):
            for k in range(kind):
                if scores[k][i][j] != 0.0:
                    has_response_map[i][j] = 1
                    break

    # Create bucket
    component_bucket = [[None]] * np.max(label_map)
    for i in range(len(component_bucket)):
        component_bucket[i] = np.zeros(kind)

    # ----------------------------------------------------------------------------------
    # Collect score
    # ----------------------------------------------------------------------------------
    class_map = np.argmax(scores, axis=0)
    for i in range(grid_height_num):
        for j in range(grid_width_num):
            if has_response_map[i][j] == 1:
                # Determine grid point coordinate tuple
                grid_p1 = (j * grid_width, i * grid_height)
                grid_p2 = (j * grid_width + grid_width, i * grid_height + grid_height)

                # Get the frequent for each component
                mapping_componenet_2_freq = Counter()
                for k in range(grid_p1[1], grid_p2[1]):
                    for m in range(grid_p1[0], grid_p2[0]):
                        if result_segment[k][m] != 0:
                            if not label_map[k][m] in mapping_componenet_2_freq:
                                mapping_componenet_2_freq[label_map[k][m]] = 1
                            else:
                                mapping_componenet_2_freq[label_map[k][m]] += 1

                # Get the most frequent class
                freq_class = mapping_componenet_2_freq.most_common(1)
                if len(freq_class) != 0:
                    freq_class = freq_class[0][0] - 1       # !!??

                    # Add result into bucket
                    _score = scores[class_map[i][j]][i][j]
                    component_bucket[freq_class][class_map[i][j]] += _score
    
    # Voting
    for i in range(len(component_bucket)): 
        component_bucket[i] = np.argmax(component_bucket[i], axis=0)

    # ----------------------------------------------------------------------------------
    # Plot the result of segmentation
    # ----------------------------------------------------------------------------------
    _, edge_graph = cv2.threshold(result_segment, 127, 255, cv2.THRESH_BINARY)
    _, contour, __ = cv2.findContours(edge_graph, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(res_img, contour, -1, (0, 0, 255), 1)

    # ----------------------------------------------------------------------------------
    # Ploting Bounding box and classification
    # (Select first 5th region)
    # ----------------------------------------------------------------------------------
    for i in range(num_segment - 1):
        bbox_p1 = (component_info_list[i + 1][cv2.CC_STAT_LEFT], component_info_list[i + 1][cv2.CC_STAT_TOP])
        bbox_p2 = (bbox_p1[0] + component_info_list[i + 1][cv2.CC_STAT_WIDTH], 
            bbox_p1[1] + component_info_list[i + 1][cv2.CC_STAT_HEIGHT])
        text_p = (int(round(0.5 * bbox_p1[0] + 0.5 * bbox_p2[0])), bbox_p1[1])
        cent_p = (text_p[0], int(round(0.5 * bbox_p1[1] + 0.5 * bbox_p2[1])))

        exam_extra_p1 =  (bbox_p1[0], bbox_p2[1])
        exam_extra_p2 =  (bbox_p2[0], bbox_p1[1])

        if has_response_map[exam_extra_p1[1] / grid_height][exam_extra_p1[0] / grid_width] != 0 or \
            has_response_map[exam_extra_p2[1] / grid_height][exam_extra_p2[0] / grid_width] != 0 or \
            has_response_map[cent_p[1] / grid_height][cent_p[0] / grid_width] != 0 or \
            has_response_map[bbox_p1[1] / grid_height][bbox_p1[0] / grid_width] != 0 or \
            has_response_map[bbox_p2[1] / grid_height][bbox_p2[0] / grid_width] != 0:

            class_index = component_bucket[i]
            cv2.rectangle(res_img, bbox_p1, bbox_p2, obj_index_2_response_color_tuple[class_index], thickness=2)
            cv2.putText(res_img, obj_index_2_name[class_index], text_p, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0))
        else:
            cv2.rectangle(res_img, bbox_p1, bbox_p2, (0, 0, 50), thickness=2)
    
    return res_img