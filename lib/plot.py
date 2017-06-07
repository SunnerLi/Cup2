from collections import Counter, defaultdict
from config import *
import numpy as np
import cv2

# Mapping object (Auto-generated)
obj_index_2_name = {index: name for name, index in obj_name_2_index.iteritems()}

# Other variable (Auto-generated)
kind = len(obj_name_2_index)
grid_height = None
grid_width = None 

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

def mergeSegmentAndScoringRes(img, result_segment, result_scoring, label_map, edge_graph,
    fast_plot=True):
    """
        Merge the segment and scoring result into the original image

        Arg:    img             - The original image
                result_segment  - The predict result after conducting the UNet
                result_scoring  - The predict result after conducting the scoring net
                label_map       - The result after conducting the connected component process
                edge_graph      - The edge image after doing the laplacian process
        Ret:    The image with merge result
    """
    # Copy image to prevent revised the original one
    res_img = np.copy(img)

    # Generate grid variable and form the vector to the original shape
    grid_height = np.shape(img)[0] / grid_height_num
    grid_width = np.shape(img)[1] / grid_width_num
    scores = np.reshape(result_scoring, [kind, grid_height_num, grid_width_num])
    res = np.copy(img)

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
    # Get the critical point of each segments
    # [   [min_x, min_y, max_x, max_y, mean_x, mean_y], [min_x, min_y, max_x, max_y, mean_x, mean_y], ... ]
    # ----------------------------------------------------------------------------------
    bbox_coordinate_list = np.asarray([[img_width, img_height, 0, 0, 0, 0]] * np.max(label_map))
    for i in range(img_height):
        for j in range(img_width):
            if label_map[i][j] != 0:
                bbox_coordinate_list[label_map[i][j]-1][0] = min(j, bbox_coordinate_list[label_map[i][j]-1][0])
                bbox_coordinate_list[label_map[i][j]-1][1] = min(i, bbox_coordinate_list[label_map[i][j]-1][1])
                bbox_coordinate_list[label_map[i][j]-1][2] = max(j, bbox_coordinate_list[label_map[i][j]-1][2])
                bbox_coordinate_list[label_map[i][j]-1][3] = max(i, bbox_coordinate_list[label_map[i][j]-1][3])
                bbox_coordinate_list[label_map[i][j]-1][4] = int(round(0.5 * bbox_coordinate_list[label_map[i][j]-1][0] 
                    + 0.5 * bbox_coordinate_list[label_map[i][j]-1][2]))
                bbox_coordinate_list[label_map[i][j]-1][5] = int(round(0.5 * bbox_coordinate_list[label_map[i][j]-1][1] 
                    + 0.5 * bbox_coordinate_list[label_map[i][j]-1][3]))

    # ----------------------------------------------------------------------------------
    # Plot the result of segmentation
    # ----------------------------------------------------------------------------------
    if fast_plot:
        res_img = coverEdge(img, edge_graph.astype(np.uint8))
    else:
        for i in range(len(bbox_coordinate_list)):
            for j in range(bbox_coordinate_list[i][1] - 1, bbox_coordinate_list[i][3] + 1):
                for k in range(bbox_coordinate_list[i][0] - 1, bbox_coordinate_list[i][2] + 1):
                    if edge_graph[j][k] != 0:
                        res_img[j][k] = [0, 0, 255]
    
    # ----------------------------------------------------------------------------------
    # Ploting Bounding box and classification
    # (Select first 5th region)
    # ----------------------------------------------------------------------------------
    for i in range(min(len(bbox_coordinate_list), 5)):
        bbox_p1 = (bbox_coordinate_list[i][0], bbox_coordinate_list[i][1])
        bbox_p2 = (bbox_coordinate_list[i][2], bbox_coordinate_list[i][3])
        text_p = (bbox_coordinate_list[i][4], bbox_coordinate_list[i][1])
        cent_p = (bbox_coordinate_list[i][4], bbox_coordinate_list[i][5])

        exam_extra_p1 =  (bbox_coordinate_list[i][0], bbox_coordinate_list[i][3])
        exam_extra_p2 =  (bbox_coordinate_list[i][2], bbox_coordinate_list[i][1])

        if has_response_map[exam_extra_p1[1] / grid_height][exam_extra_p1[0] / grid_width] != 0 or \
            has_response_map[exam_extra_p2[1] / grid_height][exam_extra_p2[0] / grid_width] != 0 or \
            has_response_map[cent_p[1] / grid_height][cent_p[0] / grid_width] != 0 or \
            has_response_map[bbox_p1[1] / grid_height][bbox_p1[0] / grid_width] != 0 or \
            has_response_map[bbox_p2[1] / grid_height][bbox_p2[0] / grid_width] != 0:
            class_index = component_bucket[i]
            cv2.rectangle(res_img, bbox_p1, bbox_p2, 
                obj_index_2_response_color_tuple[class_index], thickness=2)
            cv2.putText(res_img, obj_index_2_name[class_index], text_p, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0))
        else:
            cv2.rectangle(res_img, bbox_p1, bbox_p2, 
                (0, 0, 50), thickness=2)

    return res_img