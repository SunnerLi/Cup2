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
    # Plot the result of segmentation
    # ----------------------------------------------------------------------------------
    if fast_plot:
        res_img = coverEdge(img, edge_graph.astype(np.uint8))

    # ----------------------------------------------------------------------------------
    # Ploting classification
    # ----------------------------------------------------------------------------------
    for component_index in range(len(component_bucket)):
        min_x = -1
        max_x = -1
        min_y = -1
        max_y = -1

        # Find the position coordinate
        has_found = False
        for i in range(grid_height_num):
            has_change = False
            for j in range(grid_width_num):
                if has_response_map[i][j]:
                    if fast_plot:
                        init_height = i * grid_height
                        last_height = i * grid_height + grid_height
                        init_width = j * grid_width
                        last_width = j * grid_width + grid_width
                    else:
                        init_height = i * grid_height - grid_height
                        last_height = i * grid_height + 2 * grid_height
                        init_width = j * grid_width - grid_width
                        last_width = j * grid_width + 2 * grid_width

                    # Check the segment position for specific area
                    for k in range(init_height, last_height):
                        for m in range(init_width, last_width):
                            if not fast_plot:
                                if edge_graph[k][m] != 0:
                                    res_img[k][m] = [0, 0, 255]

                            if label_map[k][m] != 0:
                                if component_bucket[label_map[k][m]-1] == component_index:
                                    has_found = True
                                    has_change = True
                                    if min_x == -1:
                                        min_x = m
                                    else:
                                        min_x = min(min_x, m)
                                    max_x = max(max_x, m)
                                    if min_y == -1:
                                        min_y = k
                                    else:
                                        min_y = min(min_y, k)
                                    max_y = max(max_y, k)
            if has_found == True and has_change == False:
                break        

        # Form the point tuple
        box_p1 = (min_x, min_y)
        box_p2 = (max_x, max_y)
        text_p = (int(round((min_x + max_x)/2)), min_y)

        # Draw
        class_index = component_bucket[component_index]
        cv2.rectangle(res_img, box_p1, box_p2, 
                    obj_index_2_response_color_tuple[class_index], thickness=2)

        cv2.putText(res_img, obj_index_2_name[class_index], text_p, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0))
    return res_img

