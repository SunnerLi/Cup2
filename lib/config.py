img_height = 270
img_width = 480
img_channel = 3

# Mapping object (Defined)
obj_name_2_index = {
    'butterfly': 0,
    'gate': 1,
    'flower': 2,
    'lenwen': 3,
    'library': 4,
    'law': 5
}
obj_index_2_color_tuple = {
    0: (100, 0, 0),
    1: (200, 0, 0),
    2: (150, 0, 0),
    3: (250, 0, 0),
    4: (250, 0, 100),
    5: (255, 0, 100)
}
obj_index_2_response_color_tuple = {
    0: (0, 40, 0),
    1: (0, 80, 0),
    2: (0, 120, 0),
    3: (0, 160, 0),
    4: (0, 200, 0),
    5: (0, 240, 0)
}

# Grid number (Defined)
grid_height_num = 9
grid_width_num = 16

# ?
model_path = '../model/'
unet_model_name = 'unet.h5'
scorenet_model_name = 'scorenet.h5' 

# ?
scorenet_img_path = '../img/scorenet/'
scorenet_dat_path = '../dat/'

# ?
scorenet_fc_num = len(obj_name_2_index) * grid_height_num * grid_width_num

general_epoch = 400
