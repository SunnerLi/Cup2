img_height = 270
img_width = 480
img_channel = 3

# Mapping object (Defined)
obj_name_2_index = {
    'butterfly': 0,
    'gate': 1
}
obj_index_2_color_tuple = {
    0: (100, 0, 0),
    1: (200, 0, 0)
}
obj_index_2_response_color_tuple = {
    0: (0, 100, 0),
    1: (0, 200, 0)
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

general_epoch = 100