import _init_paths
from scorenet import ScoreNet
from unet import UNet
from config import *
from plot import *
import tensorflow as tf
import numpy as np
import threading
import time
import coder
import cv2

# Tensorflow graph object
graph = tf.get_default_graph()

# Camera capture related variable
cap = None
is_cap_open = False

# model object
segment_model = None
scoring_model = None

# break variable and timer to calculate fps
frame_index = 0
timer = 0.0
fps_index = 0

# ---------------------------------------------------------------
# Image objects and predict result
# ---------------------------------------------------------------
# 1st
frame_fetch = None

# 2nd
frame_process = None
result_segment = None
result_scoring = None

# 3rd
show_frame = None
segment_input = None
scoring_input = None
result_segment_lap = None
result_final = None

class fetchImgThread(threading.Thread):
    """
        The thread to fetch the image in each duration
    """
    def __init__(self):
        threading.Thread.__init__(self)

    def start(self):
        threading.Thread.__init__(self)
        threading.Thread.start(self)
    
    def join(self, _sec):
        threading.Thread.join(self, _sec)

    def run(self):
        global cap
        global is_cap_open
        global frame_fetch
        _, frame = cap.read()
        is_cap_open = _
        frame_fetch = cv2.resize(frame, (480, 270))

class deepThread(threading.Thread):
    """
        The thread to do the segmentation and classification
    """
    def __init__(self):
        threading.Thread.__init__(self)

    def start(self):
        threading.Thread.__init__(self)
        threading.Thread.start(self)

    def join(self, _sec):
        threading.Thread.join(self, _sec)

    def run(self):
        global segment_model
        global frame_process
        global result_segment
        global result_scoring
        global graph
        global sess
        global frame_index

        with graph.as_default():
            result_segment = segment_model.test(
                np.expand_dims(frame_process, 0)
            )
            if frame_index % 3 == 0:
                result_scoring = scoring_model.test(
                    np.expand_dims(frame_process, 0)
                )      

class postProcThread(threading.Thread):
    """
        The thread to merge the whole result into one image
    """
    def __init__(self):
        threading.Thread.__init__(self)

    def start(self):
        threading.Thread.__init__(self)
        threading.Thread.start(self)

    def join(self, _sec):
        threading.Thread.join(self, _sec)

    def run(self):
        global show_frame
        global segment_input
        global scoring_input
        global result_segment_lap
        global result_final
        global kernel

        # Merge the segment and scoring result
        result_final = mergeSegmentAndScoringRes(show_frame, 
            segment_input, scoring_input
        )

if __name__ == '__main__':
    # Load model
    segment_model = UNet(img_height, img_width, save_path='model/unet.h5')
    scoring_model = ScoreNet(save_path='model/47.h5')

    # Start video
    cap = cv2.VideoCapture(video_name)

    # Define the thread object
    fetch_thread = fetchImgThread()
    deep_thread = deepThread()
    post_proc_thread = postProcThread()

    # Grab 1st frame
    fetch_thread.start()
    fetch_thread.join(1)

    # Pass the input object and clean the previous status
    frame_process = np.copy(frame_fetch)
    frame_fetch = None

    # Grab 2nd frame
    fetch_thread.start()
    deep_thread.start()
    fetch_thread.join(5)
    deep_thread.join(5)

    # Pass the input object and clean the previous status
    show_frame = np.copy(frame_process)
    segment_input = np.copy(result_segment[0])
    scoring_input = np.copy(result_scoring)
    result_segment = None
    result_scoring = None
    frame_process = np.copy(frame_fetch)
    frame_fetch = None

    while cap.isOpened():
        _time = time.time()

        # Start work and wait the worker done
        fetch_thread.start()
        deep_thread.start()
        post_proc_thread.start()
        fetch_thread.join(5)
        deep_thread.join(5)
        post_proc_thread.join(5)
            
        # Show the result
        cv2.imshow('origin', show_frame)
        cv2.imshow('segment', segment_input)
        cv2.imshow('scoring', coder.decodeByVector(show_frame, scoring_input))
        cv2.imshow('final', cv2.resize(result_final, (0, 0), fx=1.5, fy=1.5))

        # Move window to regid position
        cv2.moveWindow('origin', 200, 0)
        cv2.moveWindow('segment', 200, 300)
        cv2.moveWindow('scoring', 600, 0)
        cv2.moveWindow('final', 600, 300)

        # Pring fps
        if timer > 1.0:
            print "fps: ", fps_index / timer
            fps_index = 0
            timer = 0

        # judge if we want to break
        if (cv2.waitKey(1) & 0xFF == ord('q')):
            break
        if not unicode(video_name).isnumeric():
            if frame_index > break_frame_index:
                break

        # Pass the input object and clean the previous status
        result_segment_lap = None
        result_final = None
        show_frame = np.copy(frame_process)
        segment_input = np.copy(result_segment[0])
        scoring_input = np.copy(result_scoring)
        result_segment = None

        # Clear the scoring result if predict ScoreNet in next frame
        if frame_index % 3 == 2:
            result_scoring = None
        frame_process = np.copy(frame_fetch)
        frame_fetch = None

        # Update fps computation variable
        frame_index += 1
        fps_index += 1
        timer += (time.time() - _time)

    cap.release()