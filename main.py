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

def fetchImgThread():
    global cap
    global is_cap_open
    global frame_fetch
    _, frame = cap.read()
    is_cap_open = _
    frame_fetch = cv2.resize(frame, (480, 270))

def deepThread():
    global segment_model
    global frame_process
    global result_segment
    global result_scoring
    global graph
    global sess
    global frame_index
    
    _time = time.time()
    with tf.device('/gpu:0'):
    	with graph.as_default():
    	    result_segment = segment_model.test(
    	        np.expand_dims(frame_process, 0)
    	    )
            if frame_index % 3 == 0:
    	        result_scoring = scoring_model.test(
    	            np.expand_dims(frame_process, 0)
    	        )      
    #print "deep time: ", time.time() - _time  

def postProcThread():
    global show_frame
    global segment_input
    global scoring_input
    global result_segment_lap
    global result_final
    global kernel

    """
    # Generate laplacian edge image
    segment_input = cv2.dilate(segment_input, kernel)
    _, result_segment_lap = cv2.threshold(segment_input, 127, 255, cv2.THRESH_BINARY)
    result_segment_lap = cv2.Laplacian(result_segment_lap, cv2.CV_32F)
    segment_input = segment_input.astype(np.uint8)
    """
    
    # Merge the segment and scoring result
    """
    result_final = mergeSegmentAndScoringRes(show_frame, 
        segment_input, 
        scoring_input,
        cv2.connectedComponents(segment_input, 4, cv2.CV_32S)[1],
        fast_plot=False
    )
    """
    _time = time.time()
    result_final = mergeSegmentAndScoringRes(show_frame, 
        segment_input, 
        scoring_input,
        fast_plot=False
    )
    #print "draw time: ", time.time() - _time  

if __name__ == '__main__':
    # Load model
    segment_model = UNet(img_height, img_width, save_path='model/unet.h5')
    scoring_model = ScoreNet(save_path='model/47.h5')

    # Start video
    cap = cv2.VideoCapture('./video/1.mp4')

    # Grab 1st frame
    fetch_thread = threading.Thread(target=fetchImgThread)
    fetch_thread.start()
    fetch_thread.join(1)

    # Pass the input object and clean the previous status
    frame_process = np.copy(frame_fetch)
    frame_fetch = None

    # Grab 2nd frame
    fetch_thread = threading.Thread(target=fetchImgThread)
    deep_thread = threading.Thread(target=deepThread)
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
        is_cap_open = cap.isOpened()
        if is_cap_open:
            _time = time.time()

            # Define the continuous works
            fetch_thread = threading.Thread(target=fetchImgThread)
            deep_thread = threading.Thread(target=deepThread)
            post_proc_thread = threading.Thread(target=postProcThread)

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
            #cv2.resize(result_final, (0, 0), fx=1.5, fy=1.5)
            cv2.imshow('final', cv2.resize(result_final, (0, 0), fx=1.5, fy=1.5))

            # Move window to regid position
            cv2.moveWindow('origin', 200, 0)
            cv2.moveWindow('segment', 200, 300)
            cv2.moveWindow('scoring', 600, 0)
            cv2.moveWindow('final', 600, 300)

            if timer > 1.0:
                # print "frame index: ", frame_index, "\ttime spend: ", time.time() - _time
                print "fps: ", fps_index / timer
                fps_index = 0
                timer = 0

            if (cv2.waitKey(1) & 0xFF == ord('q')):
                break
            if frame_index > 500:
                break

            # Pass the input object and clean the previous status
            result_segment_lap = None
            result_final = None
            show_frame = np.copy(frame_process)
            segment_input = np.copy(result_segment[0])
            scoring_input = np.copy(result_scoring)
            result_segment = None

            if frame_index % 3 == 2:
                result_scoring = None

            frame_process = np.copy(frame_fetch)
            frame_fetch = None

            frame_index += 1
            fps_index += 1
            timer += (time.time() - _time)

        else:
            print "failt..."
            continue


        if cv2.waitKey(10) & 0xFF == ord('q'):
            break
    
    
    cap.release()
