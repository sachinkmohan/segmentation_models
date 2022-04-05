#!/usr/bin/env python
# coding: utf-8

# ### Loading the converted tensor RT pb graph

# In[3]:


import tensorflow as tf
from tensorflow.python.platform import gfile

import segmentation_models as sm

BACKBONE = 'efficientnetb3'
preprocess_input = sm.get_preprocessing(BACKBONE)

tf_config = tf.ConfigProto()
tf_config.gpu_options.allow_growth = True
#tf_config.gpu_options.per_process_gpu_memory_fraction = 0.2
tf_sess = tf.Session(config=tf_config)
#GRAPH_PB_PATH = './trained_models_local/saved_for_lab/tf_model_base_1502.pb'
#GRAPH_PB_PATH = './converted_trt_graph/trt_graph_base_30.pb'
#GRAPH_PB_PATH = './converted_trt_graph/trt_graph_st_prg_1601_80p.pb'
GRAPH_PB_PATH_FROZEN='./save-pb/frozen_model_100ep.pb'
#GRAPH_PB_PATH_TRT = './converted_trt_graph/trt_graph_ss_model.pb'

with tf.Session() as sess:
   print("load graph")
   with gfile.FastGFile(GRAPH_PB_PATH_FROZEN,'rb') as f:
       graph_def = tf.GraphDef()
   graph_def.ParseFromString(f.read())
   sess.graph.as_default()
   tf.import_graph_def(graph_def, name='')
   graph_nodes=[n for n in graph_def.node]
   names = []
   for t in graph_nodes:
      names.append(t.name)
      
    # print operations

   #print(names)


# ### Importing the graph

# In[4]:



tf.import_graph_def(graph_def, name='')


# ### loading the first and last layers

# In[5]:


tf_input = tf_sess.graph.get_tensor_by_name('input_1:0')
print(tf_input)

tf_predictions = tf_sess.graph.get_tensor_by_name('sigmoid/Sigmoid:0')
print(tf_predictions)


# ### Real time prediction of the mask from the camera

# In[9]:


import cv2
import numpy as np
#import matplotlib.pyplot as plt
#from IPython.display import clear_output
import time


from tensorflow.python.keras.backend import set_session
graph = tf.get_default_graph()


#Capture the video from the camera

def cam_video_inference():
    #cap = cv2.VideoCapture(0) # If you are using the camera
    cap = cv2.VideoCapture('/home/mohan/git/backups/drive.mp4')
    while cap.isOpened():
        ret, frame = cap.read()
        #frame2 = frame.reshape((300,480))
        image_resized2 = cv2.resize(frame, (480,320))
        img_inf = preprocess_input(image_resized2) #adding this preprocessing step mentioned here - https://github.com/qubvel/segmentation_models/issues/373
        #final_img = np.expand_dims(img_inf, axis = 0)
        #print(resized.shape)
        #frame2 = np.expand_dims(resized, axis=0)

        #Run the Detections using model.predict

        if ret:
            t0 = time.time()
            with graph.as_default():
                set_session(sess)
                inputs, predictions = tf_sess.run([tf_input, tf_predictions], feed_dict={
                tf_input: img_inf[None, ...]
            })
            #cv2.imwrite('file5.jpeg', 255*predictions.squeeze())
            pred_image = 255*predictions.squeeze()

            ##converts pred_image to CV_8UC1 format so that ColorMap can be applied on it
            u8 = pred_image.astype(np.uint8)

            #Color map autumn is applied to the CV_8UC1 pred_image
            im_color = cv2.applyColorMap(u8, cv2.COLORMAP_AUTUMN)
            cv2.imshow('input image', image_resized2)
            cv2.imshow('prediction mask',im_color)
            t1 = time.time()
            #print('Runtime: %f seconds' % (float(t1 - t0)))
            #cv2.waitKey(0)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break


        else:
            cap.release()
            break

    cap.release()
    cv2.destroyAllWindows()

cam_video_inference()
