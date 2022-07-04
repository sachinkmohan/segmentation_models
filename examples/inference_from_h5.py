import segmentation_models as sm

import cv2
import time

import numpy as np

# segmentation_models could also use `tf.keras` if you do not have Keras installed
# or you could switch to other framework using `sm.set_framework('tf.keras')`


BACKBONE = 'efficientnetb3'
BATCH_SIZE = 8
CLASSES = ['car']
LR = 0.0001
EPOCHS = 100

preprocess_input = sm.get_preprocessing(BACKBONE)

# define network parameters
n_classes = 1 if len(CLASSES) == 1 else (len(CLASSES) + 1)  # case for binary and multiclass segmentation
activation = 'sigmoid' if n_classes == 1 else 'softmax'

#create model
model = sm.Unet(BACKBONE, classes=n_classes, activation=activation)

model.load_weights('ss_unet_40_ep.h5')


def cam_video_inference():
    # cap = cv2.VideoCapture(0) # If you are using the camera
    cap = cv2.VideoCapture('/home/mohan/git/backups/drive.mp4')
    prev_frame_time = 0
    new_frame_time = 0

    while cap.isOpened():
        new_frame_time = time.time()
        ret, frame = cap.read()
        # frame2 = frame.reshape((300,480))
        image_resized2 = cv2.resize(frame, (480, 320))
        img_inf = preprocess_input(
            image_resized2)  # adding this preprocessing step mentioned here - https://github.com/qubvel/segmentation_models/issues/373

        if ret:
            t0 = time.time()
            im3 = np.expand_dims(img_inf, axis=0)
            y_seg = model.predict(im3)
            # cv2.imwrite('file5.jpeg', 255*predictions.squeeze())
            fps = 1 / (new_frame_time - prev_frame_time)
            prev_frame_time = new_frame_time
            fps = int(fps)
            print(fps)
            pred_image = 255 * y_seg.squeeze()

            ##converts pred_image to CV_8UC1 format so that ColorMap can be applied on it
            u8 = pred_image.astype(np.uint8)

            # Color map autumn is applied to the CV_8UC1 pred_image
            im_color = cv2.applyColorMap(u8, cv2.COLORMAP_AUTUMN)
            cv2.imshow('input image', image_resized2)
            cv2.imshow('TensorFlow Network 4', im_color)
            t1 = time.time()
            # print('Runtime: %f seconds' % (float(t1 - t0)))
            # cv2.waitKey(0)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break


        else:
            cap.release()
            break

    cap.release()
    cv2.destroyAllWindows()


cam_video_inference()
