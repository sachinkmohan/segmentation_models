
import onnxruntime as rt
import time
import cv2
import numpy as np

import segmentation_models as sm

def inference_video():
    sess = rt.InferenceSession("./seg_model_unet_40_ep_op13.onnx", providers=['CUDAExecutionProvider'])
    input_name = sess.get_inputs()[0].name
    label_name = sess.get_outputs()[0].name

    cap =cv2.VideoCapture('/home/mohan/git/backups/drive.mp4')
    while cap.isOpened():
        ret, frame = cap.read()
        image_resized2 = cv2.resize(frame, (480,320))
        img_inf = preprocess_input(image_resized2) ## adding the preprocessing step for the image -> Link found from -> https://github.com/qubvel/segmentation_models/issues/373#issuecomment-660081448
        final_img = np.array(np.expand_dims(img_inf, axis = 0), dtype=np.float32)

        if ret:
            #Detections which returns a list
            t0 = time.time()
            detections = sess.run([label_name], {input_name: final_img})
            t1 = time.time()
            print(t1-t0)
            #List converted to the numpy array
            arr = np.asarray(detections)
            pred_image = 255*arr.squeeze()
            u8 = pred_image.astype(np.uint8)
            im_color = cv2.applyColorMap(u8, cv2.COLORMAP_AUTUMN)
        cv2.imshow('original', image_resized2)
        cv2.imshow('prediction mask', im_color)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
            cv2.destroyAllWindows()
            #print('done')

def inference_image():
    sess = rt.InferenceSession("./seg_model_unet_40_ep_op13.onnx", providers=['CUDAExecutionProvider'])

    input_name = sess.get_inputs()[0].name
    label_name = sess.get_outputs()[0].name

    img= cv2.imread('0001TP_009060.png')
    image_resized2 = cv2.resize(img, (480,320))
    img_inf = preprocess_input(image_resized2) ## adding the preprocessing step for the image -> Link found from -> https://github.com/qubvel/segmentation_models/issues/373#issuecomment-660081448
    final_img = np.array(np.expand_dims(img_inf, axis = 0), dtype=np.float32)

    #Detections which returns a list
    detections = sess.run([label_name], {input_name: final_img})
    #List converted to the numpy array
    arr = np.asarray(detections)
    pred_image = 255*arr.squeeze()
    u8 = pred_image.astype(np.uint8)
    im_color = cv2.applyColorMap(u8, cv2.COLORMAP_AUTUMN)

    cv2.imshow('prediction mask', im_color)
    if cv2.waitKey(0) & 0xFF == ord('q'):
        cv2.destroyAllWindows()
    print('done')

if __name__ == "__main__":
    try:
        BACKBONE = 'efficientnetb3'
        preprocess_input = sm.get_preprocessing(BACKBONE)
        #inference_image()
        inference_video()

    except BaseException as err:
        cv2.destroyAllWindows()
        raise err
