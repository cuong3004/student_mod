from object_detector import ObjectDetect, Classify
import cv2
from picamera.array import PiRGBArray
from picamera import PiCamera 
from PIL import Image
import numpy as np
import time
import sys
import RPi.GPIO as GPIO
import time
from PIL import Image, ImageOps
from imutils.video import VideoStream
from imutils.video import FPS
from picamera.array import PiRGBArray
from picamera import PiCamera
import argparse
import imutils
import time
import cv2
from pyrf24 import RF24, RF24_PA_MAX, RF24_250KBPS




class NRF24L01:
    def __init__(self):
        
        self.radio = RF24(22, 0)
        self.radio.begin()
        self.radio.setDataRate(RF24_250KBPS)
        self.radio.openWritingPipe(0xF0F0F0F0D2)
        self.radio.stopListening()
    
    def write(self, data):
        self.radio.write(data)
        time.sleep(0.01)




rf24 = NRF24L01()

#label_dir = { 0:"Angry", 1:"Disgust", 2:"Fear", 3:"Happy", 4:"Sad", 5:"Surprise", 6:"Neutral"}
eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')



time_att = time.time()
# font
font = cv2.FONT_HERSHEY_SIMPLEX
  
# org
org = (50, 50)
  
# fontScale
fontScale = 1
   
# Blue color in BGR
color = (255, 0, 0)
  
# Line thickness of 2 px
thickness = 2
import math

import time

class Timer:
    def __init__(self):
        self.start_time = None
        self.end_time = None

    def start(self):
        self.start_time = time.time()

    def stop(self):
        self.end_time = time.time()

    def elapsed_time(self):
        if self.start_time is None:
            return None
        elif self.end_time is None:
            return time.time() - self.start_time
        else:
            return self.end_time - self.start_time

timer = Timer()

def padding(img, expected_size):
    desired_size = expected_size
    delta_width = desired_size - img.size[0]
    delta_height = desired_size - img.size[1]
    pad_width = delta_width // 2
    pad_height = delta_height // 2
    padding = (pad_width, pad_height, delta_width - pad_width, delta_height - pad_height)
    return ImageOps.expand(img, padding)


def resize_with_padding(frame, expected_size):
    img = Image.fromarray(frame)
    img.thumbnail((expected_size[0], expected_size[1]))
    delta_width = expected_size[0] - img.size[0]
    delta_height = expected_size[1] - img.size[1]
    pad_width = delta_width // 2
    pad_height = delta_height // 2
    pad_width = pad_height = 0
    padding = (pad_width, pad_height, delta_width - pad_width, delta_height - pad_height)
    return np.asarray(ImageOps.expand(img, padding))

        
if __name__ == '__main__':
    while True:
        try:
            vs = VideoStream(usePiCamera=True, resolution=(1024,608)).start()
            break
        except:
            print("ERROR, Again")
            time.sleep(2.0)
    time.sleep(2.0)
    # fps = FPS().start()

    frame_rate_calc = 1
    freq = cv2.getTickFrequency()
    font = cv2.FONT_HERSHEY_SIMPLEX

    object_detector = ObjectDetect(
                "model_tflite/best-int8_edgetpu.tflite",
                "model_tflite/nms.tflite"
            )
    object_detector_eyes = ObjectDetect(
                "eyes_model/best-int8_edgetpu.tflite",
                "eyes_model/nms.tflite"
            )
    classify = Classify()
    
    windowName = 'DIY-HUS'

    cv2.namedWindow(windowName)
    
    s_happy = []
    s_sad = []
    s_surprise = []
    s_anger = []
    
    s_1 = []
    s_2 = []
    
    left_pad = np.ones((240+100, 100,3), dtype='uint8') * 255
    right_pad = np.ones((240+100,100,3), dtype='uint8') * 255
    
    count_path_1 = 0
    count_path_2 = 0
    count_path_3 = 0
    count_path_4 = 0
    
    time1 = time.time()
        
    timer.start()
    
    img_logo = cv2.imread("image_logo.jpg")
    img_logo = cv2.resize(img_logo, (416,608))
    img_logo = cv2.cvtColor(img_logo, cv2.COLOR_BGR2RGB)
    while True:
        
        frame = vs.read()
        
        
        #print(frame.shape)
        h, w, _ = frame.shape
        min_size = min(h, w)
        center_min_size = min_size//2
        centerX = w // 2
        centerY = h // 2
        cropX = centerX - min_size // 2;
        cropY = centerY - min_size // 2;
        
        cropX_2 = cropX*2
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = cv2.flip(frame, 1) 
        
        
        
        
        #bot_pad = np.ones((100, w, 3), dtype='uint8') * 255
        #print(frame.shape)
        #print(min_size)
        #print(cropY+min_size, cropX+min_size)
        #pad = np.zeros((h, cropX,3), dtype=np.uint8)
        #cv2.rectangle(frame, (0,0), (cropX_2,h), (0, 0, 0), -1)
        #cv2.rectangle(frame, (cropX+min_size,0), (w,h), (10, 0, 0), -1)
        
        #print(pad)
        
        #print(img_logo.shape)
        #frame[:,:cropX_2] += img_logo
        frameCut = frame[cropY:cropY+min_size, cropX:cropX+min_size]
        #frame = np.concatenate([pad, frameCut, pad], 1)
        #frameCutPad = resize_with_padding(frameCut, (320, 320))
        frameCutPad = cv2.resize(frameCut, (160,160))
        
        t1 = cv2.getTickCount()
        
        boxes, classes, scores =  object_detector.detect(frameCutPad)
        """
        boxes_filter = []
        classes_filter = []
        scores_filter = []
        
        for i in range(len(scores)):
            if ((scores[i] > 0.4) and (scores[i] <= 1.0)):
                scores_filter.append(scores[i])
                boxes_filter.append(boxes[i])
                classes_filter.append(classes[i])"""
        
        #print(classes.max(), scores.max())
        y_max_list = []
        #print()
        #assert np.all(np.diff(scores) <= 0)
        
        is_path_1 = False
        is_path_2 = False
        is_path_3 = False
        is_path_4 = False
        
        
        
        
        #rf24.write(b'11')
        
        clip_path_1 = int(min_size*0.25)
        clip_path_2 = int(min_size*0.5)
        clip_path_3 = int(min_size*0.75)
        
        
        is_where = 0
        for i in range(len(scores)):
            if ((scores[i] > 0.65) and (scores[i] <= 1.0)):
                #print("deteect")
                H = frameCutPad.shape[0]
                W = frameCutPad.shape[1]
        
                xmin = int(max(1,(boxes[i][0] * min_size)))
                ymin = int(max(1,(boxes[i][1] * min_size)))
                xmax = int(min(min_size,(boxes[i][2] * min_size)))
                ymax = int(min(min_size,(boxes[i][3] * min_size)))
                if (xmax-xmin)*(ymax-ymin) < 36:
                    continue
                #print((xmin,ymin), (xmax,ymax))
                if xmax <= int(clip_path_2):
                    if is_path_1:
                        continue
                    is_path_1 = True
                    is_where = 1
                    #rf24.write(b'11')

                elif xmin > clip_path_2 and xmax <= min_size:
                    if is_path_4:
                        continue
                    is_path_4 = True
                    is_where = 4
                    #rf24.write(b'41')
                else:
                    continue
                
                    
                
                y_max_list.append(ymax)
                
                
                
                
                cutframe = frameCut[ymin:ymax,xmin:xmax]
                h_eye, w_eye = cutframe.shape[:2]
                frameCutPad_eye = cv2.resize(cutframe, (64,64))
                boxes_eyes, classes_eyes, scores_eyes =  object_detector_eyes.detect(frameCutPad_eye)
                count_eyes = 0
                for j in range(len(scores_eyes)):
                    if ((scores_eyes[j] > 0.5) and (scores_eyes[j] <= 1.0)):
                        
                        x1 = int(max(1,(boxes_eyes[j][0] * w_eye)))
                        y1 = int(max(1,(boxes_eyes[j][1] * h_eye)))
                        x2 = int(min(w_eye,(boxes_eyes[j][2] * w_eye)))
                        y2 = int(min(h_eye,(boxes_eyes[j][3] * h_eye)))
                        
                        cv2.rectangle(cutframe, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        count_eyes += 1
                
                
                if count_eyes == 2:
                    if is_where == 1:
                        is_path_1 += 1
                    if is_where == 2:
                        is_path_2 += 1
                    if is_where == 3:
                        is_path_3 += 1
                    if is_where == 4:
                        is_path_4 += 1
                        
                
                #print(cutframe.shape)
                
                #gray = cv2.cvtColor(cutframe, cv2.COLOR_BGR2GRAY)
                #print(gray.shape)
                #eyes = eye_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
                #for (x, y, w, h) in eyes:
                #    cv2.rectangle(cutframe, (x, y), (x + w, y + h), (0, 255, 0), 2)
                #out = classify(cutframe)[0]
                
                cv2.rectangle(frameCut, (xmin,ymin), (xmax,ymax), (10, 255, 0), 3)
        
        #frameCut = cv2.line(frameCut, (clip_path_1,0), (clip_path_1,min_size), (0,255,0), 1)
        frameCut = cv2.line(frameCut, (clip_path_2,0), (clip_path_2,min_size), (0,255,0), 1)
        #frameCut = cv2.line(frameCut, (clip_path_3,0), (clip_path_3,min_size), (0,255,0), 1)
        
        if timer.elapsed_time() > 0.5:
            print(timer.elapsed_time(), True)
            timer.start()

            if not is_path_1 or is_path_1==1 :
                print(count_path_1)
                count_path_1 += 1
            else:
                count_path_1 = 0
            if not is_path_2 or is_path_2==1 :
                count_path_2 += 1
            else:
                count_path_2 = 0
            if not is_path_3 or is_path_3==1 :
                count_path_3 += 1
            else:
                count_path_3 = 0
            if not is_path_4 or is_path_4==1 :
                count_path_4 += 1
            else:
                count_path_4 = 0
            
            print(count_path_1)
            
            if count_path_1 >= 16:
                rf24.write(b'33')
            elif count_path_1 < 16 and count_path_1 >= 8:
                rf24.write(b'32')
            elif count_path_1 < 8 and count_path_1 >= 4:
                rf24.write(b'31')
            else:
                rf24.write(b'30')

            
            if count_path_4 >= 16:
                rf24.write(b'23')
            elif count_path_4 < 16 and count_path_4 >= 8:
                rf24.write(b'22')
            elif count_path_4 < 8 and count_path_4 >= 4:
                rf24.write(b'21')
            else:
                rf24.write(b'20')
            
            
            
                    
                
                

                
                
        

        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        cv2.putText(frame,"FPS: {0:.2f}".format(frame_rate_calc),(20,h-50),font,0.6,(0,0,255),2,cv2.LINE_AA)
        #cv2.imwrite("image.jpg", frameCutPad)
        #frame = cv2.line(frame, (centerX,0), (centerX,h), (0,255,0), 1)
        #frame = cv2.line(frame, (0,centerY), (w,centerY), (0,255,0), 1)
        print(frame.shape)
        cv2.imshow(windowName, frame)
        
        t2 = cv2.getTickCount()
        
        time1 = (t2-t1)/freq
        frame_rate_calc = 1/time1
        
        if cv2.waitKey(1) == ord('q'):
            break

        # fps.update()

    cv2.destroyAllWindows()
    vs.stop()
