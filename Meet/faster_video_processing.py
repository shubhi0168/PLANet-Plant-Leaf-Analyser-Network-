# -*- coding: utf-8 -*-
"""
Created on Wed Feb 12 21:57:29 2020

@author: Meet
"""
import imutils
import os
from imutils.video import FileVideoStream
from imutils.video import FPS
import time
from threading import Thread
import sys
import cv2
import argparse
import numpy as np

if sys.version_info >= (3, 0):
    from queue import Queue
else:
    from Queue import Queue

class FileVideoStream:
    def __init__(self,path,queueSize=128):
        self.stream = cv2.VideoCapture(path)
        self.stopped = False
        
        self.Q = Queue(maxsize = queueSize)
        
    def start(self):
        t = Thread(target=self.update,args=())
        t.daemon = True
        t.start()
        return self

    def update(self):
        while True:
            if self.stopped:
                return
            if not self.Q.full():
                (grabbed, frame) = self.stream.read()
            
                if not grabbed:
                    self.stop()
                    return
            
                self.Q.put(frame)

    def read(self):
        return self.Q.get()

    def more(self):
        return self.Q.qsize() > 0

    def stop(self):
        self.stopped = True
        
if __name__ == "__main__":
    path = r'D:\real-life-violence-situations-dataset\Real Life Violence Dataset\Violence\V_19.mp4'
    print("[INFO] starting video file thread...")
    fvs = FileVideoStream(path).start()
    time.sleep(1.0)
    
    fps = FPS().start()
    
    while fvs.more():
        frame = fvs.read()
        frame = imutils.resize(frame, width=224, height=224)
        frame = cv2.cvtColor(frame, cv2.BGR2RGB)
        frame = np.dstack([frame, frame, frame])
        
        cv2.imshow("Frame",frame)
        cv2.waitkey(1)
        fps.update()
        
    fps.stop()
    print("[INFO] elasped time: {:.2f}".format(fps.elapsed()))
    print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))
    # do a bit of cleanup
    cv2.destroyAllWindows()
    fvs.stop()