import time
import cv2
import mss
import numpy
from PIL import ImageGrab
import pyautogui




# img = Image.open('tekken-7-4k.png')
# answer2 = get_text(img)


class ScreenCapture:
    def __init__(self):
        # title of our window
        self.title = "FPS benchmark"
        
        # set start time to current time
        self.start_time = time.time()
        
        #set end time accordingly
        self.end_time = start_time + 10
        
        # displays the frame rate every 2 second
        self.display_time = 2
        
        # Set primarry FPS to 0
        self.fps = 0
        
        # Load mss library as sct
        self.sct = mss.mss()
        
        # Set monitor size to capture to MSS
        self.monitor = {"top": 40, "left": 0, "width": 800, "height": 640}
        
        # Set monitor size to capture
        mon = (0, 40, 800, 640)
    
    def screenRecordMSS(self):
        #begin our loop
        while self.start_time < self.end_time:
            
            # Get raw pixels from the screen, save it to a Numpy array
            img = numpy.array(self.sct.grab(self.monitor))
            
            # to ger real color we do this:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            # cv2.imshow(title, img)
            
            # add one to fps
            self.fps+=1
            # calculate time difference
            TIME = time.time() - start_time
            
            # this If statement is to check FPS when display time is passed            
            if (TIME) >= self.display_time :
                print("FPS: ", fps / (TIME))
                self.fps = 0
                self.start_time = time.time()
                
            
    
    
    def screenRecordPIL(self):
        
        # begin our loop
        while self.start_time < self.end_time:
            
            # Get raw pixels from the screen, save it to a Numpy array
            img = numpy.asarray(ImageGrab.grab(bbox=self.mon))
            
            # to ger real color we do this:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            # Display the picture
            # cv2.imshow(title, img)
            
            # add one to fps
            self.fps+=1
            # calculate time difference
            TIME = time.time() - start_time
            
            # this If statement is to check FPS when display time is passed
            if (TIME) >= display_time :
                print("FPS: ", fps / (TIME))
                # set fps again to zero
                self.fps = 0
                # set start time to current time again
                self.start_time = time.time()
        
                
    def screenRecordPyautogui(self):
        while self.start_time < self.end_time:
            img = pyautogui.screenshot('tekken-7-4k.png')
            
            # Display the picture
            # cv2.imshow(title, img)
            
            # add one to fps
            self.fps+=1
            # calculate time difference
            TIME = time.time() - start_time
            
            # this If statement is to check FPS when display time is passed
            if (TIME) >= display_time :
                print("FPS: ", fps / (TIME))
                # set fps again to zero
                self.fps = 0
                # set start time to current time again
                self.start_time = time.time()
