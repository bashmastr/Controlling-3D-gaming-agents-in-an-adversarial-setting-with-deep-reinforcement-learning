import numpy as np
from PIL import ImageGrab
import cv2
import time

def screen_record(): 
    flag = 10
    while(flag):

        global printscreen

        image = ImageGrab.grab(bbox=(70,70,1430,1685))
        printscreen = np.array(image)
        grayscale_image = cv2.cvtColor(printscreen, cv2.COLOR_BGR2GRAY)

        cv2.imshow('window', grayscale_image)
        if cv2.waitKey(25) & 0xFF == ord('q'):
            cv2.destroyAllWindows()
            break
        if cv2.waitKey(25) & 0xFF == ord('w'):
            image.save("screen_shot.png")
            print("Saved current window as image")
        time.sleep(10) 
        flag -= 1

if __name__ == '__main__':
    screen_record()