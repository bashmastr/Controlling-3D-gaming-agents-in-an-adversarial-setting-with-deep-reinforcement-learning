from PIL import Image, ImageEnhance
import numpy as np
import cv2
from imageOperation import ImageOperation


class GetTime:
    def __init__(self, image_path, saved_location):
        self.image_path = image_path
        self.saved_location = saved_location
        self.coordinates = (1380, 30, 1700,150)


    def captureNumberArea(self):
        image_obj = Image.open(self.image_path)
        cropped_image = image_obj.crop(self.coordinates)
        first_digit_image = cropped_image.crop((0, 10, 150, 120))
        second_digit_image = cropped_image.crop((155, 10, 315, 120))
        first_digit_operation = ImageOperation(first_digit_image)
        second_digit_operation = ImageOperation(second_digit_image)
        first_digit_operation.thresholdingImage(100)
        second_digit_operation.thresholdingImage(100)

    def captureNumberAreaOld(self):
        image_obj = Image.open(self.image_path)
        cropped_image = image_obj.crop(self.coordinates)
        first_digit_image = cropped_image.crop((0, 10, 150, 120))
        second_digit_image = cropped_image.crop((155, 10, 315, 120))
        # cropped_image = cropped_image.convert('LA')
        first_digit_image = first_digit_image.convert("LA")
        second_digit_image = second_digit_image.convert("LA")
        first_digit_enhancer = ImageEnhance.Contrast(first_digit_image)
        second_digit_enhancer = ImageEnhance.Contrast(second_digit_image)
        factor =  3
        first_digit_image = first_digit_enhancer.enhance(factor)
        second_digit_image = second_digit_enhancer.enhance(factor)
        # image_values = np.array(first_digit_image)
        # print(image_values.reshape(16000))
        cropped_image.save(self.saved_location)
        # first_digit_image.show()
        # second_digit_image.show()
        # cropped_image.show()




if __name__ == '__main__':
    for i in range(20):
        print("###############"+str(i+1))
        image_path = "tekken-"+str(i+1)+".png"
        saved_location = "crpped-number.png"
        instance = GetTime(image_path, saved_location)
        instance.captureNumberArea()
        