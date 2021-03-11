from PIL import Image
import numpy as np 



class ImageOperation:
    def __init__(self,image):
        # read image to array
        # self.image = Image.open(image_file)
        # image.show()
        
        # convert image into greyscale
        self.image_grey = image.convert("L")
        # img.show()
        
        self.dimension = self.image_grey.size
        #print(self.dimension)

        
        
    def round_int(self,x):
        if x == float("inf") or x == float("-inf"):
            return 0 # or x or return whatever makes sense
        return int(round(x))
    
    
    
    def negativeImage(self):
        image_values = np.array(self.image_grey)
        #print(image_values.shape)
        image_values = image_values.reshape(self.dimension[0]*self.dimension[1])
        for i in range(len(image_values)):
            image_values[i] = 255 - image_values[i]
                        
        image_values = image_values.reshape(self.dimension[1], self.dimension[0])
        # Creates PIL image
        img = Image.fromarray(image_values, 'L')
        img.show()
        
    
    def thresholdingImage(self, threshold):
        image_values = np.array(self.image_grey)
        count = 0
        #print(image_values.shape)
        image_values = image_values.reshape(self.dimension[0]*self.dimension[1])
        for i in range(len(image_values)):
            if image_values[i] > threshold:
                image_values[i] = 255
                count += 1
            else:
                image_values[i] = 0
        
        image_values = image_values.reshape(self.dimension[1], self.dimension[0])
        # Creates PIL image
        return count
        img = Image.fromarray(image_values, 'L')
        # img.show()
        
        
    def logrithmicImage(self, c=1):
        image_values = np.array(self.image_grey)
        #print(image_values.shape)
        
        image_values = image_values.reshape(self.dimension[0]*self.dimension[1])
        if c == 1:
            c = 255 / (np.log(1 + np.max(image_values)))
            
        for i in range(len(image_values)):
            image_values[i] = c * np.log(1+image_values[i])
        
        image_values = image_values.reshape(self.dimension[1], self.dimension[0])
        # Creates PIL image
        img = Image.fromarray(image_values, 'L')
        img.show()
        
        
    def inverseLogrithmicImage(self, c=1):
        
        image_values = np.array(self.image_grey)
        #print(image_values.shape)
        image_values = image_values.reshape(self.dimension[0]*self.dimension[1])
        if c == 1:
            c = 255 / (np.log(1 + np.max(image_values)))
        for i in range(len(image_values)):
            image_values[i] = c * self.round_int(np.exp(image_values[i]))
        
        image_values = image_values.reshape(self.dimension[1], self.dimension[0])
        # Creates PIL image
        img = Image.fromarray(image_values, 'L')
        img.show()
    
    
    
    def powerLawImage(self, gamma, c=1):
        image_values = np.array(self.image_grey)
        #print(image_values.shape)
        if c == 1:
            c = 255 / (np.log(1 + np.max(image_values)))
        image_values = image_values.reshape(self.dimension[0]*self.dimension[1])
        for i in range(len(image_values)):
            image_values[i] = c * (image_values[i] ** gamma)
        
        image_values = image_values.reshape(self.dimension[1], self.dimension[0])
        # Creates PIL image
        img = Image.fromarray(image_values, 'L')
        img.show()
        
        
    def contrastStretchingImage(self, a, b, c, d):
        image_values = np.array(self.image_grey)
        #print(image_values.shape)
        
        image_values = image_values.reshape(self.dimension[0]*self.dimension[1])
        for i in range(len(image_values)):
            image_values[i] = ((image_values[i]-c) * ( (b - a)/ (d-c)))+a
        
        image_values = image_values.reshape(self.dimension[1], self.dimension[0])
        # Creates PIL image
        img = Image.fromarray(image_values, 'L')
        img.show()
        
    
    def intensitySlicingImage(self, c, d, k,l=1):
        image_values = np.array(self.image_grey)
        #print(image_values.shape)
        
        image_values = image_values.reshape(self.dimension[0]*self.dimension[1])
        for i in range(len(image_values)):
            if image_values[i] >= c and image_values[i] <= d:
                image_values[i] =  k 
            if l != 1:
                image_values[i] =  l
        image_values = image_values.reshape(self.dimension[1], self.dimension[0])
        # Creates PIL image
        img = Image.fromarray(image_values, 'L')
        img.show()

if __name__ == "__main__":
    # image_file = '/home/halcyoona/Downloads/IMG_5084.jpg'
    image_file = 'cropped-player-one.png'
    instance = ImageOperation(image_file)
    instance.negativeImage()
    instance.thresholdingImage(100)
    instance.logrithmicImage()
    instance.inverseLogrithmicImage()
    instance.powerLawImage(0.5,1)
    instance.intensitySlicingImage(100, 250, 255)    