from PIL import Image
from imageOperation import ImageOperation 

class GetHealth:
    def __init__(self, image_path, saved_location_player_one, saved_location_player_two):
        self.image_path = image_path
        self.saved_location_player_one = saved_location_player_one
        self.saved_location_player_two = saved_location_player_two
        self.coordinates_player_two = (1700, 80, 2800,150 )
        self.coordinates_player_one = (250, 80,1350,150 )    

    def capturePlayerOneHealthBar(self):
        image_obj = Image.open(self.image_path)
        cropped_image = image_obj.crop(self.coordinates_player_one)
        image_operation_instance = ImageOperation(cropped_image)
        count = image_operation_instance.thresholdingImage(100)
        return count
        # cropped_image.save(self.saved_location_player_one)
        # cropped_image.show()


    def capturePlayerTwoHealthBar(self):
        image_obj = Image.open(self.image_path)
        cropped_image = image_obj.crop(self.coordinates_player_two)
        image_operation_instance = ImageOperation(cropped_image)
        count = image_operation_instance.thresholdingImage(100)
        return count
        # cropped_image.save(self.saved_location_player_two)
        # cropped_image.show()

if __name__ == '__main__':
    for i in range(20):
        print("#############"+str(i+1))
        image_path = "tekken-"+str(i+1)+".png"
        saved_location_player_one = 'cropped-player-one.png'
        saved_location_player_two = 'cropped-player-two.png'
        instance = GetHealth(image_path, saved_location_player_one, saved_location_player_two)
        count1 = instance.capturePlayerOneHealthBar()
        count2 = instance.capturePlayerTwoHealthBar()
        if i == 0:
            max1 = count1
            max2 = count2

        health_player_one = round((count1 / max1 ) * 100)
        health_player_two = round((count2 / max2) * 100)
        if health_player_two > 100:
            health_player_two = 100
        if health_player_one > 100:
            health_player_one = 100
        print("health_player_one = ", health_player_one)
        print("health_player_two = ", health_player_two)