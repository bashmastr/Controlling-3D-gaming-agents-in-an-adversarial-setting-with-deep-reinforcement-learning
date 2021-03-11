from PIL import Image

class GetHealth:
    def __init__(self, image_path, saved_location_player_one, saved_location_player_two):
        self.image_path = image_path
        self.saved_location_player_one = saved_location_player_one
        self.saved_location_player_two = saved_location_player_two
        self.coordinates_player_one = (1100, 80,1850,180 )
        self.coordinates_player_two = (200, 80,950,180 )    

    def capturePlayerOneHealthBar(self):
        image_obj = Image.open(self.image_path)
        cropped_image = image_obj.crop(self.coordinates_player_one)
        cropped_image.save(self.saved_location_player_one)
        cropped_image.show()


    def capturePlayerTwoHealthBar(self):
        image_obj = Image.open(self.image_path)
        cropped_image = image_obj.crop(self.coordinates_player_two)
        cropped_image.save(self.saved_location_player_two)
        cropped_image.show()

if __name__ == '__main__':
    image_path = "tekken-7-4k.png"
    saved_location_player_one = 'cropped-player-one.png'
    saved_location_player_two = 'cropped-player-two.png'
    instance = GetHealth(image_path, saved_location_player_one, saved_location_player_two)
    instance.capturePlayerOneHealthBar()
    instance.capturePlayerTwoHealthBar()
    