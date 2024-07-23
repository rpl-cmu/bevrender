from PIL import Image
from torchvision import transforms


class MapLoader:
    def __init__(self, config, logger):
        self.pil_to_tensor = transforms.PILToTensor()
        self.map_dir = config["MAP_PATH"][config["MAP_MONTH"]]

    def get_map_tensor(self, normalize_or_not):
        if normalize_or_not:
            map_tensor = self.pil_to_tensor(Image.open(self.map_dir)) / 255.0
        else:
            map_tensor = self.pil_to_tensor(Image.open(self.map_dir))
        return map_tensor

    def get_map_img(self):
        return Image.open(self.map_dir)
