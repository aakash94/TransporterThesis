import os
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from PIL import Image
import cv2
import random
import math
import glob


class ImageLoader(Dataset):

    def get_dir_list(self, root_path):
        subfolders = [f.path for f in os.scandir(root_path) if f.is_dir()]
        return subfolders

    def get_image_list(self, root_path):
        image_list = []
        png_path = os.path.join(root_path, "*.png")
        images_path = glob.glob(png_path)
        for i in images_path:
            img = cv2.imread(i)
            image_list.append(img)
        return image_list

    def get_image_tuple(self, image_list):
        image_count = len(image_list)
        i_tuples = []
        for i in range(image_count - self.t_delta):
            i1 = image_list[i]
            i2 = image_list[i + self.t_delta]
            i1 = cv2.cvtColor(i1, cv2.COLOR_BGR2RGB)
            i2 = cv2.cvtColor(i2, cv2.COLOR_BGR2RGB)
            tup = (i1, i2)
            i_tuples.append(tup)

    def __init__(self, env_name, t_delta=10):
        self.dump_path = os.path.join(".", "frames", "dump", env_name)
        self.t_delta = t_delta
        dirs = self.get_dir_list(root_path=self.dump_path)
        self.image_tuples = []
        for d in dirs:
            image_list = self.get_image_list(root_path=d)
            image_tuple = self.get_image_tuple(image_list=image_list)
            self.image_tuples.append(image_tuple)

    def __getitem__(self, index):
        tup = self.image_tuples[index]
        i, i_delta = tup
        return i, i_delta

    def __len__(self):
        return len(self.image_tuples)


def main():
    env_name = "CarRacing-v1"
    dump_path = os.path.join(".", "frames", "dump", env_name)
    subfolders = [f.path for f in os.scandir(dump_path) if f.is_dir()]
    for s in subfolders:
        png_path = os.path.join(s, "*.png")
        images_path = glob.glob(png_path)
        print("___")
        for i in images_path:
            img = Image.open(i)
            print(type(img))
            img = cv2.imread(i)
            print(type(img))


if __name__ == "__main__":
    main()
