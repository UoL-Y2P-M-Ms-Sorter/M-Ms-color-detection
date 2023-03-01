import random
import os
from shutil import copy


def main():
    random.seed(0)

    split_rate = 0.1

    color_class = [color for color in os.listdir("data/train")]
    if '.DS_Store' in color_class:
        color_class.remove('.DS_Store')
    for color in color_class:
        color_path = os.path.join('data/train', color)
        images = os.listdir(color_path)

        val_images = random.sample(images, int(len(images) * split_rate))
        for index, image in enumerate(images):
            if image in val_images:
                copy(os.path.join(color_path, image),
                     os.path.join('data/val', color, image))
                os.remove(os.path.join(color_path, image))


if __name__ == '__main__':
    main()
