"""
Change the img name of gta5 to the format of cityscapes
"""
import os.path
from shutil import copyfile


def rename(old, new):
    filelist = os.listdir(old)
    filelist.sort()
    for file in filelist:
        Olddir = os.path.join(old, file)
        if os.path.isdir(Olddir):
            continue
        filename = os.path.splitext(file)[0]
        # rename_jpg = "gta5_" + filename + "_000019_leftImg8bit.png"
        rename_jpg = "gta5_" + filename + "_000019_gtFine_labelTrainIds.png"
        copyfile(old + file, new + rename_jpg)


if __name__ == '__main__':
    img_path = ''
    new_img_path = ''
    if not os.path.exists(new_img_path):
        os.makedirs(new_img_path)

    rename(img_path, new_img_path)
