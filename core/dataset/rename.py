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

        if os.path.basename(new_img_path) == "gta5":
            # gta5(images)
            if os.path.basename(img_path) == "images":
                rename_jpg = "gta5_" + filename + "_000019_leftImg8bit.png"

            # gta5(labels)
            elif os.path.basename(img_path) == "labels":
                rename_jpg = "gta5_" + filename + "_000019_gtFine_labelTrainIds.png"

            else:
                print("please check the path!")

        else:
            # synthia(RGB)
            if os.path.basename(img_path) == "RGB":
                rename_jpg = "synthia_" + filename + "_000019_leftImg8bit.png"

            # synthia(LABELS)
            elif os.path.basename(img_path) == "LABELS":
                rename_jpg = "synthia_" + filename + "_000019_gtFine_labelTrainIds.png"

            else:
                print("please check the path!")

        copyfile(old + "/" + file, new + "/" + rename_jpg)


if __name__ == '__main__':
    # gta5(images)
    # img_path = '/data1/glc/ILM-ASSL-master/data/dataset/gtav/images'
    # new_img_path = '/data1/glc/ILM-ASSL-master/data/dataset/cityscapes/leftImg8bit/train/gta5'

    # gta5(labels)
    # img_path = '/data1/glc/ILM-ASSL-master/data/dataset/gtav/labels'
    # new_img_path = '/data1/glc/ILM-ASSL-master/data/dataset/cityscapes/gtFine/train/gta5'

    # synthia(RGB)
    # img_path = '/data1/glc/ILM-ASSL-master/data/dataset/synthia/RGB'
    # new_img_path = '/data1/glc/ILM-ASSL-master/data/dataset/cityscapes/leftImg8bit/train/synthia'

    # synthia(LABELS)
    img_path = '/data1/glc/ILM-ASSL-master/data/dataset/synthia/LABELS'
    new_img_path = '/data1/glc/ILM-ASSL-master/data/dataset/cityscapes/gtFine/train/synthia'

    if not os.path.exists(new_img_path):
        os.makedirs(new_img_path)

    rename(img_path, new_img_path)
