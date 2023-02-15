"""
根据每个城市的熵txt，以及每个城市需要选择的数量，生成第二次需要标注的txt
"""
import os
import os.path as osp
import random
from Config import *


def main():
    # 根据各城市包含图像数量的比例，计算每个城市第二次需要标注的数量(除去第一次已经标注的数量）
    select_nums = every_city_num()

    # 第一次标注中每个城市包含的图像名称
    first_dict = get_first_dict()

    # 第二次标注中每个城市包含的图像名称
    second_dict = get_second_dict(select_nums, first_dict)

    # 写入txt
    second_labeled_list = []
    for c in second_dict:
        for i in second_dict[c]:
            second_labeled_list.append(i)

    random.shuffle(second_labeled_list)

    with open(second_labeled_txt, "w") as f:
        for i in second_labeled_list:
            line = "leftImg8bit/train/" + i.split("_")[0] + "/" + i + "\n"
            f.write(line)
    f.close()


def every_city_num():
    """
    根据各城市包含图像数量的比例，计算每个城市第二次需要标注的数量(除去第一次已经标注的数量）

    Returns: 字典 城市名字:标注数量

    """
    city_all = {"aachen": 174, "bochum": 96, "bremen": 316, "cologne": 154, "darmstadt": 85, "dusseldorf": 221,
                "erfurt": 109, "hamburg": 248, "hanover": 196, "jena": 119, "krefeld": 99, "monchengladbach": 94,
                "strasbourg": 365, "stuttgart": 196, "tubingen": 144, "ulm": 95, "weimar": 142, "zurich": 122}

    select_nums = {}
    for i in city_all:
        city_name = i
        img_num = city_all[city_name]  # 该城市包含的图像数量
        # 减去first已经选择的数量
        first_num = first_labeled_num(i)

        select_num = round(sec_num * (img_num - first_num) / sum(city_all.values()))  # 每个城市需要选择的数量
        all_nums = sum(select_nums.values())

        # 根据最终的挑选数量调整最后一个城市的数量
        if i == "zurich":
            if all_nums + select_num > sec_num:
                select_num = sec_num - all_nums
            elif all_nums + select_num < sec_num:
                select_num = sec_num - all_nums

        select_nums[i] = select_num

    assert sec_num == sum(select_nums.values()), "数量不匹配"
    print("选择的数量", sum(select_nums.values()))

    return select_nums


def first_labeled_num(i):
    """
    第一次标注中该城市包含的图像数量
    Args:
        i: 城市

    Returns: 第一次标注中该城市包含的图像数量

    """
    first_num = 0
    file = open(first_labeled_txt, encoding="UTF-8", mode="r")
    for line in file.readlines():
        city_name = line.split("/")[2]
        if city_name == i:
            first_num += 1

    return first_num


def get_first_dict():
    """
    第一次标注中 每个城市包含的图像名称
    Returns:

    """
    first_dict = {}
    fist_file = open(first_labeled_txt, encoding="UTF-8", mode='r')
    for line in fist_file.readlines():
        text = line.strip('\n')
        city_name = text.split("/")[2]
        img_name = text.split("/")[-1]
        if city_name in first_dict.keys():
            first_dict[city_name].append(img_name)
        else:
            first_dict[city_name] = []
            first_dict[city_name].append(img_name)
    print("first_dict", first_dict)

    return first_dict


def get_second_dict(select_nums, first_dict):
    """
    获得第二次需要标注的信息
    Args:
        select_nums: 每个城市第二次应该选择的数量
        first_dict: 每个城市第一次标注包含的图像名称

    Returns: 第二次标注中每个城市包含的图像名称

    """
    second_dict = {}

    for i in select_nums:
        name = i
        num = select_nums[name]
        # 读取该城市每张图像的熵
        city_entropy_file = open(osp.join(city_uncertainty_dir, str(name) + ".txt"), encoding="UTF-8", mode='r')

        n = 1
        for line in city_entropy_file.readlines():
            text = line.strip('\n')
            img_name = text.split(", ")[1]
            if n <= num:
                if i in second_dict.keys():
                    if i in first_dict and img_name in first_dict[i]:
                        continue
                    second_dict[i].append(img_name)
                else:
                    second_dict[i] = []
                    if i in first_dict and img_name in first_dict[i]:
                        continue
                    second_dict[i].append(img_name)
                n += 1

    return second_dict


if __name__ == '__main__':
    # 第二次需要标注的数量
    sec_num = 120

    # cs_txt 2975, first_labeled_txt 第一次标注的信息，second_labeled_txt 第二次需要标注的信息, city_uncertainty_dir 熵文件夹
    cs_txt = "/data1/glc/STAL-master/data/cs/cs_all_list.txt"
    first_labeled_txt = "/data1/glc/STAL-master/data/gtav2cityscapes/2.2%/labeled_cs30.txt"  # 30
    second_labeled_txt = "/data1/glc/STAL-master/data/gtav2cityscapes/2.2%/labeled_cs35.txt"
    city_uncertainty_dir = "/data1/glc/STAL-master/experiments/gtav2cityscapes/1.0%/checkpoints/act_learn_out/city_uncertainty/"

    main()
