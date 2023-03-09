"""
使用DeepLabv3+ teacher模型进行检测，保存可视化结果，并将每个城市的预测不确定写入txt
"""
import os
import os.path as osp
import time
import copy
import numpy as np
import yaml
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from core.models.model_helper import ModelBuilder
from core.utils.utils import convert_state_dict, create_cityscapes_label_colormap
from Config import Detector_para, img_path


def main():
    for city in os.listdir(img_path):
        # 不检测gta5 synthia
        if str(city) != "gta5" and str(city) != "synthia":
            scores = {}
            for i in os.listdir(img_path + city):
                # 对图片进行检测获得结果
                output = img_seg(img_path, city, i)
                # 计算预测不确定的熵
                scores = prediction_uncertainty(output, i, scores)

            # 将每个城市的预测不确定写入txt
            ent_txt(scores, city)


def img_seg(img_path, city, i):
    """
    模型推理获得可视化结果
    Args:
        img_path: 输入图片的路径
        city: 每个城市的名字
        i: 每个城市中的每张图片

    Returns: [1, 19, h, w] 的推理结果

    """
    image_path = osp.join(img_path, city, i)
    image_name = image_path.split('/')[-1]
    image = Image.open(image_path).convert('RGB')
    old_img = copy.deepcopy(image)
    image = np.asarray(image).astype(np.float32)
    h, w, _ = image.shape

    # 图片预处理
    cfg_dset = cfg["dataset"]
    mean, std = cfg_dset["mean"], cfg_dset["std"]
    image = (image - mean) / std

    image = torch.Tensor(image).permute(2, 0, 1)
    image = image.unsqueeze(dim=0)
    input_scale = [769, 769]
    # 图片下采样
    image = F.interpolate(image, input_scale, mode="bilinear", align_corners=True)

    start_time_infer = time.time()
    output = net_process(model, image)
    end_time_infer = time.time()
    t_infer = end_time_infer - start_time_infer

    # 图片上采样
    output = F.interpolate(output, (h, w), mode="bilinear", align_corners=True)

    print("图片:{} 检测用时: {}秒, FPS={}".format(i, round(t_infer, 2), round(1 / t_infer, 1)))

    # 可视化结果保存
    if result_view:
        if not osp.exists(osp.join(act_learn_out, str("result_view"), city)):
            os.makedirs(osp.join(act_learn_out, str("result_view"), city))

        # 结果后处理 渲染
        start_time_post = time.time()
        mask = torch.argmax(output, dim=1).squeeze().cpu().numpy()  # 确定每一个像素的类别对应的索引
        colormap = create_cityscapes_label_colormap()  # 类别的颜色映射
        color_mask = Image.fromarray(colorful(mask, colormap))  # 渲染结果
        # 将新图与原图及进行混合
        blend_img = Image.blend(old_img, color_mask, 0.7)  # 可保存blend混合图像
        end_time_post = time.time()
        t_post = end_time_post - start_time_post
        color_mask.save(osp.join(act_learn_out, str("result_view"), city, i))

        print("后处理用时: {}秒, FPS={}".format(round(t_post, 2), round(1 / t_post, 1)))

    return output


@torch.no_grad()
def net_process(model, image):
    """
    模型对图片进行推理
    """
    input = image.cuda()
    output = model(input)["pred"]
    return output


def prediction_uncertainty(logit, i, scores):
    """
    计算每张图片中每个像素的预测不确定性(熵），并对整个图片取平均值作为该张图片的不确定性
    Args:
        logit: 模型推理得到的结果 [1,19,h,w]
        i: 图片
        scores: 用于存放名字和对应熵的字典

    Returns: 图像的熵

    """
    logit = logit.squeeze(dim=0)  # [19, h ,w]
    p = torch.softmax(logit, dim=0)  # [19, h, w]

    pixel_entropy = torch.sum(-p * torch.log(p + 1e-6), dim=0).unsqueeze(dim=0).unsqueeze(dim=0) / math.log(19)
    pixel_entropy_arr = pixel_entropy.squeeze().cpu().numpy()
    # 计算所有像素预测熵的和并求平均值
    img_entropy = np.mean(pixel_entropy_arr)
    scores[i] = img_entropy

    return scores


def colorful(mask, colormap):
    """
    将mask的检测结果根据cityscapes的颜色映射进行转换
    """
    color_mask = np.zeros([mask.shape[0], mask.shape[1], 3])
    for i in np.unique(mask):
        color_mask[mask == i] = colormap[i]

    return np.uint8(color_mask)


def ent_txt(scores, city):
    """
    将一个城市所有图像的熵写入txt
    Args:
        scores: 熵，图像名字
        city: 城市名字

    Returns:txt

    """
    scores = zip(scores.values(), scores.keys())
    scores = sorted(scores, reverse=True)
    score_dir = osp.join(act_learn_out, str("city_uncertainty"))
    if not osp.exists(score_dir):
        os.makedirs(score_dir)
    with open(osp.join(score_dir, str(city) + ".txt"), 'w') as f1:
        for i in scores:
            line = str(i).replace("'", '')
            line = line.strip("()")
            f1.write(line + '\n')
    f1.close()


if __name__ == '__main__':
    img_path = img_path  # 需要检测图片的路径
    # 检测结果保存路径
    act_learn_out = Detector_para["act_learn_out"]
    if not osp.exists(act_learn_out):
        os.makedirs(act_learn_out)

    result_view = Detector_para["result_view"]  # 是否保存可视化结果

    # 模型参数
    config = Detector_para["config_file"]
    model_path = Detector_para["model_file"]

    # 模型初始化
    cfg = yaml.load(open(config, "r"), Loader=yaml.Loader)
    cfg["net"]["sync_bn"] = False
    model = ModelBuilder(cfg["net"])
    checkpoint = torch.load(model_path)
    # 整个网络
    # key = "teacher_state" if "teacher_state" in checkpoint.keys() else "model_state"
    # saved_state_dict = convert_state_dict(checkpoint[key])
    # 只包含teacher
    saved_state_dict = convert_state_dict(checkpoint)

    model.load_state_dict(saved_state_dict, strict=False)
    model.cuda()
    model.eval()

    main()
