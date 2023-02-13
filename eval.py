import logging
import os
import time
from argparse import ArgumentParser
import imageio
import cv2
import numpy as np
import torch
import torch.nn.functional as F
import torch.nn.parallel
import torch.optim
import torch.utils.data
import yaml
from PIL import Image

from stal.models.model_helper import ModelBuilder
from stal.utils.utils import (
    AverageMeter,
    check_makedirs,
    colorize,
    convert_state_dict,
    create_cityscapes_label_colormap,
    create_pascal_label_colormap,
    intersectionAndUnion,
    get_palette,
)


# Setup Parser
def get_parser():
    parser = ArgumentParser(description="PyTorch Evaluation")
    parser.add_argument(
        "--base_size", type=int, default=2048, help="based size for scaling"
    )
    parser.add_argument(
        "--scales", type=float, default=[1.0], nargs="+", help="evaluation scales"
    )
    parser.add_argument("--config", type=str, default="config.yaml")
    parser.add_argument(
        "--model_path",
        type=str,
        default="checkpoints/psp_best.pth",
        help="evaluation model path",
    )
    parser.add_argument(
        "--save_folder",
        type=str,
        default="checkpoints/results/",
        help="results save folder",
    )
    parser.add_argument(
        "--names_path",
        type=str,
        default="../../vis_meta/cityscapes/cityscapesnames.mat",
        help="path of dataset category names",
    )
    parser.add_argument(
        "--crop", action="store_true", default=False, help="whether use crop evaluation"
    )
    parser.add_argument(
        "--save", action="store_true", help="whether to save the results"
    )
    return parser


def get_logger():
    logger_name = "main-logger"
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.INFO)
    handler = logging.StreamHandler()
    fmt = "[%(asctime)s %(levelname)s %(filename)s line %(lineno)d %(process)d] %(message)s"
    handler.setFormatter(logging.Formatter(fmt))
    logger.addHandler(handler)
    return logger


def main():
    global args, logger, cfg, colormap
    args = get_parser().parse_args()
    cfg = yaml.load(open(args.config, "r"), Loader=yaml.Loader)
    logger = get_logger()
    logger.info(args)

    cfg_dset = cfg["dataset"]
    mean, std = cfg_dset["mean"], cfg_dset["std"]
    num_classes = cfg["net"]["num_classes"]
    crop_size = cfg_dset["val"]["crop"]["size"]
    crop_h, crop_w = crop_size

    assert num_classes > 1

    if cfg["dataset"]["val"]["data_list"].split("/")[-1].split(".")[0] == "val":
        gray_folder = os.path.join(args.save_folder, "gray")
        color_folder = os.path.join(args.save_folder, "color")

    elif cfg["dataset"]["val"]["data_list"].split("/")[-1].split(".")[0] == "val_gtav":
        gray_folder = os.path.join(args.save_folder, "gray_gtav")
        color_folder = os.path.join(args.save_folder, "color_gtav")

    elif cfg["dataset"]["val"]["data_list"].split("/")[-1].split(".")[0] == "val_synthia":
        gray_folder = os.path.join(args.save_folder, "gray_synthia")
        color_folder = os.path.join(args.save_folder, "color_synthia")

    os.makedirs(gray_folder, exist_ok=True)
    os.makedirs(color_folder, exist_ok=True)

    cfg_dset = cfg["dataset"]
    data_root, f_data_list = cfg_dset["val"]["data_root"], cfg_dset["val"]["data_list"]
    data_list = []

    if "cityscapes" in data_root:
        colormap = create_cityscapes_label_colormap()
        for line in open(f_data_list, "r"):
            arr = [
                line.strip(),
                "gtFine/" + line.strip()[12:-15] + "gtFine_labelTrainIds.png",
            ]
            arr = [os.path.join(data_root, item) for item in arr]
            data_list.append(arr)

    else:
        colormap = create_pascal_label_colormap()
        for line in open(f_data_list, "r"):
            arr = [
                "JPEGImages/{}.jpg".format(line.strip()),
                "SegmentationClassAug/{}.png".format(line.strip()),
            ]
            arr = [os.path.join(data_root, item) for item in arr]
            data_list.append(arr)

    # Create network.
    args.use_auxloss = True if cfg["net"].get("aux_loss", False) else False

    cfg["net"]["sync_bn"] = False
    model = ModelBuilder(cfg["net"])
    # for model_path in ["checkpoints/ckpt_best.pth", "checkpoints/ckpt.pth"]:
    for model_path in ["checkpoints/ckpt_best.pth"]:
        logger.info("=> creating model from '{}' ...".format(model_path))

        for key in ["model_state", "teacher_state"]:
            checkpoint = torch.load(model_path)
            if not key in checkpoint.keys():
                continue

            logger.info(f"=> load checkpoint[{key}]")

            saved_state_dict = convert_state_dict(checkpoint[key])
            model.load_state_dict(saved_state_dict, strict=False)
            model.cuda()
            logger.info("Load Model Done!")
            if "cityscapes" in cfg["dataset"]["type"]:
                validate_city(
                    model,
                    num_classes,
                    data_list,
                    mean,
                    std,
                    args.base_size,
                    crop_h,
                    crop_w,
                    args.scales,
                    gray_folder,
                    color_folder,
                )
            else:
                valiadte_whole(
                    model,
                    num_classes,
                    data_list,
                    mean,
                    std,
                    args.scales,
                    gray_folder,
                    color_folder,
                )
    # cal_acc(data_list, gray_folder, num_classes)


@torch.no_grad()
def net_process(model, image):
    b, c, h, w = image.shape
    # num_classes = cfg['net']['num_classes']
    # output_all = torch.zeros((6, b, num_classes, h, w)).cuda()
    input = image.cuda()
    output = model(input)["pred"]
    output = F.interpolate(output, (h, w), mode="bilinear", align_corners=True)
    # output_all[0] = F.softmax(output, dim=1)
    #
    # output = model(torch.flip(input, [3]))["pred"]
    # output = F.interpolate(output, (h, w), mode="bilinear", align_corners=True)
    # output = F.softmax(output, dim=1)
    # output_all[1] = torch.flip(output, [3])
    #
    # scales = [(961, 961), (841, 841), (721, 721), (641, 641)]
    # for k, scale in enumerate(scales):
    #     input_scale = F.interpolate(input, scale, mode="bilinear", align_corners=True)
    #     output = model(input_scale)["pred"]
    #     output = F.interpolate(output, (h, w), mode="bilinear", align_corners=True)
    #     output_all[k + 2] = F.softmax(output, dim=1)
    #
    # output = torch.mean(output_all, dim=0)
    return output


def scale_crop_process(model, image, classes, crop_h, crop_w, h, w, stride_rate=2 / 3):
    ori_h, ori_w = image.size()[-2:]
    pad_h = max(crop_h - ori_h, 0)
    pad_w = max(crop_w - ori_w, 0)
    pad_h_half = int(pad_h / 2)
    pad_w_half = int(pad_w / 2)
    if pad_h > 0 or pad_w > 0:
        border = (pad_w_half, pad_w - pad_w_half, pad_h_half, pad_h - pad_h_half)
        image = F.pad(image, border, mode="constant", value=0.0)
    new_h, new_w = image.size()[-2:]
    stride_h = int(np.ceil(crop_h * stride_rate))
    stride_w = int(np.ceil(crop_w * stride_rate))
    grid_h = int(np.ceil(float(new_h - crop_h) / stride_h) + 1)
    grid_w = int(np.ceil(float(new_w - crop_w) / stride_w) + 1)
    prediction_crop = torch.zeros((1, classes, new_h, new_w), dtype=torch.float).cuda()
    count_crop = torch.zeros((new_h, new_w), dtype=torch.float).cuda()
    for index_h in range(0, grid_h):
        for index_w in range(0, grid_w):
            s_h = index_h * stride_h
            e_h = min(s_h + crop_h, new_h)
            s_h = e_h - crop_h
            s_w = index_w * stride_w
            e_w = min(s_w + crop_w, new_w)
            s_w = e_w - crop_w
            image_crop = image[:, :, s_h:e_h, s_w:e_w].contiguous()
            count_crop[s_h:e_h, s_w:e_w] += 1

            with torch.no_grad():
                prediction_crop[:, :, s_h:e_h, s_w:e_w] += net_process(
                    model, image_crop
                )

    prediction_crop /= count_crop
    prediction_crop = prediction_crop[
                      :, :, pad_h_half: pad_h_half + ori_h, pad_w_half: pad_w_half + ori_w
                      ]
    prediction = F.interpolate(
        prediction_crop, size=(h, w), mode="bilinear", align_corners=True
    )
    return prediction[0]


def scale_whole_process(model, image, h, w):
    with torch.no_grad():
        prediction = net_process(model, image)
    prediction = F.interpolate(
        prediction, size=(h, w), mode="bilinear", align_corners=True
    )
    return prediction[0]


def validate_city(
        model,
        classes,
        data_list,
        mean,
        std,
        base_size,
        crop_h,
        crop_w,
        scales,
        gray_folder,
        color_folder,
):
    global colormap
    logger.info(">>>>>>>>>>>>>>>> Start Crop Evaluation >>>>>>>>>>>>>>>>")
    data_time = AverageMeter()
    batch_time = AverageMeter()
    intersection_meter = AverageMeter()
    union_meter = AverageMeter()

    model.eval()
    end = time.time()
    for i, (input_pth, label_path) in enumerate(data_list):
        data_time.update(time.time() - end)

        # image = Image.open(input_pth).convert("RGB")
        # image = np.asarray(image).astype(np.float32)
        # label = Image.open(label_path).convert("L")
        # label = np.asarray(label).astype(np.uint8)

        if str(label_path.split('/')[-1]).split('_')[0] != "gta5" and str(label_path.split('/')[-1]).split('_')[
            0] != "synthia":
            image = Image.open(input_pth).convert("RGB")
            image = np.asarray(image).astype(np.float32)
            label = Image.open(label_path).convert("L")
            label = np.asarray(label).astype(np.uint8)

        elif str(label_path.split('/')[-1]).split('_')[0] == "gta5":
            image = Image.open(input_pth).convert("RGB")
            image = image.resize((2048, 1024), Image.ANTIALIAS)
            image = np.asarray(image).astype(np.float32)  # 1024 2048 3

            label = Image.open(label_path)
            label = label.resize((2048, 1024), Image.ANTIALIAS)
            label = np.asarray(label).astype(np.uint8)  # 1024 2048
            # 重新分配标签以匹配 Cityscapes 的格式
            label_copy = 255 * np.ones(label.shape, dtype=np.uint8)  # 1024 2048
            id_to_trainid = {7: 0, 8: 1, 11: 2, 12: 3, 13: 4, 17: 5, 19: 6, 20: 7, 21: 8, 22: 9, 23: 10, 24: 11, 25: 12,
                             26: 13, 27: 14, 28: 15, 31: 16, 32: 17, 33: 18}
            for k, v in id_to_trainid.items():
                label_copy[label == k] = v
            label = label_copy  # 1024 2048

        elif str(label_path.split('/')[-1]).split('_')[0] == "synthia":
            image = Image.open(input_pth).convert("RGB")
            image = image.resize((2048, 1024), Image.ANTIALIAS)
            image = np.asarray(image).astype(np.float32)  # 1024 2048 3

            label = imageio.imread(label_path, format='PNG-FI')
            label = cv2.resize(label, (2048, 1024))
            label = np.asarray(label)[:, :, 0]  # uint16  1024 2048
            # 重新分配标签以匹配 Cityscapes 的格式
            label_copy = 255 * np.ones(label.shape, dtype=np.float32)
            id_to_trainid_synthia = {3: 0, 4: 1, 2: 2, 21: 3, 5: 4, 7: 5, 15: 6, 9: 7, 6: 8, 1: 9, 10: 10, 17: 11,
                                     8: 12, 19: 13, 12: 14, 11: 15}
            for k, v in id_to_trainid_synthia.items():
                label_copy[label == k] = v
            label = label_copy

        image = (image - mean) / std
        image = torch.Tensor(image).permute(2, 0, 1)
        image = image.contiguous().unsqueeze(dim=0)
        h, w = image.size()[-2:]
        prediction = torch.zeros((classes, h, w), dtype=torch.float).cuda()
        for scale in scales:
            long_size = round(scale * base_size)
            new_h = long_size
            new_w = long_size
            if h > w:
                new_w = round(long_size / float(h) * w)
            else:
                new_h = round(long_size / float(w) * h)
            image_scale = F.interpolate(
                image, size=(new_h, new_w), mode="bilinear", align_corners=True
            )
            prediction += scale_crop_process(
                model, image_scale, classes, crop_h, crop_w, h, w
            )
        prediction = torch.max(prediction, dim=0)[1].cpu().numpy()
        batch_time.update(time.time() - end)
        end = time.time()
        if (i + 1) % 10 == 0:
            logger.info(
                "Test: [{}/{}] "
                "Data {data_time.val:.3f} ({data_time.avg:.3f}) "
                "Batch {batch_time.val:.3f} ({batch_time.avg:.3f}).".format(
                    i + 1, len(data_list), data_time=data_time, batch_time=batch_time
                )
            )
        gray = np.uint8(prediction)  # 1024 2048
        # synthia
        if str(label_path.split('/')[-1]).split('_')[0] == "synthia":
            pred = transform_color(gray)
            color = get_color_pallete(pred, "city")  # 2048 10124
            if color.mode == 'P':
                color = color.convert('RGB')
        # cs or gtav
        else:
            color = colorize(gray, colormap)

        # if args.save:
        image_path, _ = data_list[i]
        image_name = image_path.split("/")[-1].split(".")[0]
        color_path = os.path.join(color_folder, image_name + ".png")
        color.save(color_path)

        gray_path = os.path.join(gray_folder, image_name + ".png")
        cv2.imwrite(gray_path, gray)

        intersection, union, target = intersectionAndUnion(gray, label, classes)
        intersection_meter.update(intersection)
        union_meter.update(union)

    iou_class = intersection_meter.sum / (union_meter.sum + 1e-10)
    for i, iou in enumerate(iou_class):
        logger.info(" * class [{}] IoU {:.2f}".format(i, iou * 100))
    logger.info(" * mIoU {:.2f}".format(np.mean(iou_class) * 100))
    logger.info("<<<<<<<<<<<<<<<<< End Crop Evaluation <<<<<<<<<<<<<<<<<")


def valiadte_whole(
        model, classes, data_list, mean, std, scales, gray_folder, color_folder
):
    logger.info(">>>>>>>>>>>>>>>> Start Evaluation >>>>>>>>>>>>>>>>")
    data_time = AverageMeter()
    batch_time = AverageMeter()
    model.eval()
    end = time.time()
    for i, (input_pth, _) in enumerate(data_list):
        data_time.update(time.time() - end)
        image = Image.open(input_pth).convert("RGB")
        image = np.asarray(image).astype(np.float32)
        image = (image - mean) / std
        image = torch.Tensor(image).permute(2, 0, 1)
        image = image.contiguous().unsqueeze(dim=0)
        h, w = image.size()[-2:]
        prediction = torch.zeros((classes, h, w), dtype=torch.float).cuda()
        for scale in scales:
            new_h = round(h * scale)
            new_w = round(w * scale)
            image_scale = F.interpolate(
                image, size=(new_h, new_w), mode="bilinear", align_corners=True
            )
            prediction += scale_whole_process(model, image_scale, h, w)
        prediction = (
            torch.max(prediction, dim=0)[1].cpu().numpy()
        )  ##############attention###############
        batch_time.update(time.time() - end)
        end = time.time()
        if (i + 1) % 10 == 0:
            logger.info(
                "Test: [{}/{}] "
                "Data {data_time.val:.3f} ({data_time.avg:.3f}) "
                "Batch {batch_time.val:.3f} ({batch_time.avg:.3f}).".format(
                    i + 1, len(data_list), data_time=data_time, batch_time=batch_time
                )
            )
        check_makedirs(gray_folder)
        check_makedirs(color_folder)
        gray = np.uint8(prediction)
        color = colorize(gray)
        image_path, _ = data_list[i]
        image_name = image_path.split("/")[-1].split(".")[0]
        gray_path = os.path.join(gray_folder, image_name + ".png")
        color_path = os.path.join(color_folder, image_name + ".png")
        gray = Image.fromarray(gray)
        gray.save(gray_path)
        color.save(color_path)
    logger.info("<<<<<<<<<<<<<<<<< End  Evaluation <<<<<<<<<<<<<<<<<")


def transform_color(pred):
    synthia_to_city = {
        0: 0,
        1: 1,
        2: 2,
        3: 3,
        4: 4,
        5: 5,
        6: 6,
        7: 7,
        8: 8,
        9: 10,
        10: 11,
        11: 12,
        12: 13,
        13: 15,
        14: 17,
        15: 18,
    }
    label_copy = 255 * np.ones(pred.shape, dtype=np.float32)
    for k, v in synthia_to_city.items():
        label_copy[pred == k] = v
    return label_copy.copy()


def get_color_pallete(npimg, dataset='voc'):
    out_img = Image.fromarray(npimg.astype('uint8')).convert('P')
    if dataset == 'city':
        cityspallete = [
            128, 64, 128,
            244, 35, 232,
            70, 70, 70,
            102, 102, 156,
            190, 153, 153,
            153, 153, 153,
            250, 170, 30,
            220, 220, 0,
            107, 142, 35,
            152, 251, 152,
            0, 130, 180,
            220, 20, 60,
            255, 0, 0,
            0, 0, 142,
            0, 0, 70,
            0, 60, 100,
            0, 80, 100,
            0, 0, 230,
            119, 11, 32,
        ]
        out_img.putpalette(cityspallete)
    else:
        vocpallete = get_palette(256)
        out_img.putpalette(vocpallete)
    return out_img


if __name__ == "__main__":
    main()
