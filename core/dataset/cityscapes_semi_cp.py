import copy
import os
import os.path
import random

import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from . import augmentation as psp_trsform
from .base import BaseDataset
from PIL import Image
import math
import imageio
import cv2


class city_dset(BaseDataset):
    def __init__(
            self,
            data_root,
            data_list,
            trs_form,
            seed,
            n_sup,
            split="val",
            unsup=False,
            coarse=False,
            coarse_num=3000,
            fm=False,
            acp=False,
            paste_trs=None,
            prob=0.5,
            acm=False,
    ):
        super(city_dset, self).__init__(data_list)
        self.data_root = data_root
        self.transform = trs_form
        self.paste_trs = paste_trs
        self.fm = fm
        self.acp = acp and split == "train"
        self.prob = prob
        self.acm = acm
        # gtav
        self.id_to_trainid = {7: 0,
                              8: 1,
                              11: 2,
                              12: 3,
                              13: 4,
                              17: 5,
                              19: 6,
                              20: 7,
                              21: 8,
                              22: 9,
                              23: 10,
                              24: 11,
                              25: 12,
                              26: 13,
                              27: 14,
                              28: 15,
                              31: 16,
                              32: 17,
                              33: 18}
        # synthia
        self.id_to_trainid_synthia = {3: 0,
                                      4: 1,
                                      2: 2,
                                      21: 3,
                                      5: 4,
                                      7: 5,
                                      15: 6,
                                      9: 7,
                                      6: 8,
                                      1: 9,
                                      10: 10,
                                      17: 11,
                                      8: 12,
                                      19: 13,
                                      12: 14,
                                      11: 15}

        random.seed(seed)
        # print("n_sup", n_sup)
        if len(self.list_sample) >= n_sup and split == "train":  # train
            if unsup and not coarse:  # unlabeled
                self.list_sample_new = random.sample(self.list_sample, n_sup)
                # transform to tuple
                for i in range(len(self.list_sample)):
                    self.list_sample[i] = tuple(self.list_sample[i])
                for i in range(len(self.list_sample_new)):
                    self.list_sample_new[i] = tuple(self.list_sample_new[i])
                print("unlabeled: ", len(self.list_sample_new))

            elif unsup and coarse:
                try:
                    coarse_list = data_list.replace("fine_train", "coarse_train")
                    self.list_sample_new = [
                        line.strip().split(" ") for line in open(coarse_list, "r")
                    ]
                except:
                    coarse_list = data_list.replace("fine_trainval", "coarse_train")
                    self.list_sample_new = [
                        line.strip().split(" ") for line in open(coarse_list, "r")
                    ]
                random.seed(seed)
                if coarse_num < len(self.list_sample_new):
                    self.list_sample_new = random.sample(
                        self.list_sample_new, coarse_num
                    )
                else:
                    random.shuffle(self.list_sample_new)
            else:  # label
                list_sample_gtav = []
                list_sample_cs = []

                for i in self.list_sample:
                    city_name = i[0].split('/')[-1].split('_')[0]
                    if city_name != "gta5" and city_name != "synthia":
                        list_sample_cs.append(i)
                    else:
                        list_sample_gtav.append(i)

                unlabel_num = 2975 - len(list_sample_cs)
                num_repeat = math.ceil(unlabel_num / len(list_sample_cs))
                list_sample_cs_new = list_sample_cs * num_repeat
                list_sample_cs_new = random.sample(list_sample_cs_new, unlabel_num)

                self.list_sample_new = list_sample_cs_new + list_sample_gtav
                self.list_sample_new = random.sample(self.list_sample_new, unlabel_num + len(list_sample_gtav))
                print("labeled: ", len(self.list_sample_new))
        else:
            self.list_sample_new = self.list_sample  # val
            print("val: ", len(self.list_sample_new))

    def __getitem__(self, index):
        # load image and its label
        image_path = os.path.join(self.data_root, self.list_sample_new[index][0])
        label_path = os.path.join(self.data_root, self.list_sample_new[index][1])
        # image = self.img_loader(image_path, "RGB")
        # label = self.img_loader(label_path, "L")

        if str(label_path.split('/')[-1]).split('_')[0] != "gta5" and str(label_path.split('/')[-1]).split('_')[
            0] != "synthia":
            image = self.img_loader(image_path, "RGB")
            label = self.img_loader(label_path, "L")

        elif str(label_path.split('/')[-1]).split('_')[0] == "gta5":
            image = self.img_loader(image_path, "RGB")
            image = image.resize((2048, 1024), Image.ANTIALIAS)
            # open label
            label = Image.open(label_path)
            label = label.resize((2048, 1024), Image.ANTIALIAS)
            label = np.array(label, dtype=np.uint8)

            # 重新分配标签以匹配 Cityscapes 的格式
            label_copy = 255 * np.ones(label.shape, dtype=np.uint8)
            for k, v in self.id_to_trainid.items():
                label_copy[label == k] = v

            label = Image.fromarray(label_copy)

        elif str(label_path.split('/')[-1]).split('_')[0] == "synthia":
            image = self.img_loader(image_path, "RGB")
            image = image.resize((2048, 1024), Image.ANTIALIAS)
            # open label
            label = imageio.imread(label_path, format='PNG-FI')
            label = cv2.resize(label, (2048, 1024))

            label = np.asarray(label)[:, :, 0]  # uint16

            # 重新分配标签以匹配 Cityscapes 的格式
            label_copy = 255 * np.ones(label.shape, dtype=np.float32)
            for k, v in self.id_to_trainid_synthia.items():
                label_copy[label == k] = v
            label = Image.fromarray(label_copy)

        # loader paste img and mask
        if self.acp:
            if random.random() > self.prob:
                paste_idx = random.randint(0, self.__len__() - 1)
                paste_img_path = os.path.join(
                    self.data_root, self.list_sample_new[paste_idx][0]
                )
                paste_img = self.img_loader(paste_img_path, "RGB")
                paste_label_path = os.path.join(
                    self.data_root, self.list_sample_new[paste_idx][1]
                )
                paste_label = self.img_loader(paste_label_path, "L")
                paste_img, paste_label = self.paste_trs(paste_img, paste_label)
            else:
                paste_img, paste_label = None, None

        if self.fm:
            inputs = self.transform(image, label)
            if len(inputs) == 5:
                image_weak, label_weak, image_strong, label_strong, valid = inputs
                return (
                    image_weak[0],
                    label_weak[0, 0].long(),
                    image_strong[0],
                    label_strong[0, 0].long(),
                    valid[0, 0].long(),
                )
            else:
                image, label, valid = inputs
                return image[0], label[0, 0].long(), valid[0, 0].long()

        elif self.acm:
            image, label = self.transform(image, label)
            return image[0], label[0, 0].long(), index
        else:
            image, label = self.transform(image, label)

        if self.acp:
            if paste_img is not None:
                return torch.cat((image[0], paste_img[0]), dim=0), torch.cat(
                    [label[0, 0].long(), paste_label[0, 0].long()], dim=0
                )
            else:
                h, w = image[0].shape[1], image[0].shape[2]
                paste_img = torch.zeros(3, h, w)
                paste_label = torch.zeros(h, w)
                return torch.cat((image[0], paste_img), dim=0), torch.cat(
                    [label[0, 0].long(), paste_label.long()], dim=0
                )

        return image[0], label[0, 0].long()

    def __len__(self):
        return len(self.list_sample_new)


def build_transfrom(cfg, fm=False, acp=False):
    trs_form = []
    mean, std, ignore_label = cfg["mean"], cfg["std"], cfg["ignore_label"]
    trs_form.append(psp_trsform.ToTensor())
    trs_form.append(psp_trsform.Normalize(mean=mean, std=std))
    if cfg.get("resize", False):
        trs_form.append(psp_trsform.Resize(cfg["resize"]))
    if cfg.get("rand_resize", False):
        if not acp:
            trs_form.append(psp_trsform.RandResize(cfg["rand_resize"]))
        else:
            trs_form.append(psp_trsform.RandResize(cfg["acp"]["rand_resize"]))
    if cfg.get("rand_rotation", False):
        rand_rotation = cfg["rand_rotation"]
        trs_form.append(
            psp_trsform.RandRotate(rand_rotation, ignore_label=ignore_label)
        )
    if cfg.get("GaussianBlur", False) and cfg["GaussianBlur"]:
        trs_form.append(psp_trsform.RandomGaussianBlur())
    if cfg.get("flip", False) and cfg.get("flip"):
        trs_form.append(psp_trsform.RandomHorizontalFlip())
    if cfg.get("crop", False):
        crop_size, crop_type = cfg["crop"]["size"], cfg["crop"]["type"]
        trs_form.append(
            psp_trsform.Crop(crop_size, crop_type=crop_type, ignore_label=ignore_label)
        )
    if fm and cfg.get("cutout", False):
        n_holes, length = cfg["cutout"]["n_holes"], cfg["cutout"]["length"]
        trs_form.append(psp_trsform.Cutout(n_holes=n_holes, length=length))
    if fm and cfg.get("cutmix", False):
        n_holes, prop_range = cfg["cutmix"]["n_holes"], cfg["cutmix"]["prop_range"]
        trs_form.append(psp_trsform.Cutmix(prop_range=prop_range, n_holes=n_holes))
    return psp_trsform.Compose(trs_form)


def build_city_semi_loader_cp(split, all_cfg, seed=0):
    cfg_dset = all_cfg["dataset"]
    cfg_trainer = all_cfg["trainer"]

    fm = (
        True
        if "cutout" in cfg_dset["train"].keys() or "cutmix" in cfg_dset["train"].keys()
        else False
    )
    acp = True if "acp" in cfg_dset.keys() else False
    acm = cfg_dset["train"].get("acm", False)
    cfg = copy.deepcopy(cfg_dset)
    cfg.update(cfg.get(split, {}))

    workers = cfg.get("workers", 2)
    batch_size = cfg.get("batch_size", 1)
    n_sup = cfg.get("n_sup", 2975)
    coarse = cfg.get("coarse", False)
    coarse_num = cfg.get("coarse_num", 3000)
    prob = cfg["acp"].get("prob", 0.5)
    # build transform
    trs_form = build_transfrom(cfg)
    trs_form_unsup = build_transfrom(cfg, fm=fm)
    if acp:
        paste_trs = build_transfrom(cfg, acp=True)
    else:
        paste_trs = None
    dset = city_dset(
        cfg["data_root"],
        cfg["data_list"],
        trs_form,
        seed,
        n_sup,
        split,
        acp=acp,
        paste_trs=paste_trs,
        prob=prob,
    )

    # build sampler
    sample = DistributedSampler(dset)
    loader = DataLoader(
        dset,
        batch_size=batch_size,
        num_workers=workers,
        sampler=sample,
        shuffle=False,
        pin_memory=False,
    )

    # build sampler for unlabeled set
    dset_unsup = city_dset(
        cfg["data_root"],
        cfg["data_list"].replace("labeled.txt", "unlabeled.txt"),
        trs_form_unsup,
        seed,
        2975 - n_sup,
        split,
        unsup=True,
        coarse=coarse,
        coarse_num=coarse_num,
        fm=fm,
        acm=acm,
    )
    if split == "train":
        sample_unsup = DistributedSampler(dset_unsup)
        loader_unsup = DataLoader(
            dset_unsup,
            batch_size=batch_size,
            num_workers=workers,
            sampler=sample_unsup,
            shuffle=False,
            pin_memory=False,
            drop_last=True,
        )
        return loader, loader_unsup
    return loader
