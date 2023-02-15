# 检测参数
img_path = "/data1/glc/STAL-master/data/dataset/cityscapes/leftImg8bit/train/"
model_base = '/data1/glc/STAL-master/experiments/gtav2cityscapes/1.0%/'
Detector_para = {
    "config_file": model_base + 'config.yaml',
    "model_file": model_base + 'checkpoints/ckpt_best_teacher.pth',
    "act_learn_out": model_base + 'checkpoints/act_learn_out/',
    "result_view": True,
    "source_domain": model_base + "checkpoints/act_learn_out/city_uncertainty/",
}
