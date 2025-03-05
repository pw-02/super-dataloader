#gpu processing rates on V100, ImageNet, batch size 128

image_classification_speeds = {
    "ResNet-18":  0.103698822,
    "ResNet-50":  0.338702304,
    "shufflenet_v2_x1_0": 0.055491176,
    "vgg16": 0.507015224,

}

image_transformer_speed = {
    "levit_128": 0.087138107,
    "mixer_b32_224": 0.238486257,
    "imagenet_vit_b_32": 0.314981863,
    "vit_small_patch32_224":0.087177517,
}