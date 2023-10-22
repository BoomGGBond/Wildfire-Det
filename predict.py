import os
import time
import json
import datetime
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import torch
from torchvision import transforms

from network_files import MaskRCNN
from backbone import BackboneWithFPN, LastLevelMaxPool
from draw_box_utils import draw_objs


def create_model(num_classes, box_thresh=0.5):
    import torchvision
    from torchvision.models.feature_extraction import create_feature_extractor
    
    backbone = torchvision.models.convnext_tiny(pretrained=True)
    return_layers = {"features.3" : "0", #outputs_size:28*28 (3):Sequential(groups:192)
                    "features.5" : "1", #outputs_size:14*14 (5):Sequential(groups:384)
                    "features.7" : "2"} #outputs_size:7*7 (7):Sequential(groups:768)
    in_channels_list = [192,384,768]
    new_backbone = create_feature_extractor(backbone, return_layers)
    backbone_with_fpn = BackboneWithFPN(new_backbone,
                                        return_layers=return_layers,
                                        in_channels_list=in_channels_list,
                                        out_channels=96,
                                        extra_blocks=LastLevelMaxPool(),
                                        re_getter=False)
    model = MaskRCNN(backbone_with_fpn,
                     num_classes=num_classes,
                     rpn_score_thresh=box_thresh,
                     box_score_thresh=box_thresh)

    return model


def time_synchronized():
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    return time.time()


def main():
    num_classes = 1
    box_thresh = 0.5
    weights_path = "./save_weights/model_24.pth"
    img_path = "./test.jpg"
    label_json_path = './class.json'

    # get devices
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("using {} device.".format(device))

    # create model
    model = create_model(num_classes=num_classes + 1, box_thresh=box_thresh)

    # load train weights
    assert os.path.exists(weights_path), "{} file dose not exist.".format(weights_path)
    weights_dict = torch.load(weights_path, map_location='cpu')
    weights_dict = weights_dict["model"] if "model" in weights_dict else weights_dict
    model.load_state_dict(weights_dict)
    model.to(device)

    # read class_indict
    assert os.path.exists(label_json_path), "json file {} dose not exist.".format(label_json_path)
    with open(label_json_path, 'r') as json_file:
        category_index = json.load(json_file)

    # load image
    assert os.path.exists(img_path), f"{img_path} does not exits."
    original_img = Image.open(img_path).convert('RGB')

    # from pil image to tensor, do not normalize image
    data_transform = transforms.Compose([transforms.ToTensor()])
    img = data_transform(original_img)
    # expand batch dimension
    img = torch.unsqueeze(img, dim=0)

    model.eval() 
    with torch.no_grad():
        # init
        img_height, img_width = img.shape[-2:]
        init_img = torch.zeros((1, 3, img_height, img_width), device=device)
        model(init_img)

        t_start = time_synchronized()
        predictions = model(img.to(device))[0]
        t_end = time_synchronized()
        print("inference+NMS time: {}".format(t_end - t_start))

        predict_boxes = predictions["boxes"].to("cpu").numpy()
        predict_classes = predictions["labels"].to("cpu").numpy()
        predict_scores = predictions["scores"].to("cpu").numpy()
        predict_mask = predictions["masks"].to("cpu").numpy()
        predict_mask = np.squeeze(predict_mask, axis=1)  # [batch, 1, h, w] -> [batch, h, w]

        if len(predict_boxes) == 0:
            print("没有检测到任何目标!")
            return

        plot_img = draw_objs(original_img,
                             boxes=predict_boxes,
                             classes=predict_classes,
                             scores=predict_scores,
                             masks=predict_mask,
                             category_index=category_index,
                             line_thickness=3,
                             font='arial.ttf',
                             font_size=20)
        plt.imshow(plot_img)
        plt.show()
        plot_img.save('test_result{}.png'.format(datetime.datetime.now().strftime("%Y%m%d-%H%M%S")))


if __name__ == '__main__':
    main()

