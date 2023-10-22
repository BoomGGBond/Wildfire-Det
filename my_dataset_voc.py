import os
import json

from lxml import etree
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
from train_utils import convert_to_coco_api


class VOCInstances(Dataset):
    def __init__(self, voc_root, year="2007", txt_name: str = "train.txt", transforms=None):
        super().__init__()
        if isinstance(year, int):
            year = str(year)
        assert year in ["2007", "2012"], "year must be in ['2007', '2012']"
        if "VOCdevkit" in voc_root:
            root = os.path.join(voc_root, f"VOC{year}")
        else:
            root = os.path.join(voc_root, "VOCdevkit", f"VOC{year}")
        assert os.path.exists(root), "path '{}' does not exist.".format(root)
        image_dir = os.path.join(root, 'JPEGImages')
        xml_dir = os.path.join(root, 'Annotations')
        mask_dir = os.path.join(root, 'SegmentationObjectPNG')

        txt_path = os.path.join(root, "ImageSets", "Segmentation", txt_name)
        assert os.path.exists(txt_path), "file '{}' does not exist.".format(txt_path)
        with open(os.path.join(txt_path), "r") as f:
            file_names = [x.strip() for x in f.readlines() if len(x.strip()) > 0]

        # read class_indict
        json_file = 'class.json'
        assert os.path.exists(json_file), "{} file not exist.".format(json_file)
        with open(json_file, 'r') as f:
            idx2classes = json.load(f)
            self.class_dict = dict([(v, k) for k, v in idx2classes.items()])

        self.images_path = []
        self.xmls_path = []
        self.xmls_info = []
        self.masks_path = []
        self.objects_bboxes = []
        self.masks = []

        images_path = [os.path.join(image_dir, x + ".jpg") for x in file_names]
        xmls_path = [os.path.join(xml_dir, x + '.xml') for x in file_names]
        masks_path = [os.path.join(mask_dir, x + ".png") for x in file_names]
        for idx, (img_path, xml_path, mask_path) in enumerate(zip(images_path, xmls_path, masks_path)):
            assert os.path.exists(img_path), f"not find {img_path}"
            assert os.path.exists(xml_path), f"not find {xml_path}"
            assert os.path.exists(mask_path), f"not find {mask_path}"

            with open(xml_path) as fid:
                xml_str = fid.read()
            xml = etree.fromstring(xml_str)
            obs_dict = parse_xml_to_dict(xml)["annotation"]
            obs_bboxes = parse_objects(obs_dict, xml_path, self.class_dict, idx)
            num_objs = obs_bboxes["boxes"].shape[0]

            instances_mask = Image.open(mask_path)
            instances_mask = np.array(instances_mask)
            instances_mask[instances_mask == 255] = 0

            num_instances = instances_mask.max()
            if num_objs != num_instances:
                print(f"warning: num_boxes:{num_objs} and num_instances:{num_instances} do not correspond. "
                      f"skip image:{img_path}")
                continue

            self.images_path.append(img_path)
            self.xmls_path.append(xml_path)
            self.xmls_info.append(obs_dict)
            self.masks_path.append(mask_path)
            self.objects_bboxes.append(obs_bboxes)
            self.masks.append(instances_mask)

        self.transforms = transforms
        self.coco = convert_to_coco_api(self)

    def parse_mask(self, idx: int):
        mask = self.masks[idx]
        c = mask.max()
        masks = []
        for i in range(1, c+1):
            masks.append(mask == i)
        masks = np.stack(masks, axis=0)
        return torch.as_tensor(masks, dtype=torch.uint8)

    def __getitem__(self, idx):
        """
        Args:
            idx (int): Index

        Returns:
            tuple: (image, target) where target is the image segmentation.
        """
        img = Image.open(self.images_path[idx]).convert('RGB')
        target = self.objects_bboxes[idx]
        masks = self.parse_mask(idx)
        target["masks"] = masks

        if self.transforms is not None:
            img, target = self.transforms(img, target)

        return img, target

    def __len__(self):
        return len(self.images_path)

    def get_height_and_width(self, idx):
        # read xml
        data = self.xmls_info[idx]
        data_height = int(data["size"]["height"])
        data_width = int(data["size"]["width"])
        return data_height, data_width

    def get_annotations(self, idx):
        data = self.xmls_info[idx]
        h = int(data["size"]["height"])
        w = int(data["size"]["width"])
        target = self.objects_bboxes[idx]
        masks = self.parse_mask(idx)
        target["masks"] = masks
        return target, h, w

    @staticmethod
    def collate_fn(batch):
        return tuple(zip(*batch))


def parse_xml_to_dict(xml):
    if len(xml) == 0: 
        return {xml.tag: xml.text}

    result = {}
    for child in xml:
        child_result = parse_xml_to_dict(child) 
        if child.tag != 'object':
            result[child.tag] = child_result[child.tag]
        else:
            if child.tag not in result:  
                result[child.tag] = []
            result[child.tag].append(child_result[child.tag])
    return {xml.tag: result}


def parse_objects(data: dict, xml_path: str, class_dict: dict, idx: int):
    boxes = []
    labels = []
    iscrowd = []
    assert "object" in data, "{} lack of object information.".format(xml_path)
    for obj in data["object"]:
        xmin = float(obj["bndbox"]["xmin"])
        xmax = float(obj["bndbox"]["xmax"])
        ymin = float(obj["bndbox"]["ymin"])
        ymax = float(obj["bndbox"]["ymax"])

        if xmax <= xmin or ymax <= ymin:
            print("Warning: in '{}' xml, there are some bbox w/h <=0".format(xml_path))
            continue

        boxes.append([xmin, ymin, xmax, ymax])
        labels.append(int(class_dict[obj["name"]]))
        if "difficult" in obj:
            iscrowd.append(int(obj["difficult"]))
        else:
            iscrowd.append(0)

    # convert everything into a torch.Tensor
    boxes = torch.as_tensor(boxes, dtype=torch.float32)
    labels = torch.as_tensor(labels, dtype=torch.int64)
    iscrowd = torch.as_tensor(iscrowd, dtype=torch.int64)
    image_id = torch.tensor([idx])
    area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])

    return {"boxes": boxes,
            "labels": labels,
            "iscrowd": iscrowd,
            "image_id": image_id,
            "area": area}


if __name__ == '__main__':
    dataset = VOCInstances(voc_root="")
    print(len(dataset))
    d1 = dataset[0]
