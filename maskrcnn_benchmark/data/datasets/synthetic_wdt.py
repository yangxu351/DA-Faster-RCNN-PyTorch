import os

import torch
import torch.utils.data
from PIL import Image
import sys

if sys.version_info[0] == 2:
    import xml.etree.cElementTree as ET
else:
    import xml.etree.ElementTree as ET


from maskrcnn_benchmark.structures.bounding_box import BoxList



class SyntheticWDT(torch.utils.data.Dataset):
    # tag: yang changed
    CLASSES = (
        "__background__ ",
        "WindTurbine",
    )

    def __init__(self, data_dir, split, use_difficult=False, transforms=None, is_source=True):
        self.root = data_dir
        self.image_set = split
        self.keep_difficult = use_difficult
        self.transforms = transforms
        # tag:yang adds
        self.is_source = is_source
        
        self._annopath = os.path.join(self.root, "Annotations", "%s.xml")
        self._imgpath = os.path.join(self.root, "JPEGImages", "%s.jpg")
        self._imgsetpath = os.path.join(self.root, "ImageSets", "Main", "%s.txt")

        with open(self._imgsetpath % self.image_set) as f:
            self.ids = f.readlines()
        # self.ids = [x.strip("\n") for x in self.ids]
        #tag: yang changed
        self.ids = [x.split("\t")[0] for x in self.ids]
        self.id_to_img_map = {k: v for k, v in enumerate(self.ids)}

        cls = SyntheticWDT.CLASSES
        self.class_to_ind = dict(zip(cls, range(len(cls))))
        self.categories = dict(zip(range(len(cls)), cls))

    def __getitem__(self, index):
        img_id = self.ids[index]
        img = Image.open(self._imgpath % img_id).convert("RGB")

        target = self.get_groundtruth(index)
        target = target.clip_to_image(remove_empty=True)

        if self.transforms is not None:
            img, target = self.transforms(img, target)

        return img, target, index

    def __len__(self):
        return len(self.ids)

    def get_groundtruth(self, index):
        img_id = self.ids[index]
        anno = ET.parse(self._annopath % img_id).getroot()
        anno = self._preprocess_annotation(anno)

        height, width = anno["im_info"]
        target = BoxList(anno["boxes"], (width, height), mode="xyxy")
        target.add_field("labels", anno["labels"])
        target.add_field("difficult", anno["difficult"])
        # tag: yang added
        classes = anno["labels"]
        domain_labels = torch.ones_like(classes, dtype=torch.uint8) if self.is_source else torch.zeros_like(classes, dtype=torch.uint8)
        domain_labels = domain_labels.bool()
        target.add_field("is_source", domain_labels)
        return target

    def _preprocess_annotation(self, target):
        boxes = []
        gt_classes = []
        difficult_boxes = []
        TO_REMOVE = 1

        for obj in target.iter("object"):
            difficult = int(obj.find("difficult").text) == 1
            if not self.keep_difficult and difficult:
                continue
            # name = obj.find("name").text.lower().strip()
            #tag: yang changed
            name = obj.find("name").text.strip()
            #tag: ignore if not WindTurbine
            if not name == "WindTurbine":
                continue
            bb = obj.find("bndbox")
            # Make pixel indexes 0-based
            # Refer to "https://github.com/rbgirshick/py-faster-rcnn/blob/master/lib/datasets/pascal_voc.py#L208-L211"
            box = [
                bb.find("xmin").text,
                bb.find("ymin").text,
                bb.find("xmax").text,
                bb.find("ymax").text,
            ]
            bndbox = tuple(
                map(lambda x: x - TO_REMOVE, list(map(int, box)))
            )

            boxes.append(bndbox)
            gt_classes.append(self.class_to_ind[name])
            difficult_boxes.append(difficult)

        size = target.find("size")
        im_info = tuple(map(int, (size.find("height").text, size.find("width").text)))

        res = {
            "boxes": torch.tensor(boxes, dtype=torch.float32),
            "labels": torch.tensor(gt_classes),
            "difficult": torch.tensor(difficult_boxes),
            "im_info": im_info,
        }
        return res

    def get_img_info(self, index):
        img_id = self.ids[index]
        anno = ET.parse(self._annopath % img_id).getroot()
        size = anno.find("size")
        im_info = tuple(map(int, (size.find("height").text, size.find("width").text)))
        return {"height": im_info[0], "width": im_info[1]}

    def map_class_id_to_class_name(self, class_id):
        return SyntheticWDT.CLASSES[class_id]

if __name__ == '__main__':
    from maskrcnn_benchmark.config import cfg
    from maskrcnn_benchmark.data import make_data_loader

    data_loader = {}
    cfg.merge_from_file('configs/wdt/e2e_faster_rcnn_R_50_FPN_1x_gpu_wdt_voc.yaml')
    data_loader["source"] = make_data_loader(
        cfg,
        is_train=True,
        is_source=True,
        is_distributed=False,
        start_iter=0,
    )