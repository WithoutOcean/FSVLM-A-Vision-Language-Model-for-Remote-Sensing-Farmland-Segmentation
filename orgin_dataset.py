import glob
import json
import os
import random
import tifffile as tf 
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from pycocotools.coco import COCO
from transformers import CLIPImageProcessor

from model.llava import conversation as conversation_lib
from model.segment_anything.utils.transforms import ResizeLongestSide

from .utils import ANSWER_LIST, SHORT_QUESTION_LIST


sample_ratio=0.7
def init_dataset(base_image_dir):
    # 取classes
    classes = []
    with open("./utils/CLASS_NAME") as f:
        for line in f.readlines():
            classes.append(line.strip().split(": ")[-1])
    classes = np.array(classes)
    images = []
    # 取label路径
    labels = glob.glob(
        os.path.join(base_image_dir, "dataset", "lbl", "*.png")
    )
    # 取image路径
    images = [
        x.replace("lbl", "img") for x in labels
    ]
    ratio=int(len(images)*sample_ratio)
    # print(' ratio: ', ratio)
    return classes, images[:ratio], labels[:ratio]

def gdal_loader(img_path, mask_path):
    img = tf.imread(img_path)
    mask = tf.imread(mask_path)

    img = img / 255.0  
    return img, mask

class orginDataset(torch.utils.data.Dataset):
    pixel_mean = torch.Tensor([123.675, 116.28, 103.53]).view(-1, 1, 1)
    pixel_std = torch.Tensor([58.395, 57.12, 57.375]).view(-1, 1, 1)
    img_size = 1024
    ignore_label = 255

    def __init__(
        self,
        base_image_dir,
        tokenizer,
        vision_tower,
        samples_per_epoch=500 * 8 * 2 * 10,
        precision: str = "fp32",
        image_size: int = 224,
        num_classes_per_sample: int = 3,
        exclude_val=False,
        orgin_data="FIT",
    ):
        self.exclude_val = exclude_val
        self.samples_per_epoch = samples_per_epoch
        self.num_classes_per_sample = num_classes_per_sample

        self.base_image_dir = base_image_dir
        self.image_size = image_size
        self.tokenizer = tokenizer
        self.precision = precision
        self.transform = ResizeLongestSide(image_size)
        self.clip_image_processor = CLIPImageProcessor.from_pretrained(vision_tower)

        self.short_question_list = SHORT_QUESTION_LIST
        self.answer_list = ANSWER_LIST

        self.data2list = {}
        self.data2classes = {}

        self.sem_seg_datas = orgin_data
        ds =self.sem_seg_datas
        classes, images, labels = eval("init_{}".format(ds))(base_image_dir)
        self.data2list[ds] = (images, labels)
        self.data2classes[ds] = classes
        self.length=len(images)
    
        if ds in self.sem_seg_datas:
            self.hunan_class2index = {
                c: i for i, c in enumerate(self.data2classes[ds])
            }

    def __len__(self):
        return self.samples_per_epoch

    def preprocess(self, x: torch.Tensor) -> torch.Tensor:
        """Normalize pixel values and pad to a square input."""
        # Normalize colors
        x = (x - self.pixel_mean) / self.pixel_std

        # Pad
        h, w = x.shape[-2:]
        padh = self.img_size - h
        padw = self.img_size - w
        x = F.pad(x, (0, padw, 0, padh))
        return x

    def __getitem__(self, idx):
        ds = self.sem_seg_datas
        image, labels = self.data2list[ds]
        # print("seg: ", len(image))
        idx = random.randint(0, len(image) - 1)
        image_path = image[idx]
        label_path = labels[idx]

        label = Image.open(label_path)
        label = np.array(label)
        label[label == 0] = 255
        img = cv2.imread(image_path)
        image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # preprocess image for clip
        image_clip = self.clip_image_processor.preprocess(
            image, return_tensors="pt"
        )["pixel_values"][0]
        image = self.transform.apply_image(image)  # preprocess image for sam
        resize = image.shape[:2]
        unique_label = np.unique(label).tolist()
        if 255 in unique_label:
            unique_label.remove(255)
        if len(unique_label) == 0:
            return self.__getitem__(0)

        classes = [self.data2classes[ds][class_id] for class_id in unique_label]
        
        if len(classes) >= self.num_classes_per_sample:
            sampled_classes = np.random.choice(
                 classes, size=self.num_classes_per_sample, replace=False
            ).tolist()
        else:
            sampled_classes = classes
            
        questions = []
        answers = []
        class_ids = []
        for sampled_cls in sampled_classes:
            text = sampled_cls

            assert len(text.split("||")) == 1
            question_template = random.choice(self.short_question_list)
            questions.append(question_template.format(class_name=text.lower()))

            answers.append(random.choice(self.answer_list))

            if ds in ["paco_lvis", "pascal_part"]:
                continue

            class_id = self.data2classes[ds].tolist().index(sampled_cls)
            class_ids.append(class_id)

        conversations = []
        conv = conversation_lib.default_conversation.copy()

        i = 0
        while i < len(questions):
            conv.messages = []
            conv.append_message(conv.roles[0], questions[i])
            conv.append_message(conv.roles[1], answers[i])
            conversations.append(conv.get_prompt())
            i += 1

        image = self.preprocess(torch.from_numpy(image).permute(2, 0, 1).contiguous())

        label = torch.from_numpy(label).long()
        masks = []
        for class_id in class_ids:
            masks.append(label == class_id)
        masks = torch.stack(masks, dim=0)
        return (
            image_path,
            image,
            image_clip,
            conversations,
            masks,
            label,
            resize,
            questions,
            sampled_classes,
        )
