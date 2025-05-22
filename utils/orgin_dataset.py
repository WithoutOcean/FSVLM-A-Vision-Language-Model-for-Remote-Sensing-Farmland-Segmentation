from utils.utils import (DEFAULT_IM_END_TOKEN, DEFAULT_IM_START_TOKEN,
                         DEFAULT_IMAGE_PATCH_TOKEN,DEFAULT_IMAGE_VISION_START,
DEFAULT_IMAGE_VISION_END,
DEFAULT_IMAGE_PAD,
)
import os
import random
import json
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from pycocotools import mask
from transformers import CLIPImageProcessor
import glob
from model.llava import conversation as conversation_lib
from model.segment_anything.utils.transforms import ResizeLongestSide
from PIL import Image
from .utils import ANSWER_LIST, SHORT_QUESTION_LIST
import re
def init_farmsegvl(base_image_dir):

    images=glob.glob(os.path.join(base_image_dir,"train","img","*.png"))
    masks=[x.replace("img","lbl") for x in images]
    texts=[x.replace("img","json") for x in images]
    texts=[x.replace(".png",".json") for x in texts]
    return texts, images, masks

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
        sem_seg_data="farmsegvl",
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
        self.sem_seg_datas = sem_seg_data
        ds =self.sem_seg_datas
        text, images, labels = eval("init_{}".format(ds))(base_image_dir)
        self.data2list[ds] = (text, images, labels)
        self.length=len(images)
        print('len(images):',len(images))
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
        texts,images, labels= self.data2list[ds]
        # print("refer: ", len(image))
        idx = random.randint(0, len(images) - 1)
        image_path = images[idx]
        label_path = labels[idx]
        text_path=texts[idx]
        with open(text_path, "r", encoding="utf-8") as f:
            data = json.load(f) 
        # f = open(text_path, encoding = 'utf-8')
        # path=f.read()
        text =data['img_description_eg']
        sampled_classes=text
        questions = []
        answers = []
        text = text.strip()
        assert len(text.split("||")) == 1
        question_template = random.choice(self.short_question_list)
        questions.append(question_template.format(text_name=text,class_name="farmland"))
        answers.append(random.choice(self.answer_list))

        conversations = []
        conv = conversation_lib.default_conversation.copy()

        i = 0
        while i < len(questions):
            conv.messages = []
            conv.append_message(conv.roles[0], questions[i])
            conv.append_message(conv.roles[1], answers[i])
            conversations.append(conv.get_prompt())
            i += 1

        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # preprocess image for clip
        image_clip = self.clip_image_processor.preprocess(image, return_tensors="pt")[
            "pixel_values"
        ][0]

        image = self.transform.apply_image(image)  # preprocess image for sam
        resize = image.shape[:2]
        image = self.preprocess(torch.from_numpy(image).permute(2, 0, 1).contiguous())

        flag = False
        mask = Image.open(label_path)
        masks = np.array(mask)
        # masks=masks.max(axis=2)
        masks[masks!=0]=1
        masks =  masks[np.newaxis,:,:]
        # masks = np.stack(masks, axis=0)
        masks = torch.from_numpy(masks)
        label = torch.ones(masks.shape[1], masks.shape[2]) * self.ignore_label
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
