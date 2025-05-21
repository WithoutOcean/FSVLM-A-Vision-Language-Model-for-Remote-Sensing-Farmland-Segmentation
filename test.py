import argparse
import os
import sys
import deepspeed
import cv2
import numpy as np
import json
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, BitsAndBytesConfig, CLIPImageProcessor
from utils.utils import (DEFAULT_IM_END_TOKEN, DEFAULT_IM_START_TOKEN,
                         AverageMeter, ProgressMeter, Summary, dict_to_cuda,
                         intersectionAndUnionGPU)
from model.FSVLM import FSVLMForCausalLM
from model.llava import conversation as conversation_lib
from model.llava.mm_utils import tokenizer_image_token
from model.segment_anything.utils.transforms import ResizeLongestSide
from utils.utils import (DEFAULT_IM_END_TOKEN, DEFAULT_IM_START_TOKEN,
                         DEFAULT_IMAGE_TOKEN, IMAGE_TOKEN_INDEX)
import glob
import csv
from utils.metrics import Evaluator 
from PIL import Image

def parse_args(args):
    parser = argparse.ArgumentParser(description="chat")
    parser.add_argument("--version", default="yuor weight")
    parser.add_argument("--vis_save_path", default="./vis_output", type=str)
    parser.add_argument("--base_dir", default="your data root path", type=str)
    # parser.add_argument("--RSCLIP_weight_path", default="/opt/data/private/FSVLM1.0/RS5MVIT/RS5M_ViT-H-14.pt", type=str)  ###
    # parser.add_argument("--GeoRS_path", default="/opt/data/private/FSVLM1.0/RS5Model_init/open_clip_pytorch_model.bin", type=str)  ###
    parser.add_argument(
        "--precision",
        default="bf16",
        type=str,
        choices=["fp32", "bf16", "fp16"],
        help="precision for inference",
    )
    parser.add_argument("--encode_type", default="clip-vit")
    parser.add_argument("--image_size", default=1024, type=int, help="image size")
    parser.add_argument("--model_max_length", default=512, type=int)
    parser.add_argument("--lora_r", default=8, type=int)
    parser.add_argument(
        "--vision-tower", default="openai/clip-vit-large-patch14", type=str
    )
    parser.add_argument("--local-rank", default=0, type=int, help="node rank")
    parser.add_argument("--load_in_8bit", action="store_true", default=False)
    parser.add_argument("--load_in_4bit", action="store_true", default=False)
    parser.add_argument("--use_mm_start_end", action="store_true", default=True)
    parser.add_argument(
        "--conv_type",
        default="llava_v1",
        type=str,
        choices=["llava_v1", "llava_llama_2"],
    )
    return parser.parse_args(args)


def preprocess(
    x,
    pixel_mean=torch.Tensor([123.675, 116.28, 103.53]).view(-1, 1, 1),
    pixel_std=torch.Tensor([58.395, 57.12, 57.375]).view(-1, 1, 1),
    img_size=1024,
) -> torch.Tensor:
    """Normalize pixel values and pad to a square input."""
    # Normalize colors
    x = (x - pixel_mean) / pixel_std
    # Pad
    h, w = x.shape[-2:]
    padh = img_size - h
    padw = img_size - w
    x = F.pad(x, (0, padw, 0, padh))
    return x


def main(args):
    args = parse_args(args)
    os.makedirs(args.vis_save_path, exist_ok=True)

    # Create model
    tokenizer = AutoTokenizer.from_pretrained(
        args.version,
        cache_dir=None,
        model_max_length=args.model_max_length,
        padding_side="right",
        use_fast=False,
    )
    tokenizer.pad_token = tokenizer.unk_token
    args.seg_token_idx = tokenizer("[SEG]", add_special_tokens=False).input_ids[0]
    torch_dtype = torch.float32
    if args.precision == "bf16":
        torch_dtype = torch.bfloat16
    elif args.precision == "fp16":
        torch_dtype = torch.half

    kwargs = {"torch_dtype": torch_dtype}
    if args.load_in_4bit:
        kwargs.update(
            {
                "torch_dtype": torch.half,
                "load_in_4bit": True,
                "quantization_config": BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.float16,
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_quant_type="nf4",
                    llm_int8_skip_modules=["visual_model"],
                ),
            }
        )
    elif args.load_in_8bit:
        kwargs.update(
            {
                "torch_dtype": torch.half,
                "quantization_config": BitsAndBytesConfig(
                    llm_int8_skip_modules=["visual_model"],
                    load_in_8bit=True,
                ),
            }
        )

    model = FSVLMForCausalLM.from_pretrained(
        args.version, low_cpu_mem_usage=True, vision_tower=args.vision_tower, seg_token_idx=args.seg_token_idx, **kwargs
    )

    model.config.eos_token_id = tokenizer.eos_token_id
    model.config.bos_token_id = tokenizer.bos_token_id
    model.config.pad_token_id = tokenizer.pad_token_id

    model.get_model().initialize_vision_modules(model.get_model().config)
    vision_tower = model.get_model().get_vision_tower()
    vision_tower.to(dtype=torch_dtype)

    if args.precision == "bf16":
        model = model.bfloat16().cuda()
    elif (
        args.precision == "fp16" and (not args.load_in_4bit) and (not args.load_in_8bit)
    ):
        vision_tower = model.get_model().get_vision_tower()
        model.model.vision_tower = None
        

        model_engine = deepspeed.init_inference(
            model=model,
            dtype=torch.half,
            replace_with_kernel_inject=True,
            replace_method="auto",
        )
        model = model_engine.module
        model.model.vision_tower = vision_tower.half().cuda()
    elif args.precision == "fp32":
        model = model.float().cuda()

    vision_tower = model.get_model().get_vision_tower()
    vision_tower.to(device=args.local_rank)

    clip_image_processor = CLIPImageProcessor.from_pretrained("openai/clip-vit-large-patch14")
    transform = ResizeLongestSide(args.image_size)

    model.eval()
    
    masks_paths=glob.glob(os.path.join(args.base_dir,"test","lbl","*.png"))

    with torch.no_grad():
        intersection_meter = AverageMeter("Intersec", ":6.3f", Summary.SUM)
        union_meter = AverageMeter("Union", ":6.3f", Summary.SUM)
        acc_iou_meter = AverageMeter("gIoU", ":6.3f", Summary.SUM)
        evaluator = Evaluator(2)
        evaluator.reset()
        for mask_path in masks_paths:
            image_path=mask_path.replace("lbl","img")
           
            # f=open(json_path,encoding="utf-8")
            # file_path=f.read()
            text="This is a remote sensing image."#json.loads(file_path)["img_description_eg"]
            prompt= f" {text} Please segment farmland"
            prompt = DEFAULT_IMAGE_TOKEN + "\n" + prompt
            if args.use_mm_start_end:
                    replace_token = (
                        DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN
                    )
            prompt = prompt.replace(DEFAULT_IMAGE_TOKEN, replace_token)

            conv = conversation_lib.conv_templates[args.conv_type].copy()
            conv.messages = []
            if not os.path.exists(image_path):
                    print("File not found in {}".format(image_path))
                    continue

            image_np = cv2.imread(image_path)
            image_np = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)
            original_size_list = [image_np.shape[:2]]
                
            conv.append_message(conv.roles[0], prompt)
            conv.append_message(conv.roles[1], "")
            prompt = conv.get_prompt()
            image_clip = (
                    clip_image_processor.preprocess(image_np, return_tensors="pt")[
                        "pixel_values"
                    ][0]
                    .unsqueeze(0)
                    .cuda()
                )
            
            if args.precision == "bf16":
                    image_clip = image_clip.bfloat16()
            elif args.precision == "fp16":
                    image_clip = image_clip.half()
            else:
                    image_clip = image_clip.float()

            image = transform.apply_image(image_np)
            resize_list = [image.shape[:2]]

            image = (
                    preprocess(torch.from_numpy(image).permute(2, 0, 1).contiguous())
                    .unsqueeze(0)
                    .cuda()
                )
                #mask
            if not os.path.exists(mask_path):
                continue
            mask = Image.open(mask_path)
            masks = np.array(mask)
            #masks=np.max(masks,axis=2)
            masks=masks[np.newaxis,:,:]
            masks = torch.from_numpy(masks)
            masks[masks!=0]=1
            if args.precision == "bf16":
                    image = image.bfloat16()
            elif args.precision == "fp16":
                    image = image.half()
            else:
                    image = image.float()
                
            input_ids = tokenizer_image_token(prompt, tokenizer, return_tensors="pt")
            input_ids = input_ids.unsqueeze(0).cuda()

            output_ids, pred_masks= model.evaluate(
                    image_clip,
                    image,
                    input_ids,
                    resize_list,
                    original_size_list,
                    max_new_tokens=512,
                    tokenizer=tokenizer,
                )
            # print(pred_masks)
            output_ids = output_ids[0][output_ids[0] != IMAGE_TOKEN_INDEX]
            text_output = tokenizer.decode(output_ids, skip_special_tokens=False)
            text_output = text_output.replace("\n", "").replace("  ", " ")
            print("text_output: ", text_output)
        
            masks_list = masks.int()
            output_list =(pred_masks[0] >-0.0).int()
            
            
            evaluator.add_batch(masks_list.cpu().numpy(),output_list.cpu().numpy())
                # generate_png(output_list.cpu().numpy(),masks_list.cpu().numpy(),i)
            assert len(pred_masks) == 1

            intersection, union, acc_iou = 0.0, 0.0, 0.0
            for mask_i, output_i in zip(masks_list, output_list):
                    intersection_i, union_i, _ = intersectionAndUnionGPU(
                        output_i.contiguous().clone(), mask_i.to("cuda:0").contiguous(), 2, ignore_index=255
                    )
                    intersection += intersection_i
                    union += union_i
                    acc_iou += intersection_i / (union_i + 1e-5)
                    acc_iou[union_i == 0] += 1.0  # no-object target
            intersection, union = intersection.cpu().numpy(), union.cpu().numpy()
            acc_iou = acc_iou.cpu().numpy() / masks_list.shape[0]
            intersection_meter.update(intersection), union_meter.update(
                    union
                ), acc_iou_meter.update(acc_iou, n=masks_list.shape[0])

               # save
            for i, pred_mask in enumerate(pred_masks):
                    if pred_mask.shape[0] == 0:
                        continue

                    pred_mask = pred_mask.detach().cpu().numpy()[0]
                    pred_mask = pred_mask > 0

                    save_path = "{}/{}_mask_{}.png".format(
                        args.vis_save_path, image_path.split("/")[-1].split(".")[0], i
                    )
                    cv2.imwrite(save_path, pred_mask * 255)
                    print("{} has been saved.".format(save_path))

                    save_path = "{}/{}_masked_img_{}.png".format(
                        args.vis_save_path, image_path.split("/")[-1].split(".")[0], i
                    )
                    save_img = image_np.copy()
                    save_img[pred_mask] = (
                        image_np * 0.5
                        + pred_mask[:, :, None].astype(np.uint8) * np.array([255, 0, 0]) * 0.5
                    )[pred_mask]
                    save_img = cv2.cvtColor(save_img, cv2.COLOR_RGB2BGR)
                    cv2.imwrite(save_path, save_img)
                    print("{} has been saved.".format(save_path))
        Acc = evaluator.Pixel_Accuracy()
        Acc_class = evaluator.Pixel_Accuracy_Class()
        mIoU = evaluator.Mean_Intersection_over_Union()
        FWIoU = evaluator.Frequency_Weighted_Intersection_over_Union()
        IOU,dice,recall=evaluator.calculate_iou()
        intersection_meter.all_reduce()
        union_meter.all_reduce()
        acc_iou_meter.all_reduce()
        print("Acc:{}, Acc_class:{}, mIoU:{}, fwIoU: {}, Mdice:{}, Mrecall: {}".format(Acc, Acc_class, mIoU, FWIoU,dice,recall))
        csv_filename = f"./best_result"
            
        with open(csv_filename, mode='w', newline='') as file:
                writers = csv.writer(file)
                writers.writerow(['Accuracy', 'Accuracy_Class', 'mIoU', 'fwIoU',"dice","recall"])
                writers.writerow([Acc, Acc_class, mIoU, FWIoU,dice,recall])

      


if __name__ == "__main__":
    main(sys.argv[1:])