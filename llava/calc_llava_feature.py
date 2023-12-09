import os, torch

from LLaVA.llava.model.builder import load_pretrained_model
from LLaVA.llava.mm_utils import get_model_name_from_path
from LLaVA.llava.eval.run_llava import eval_model
from LLaVA.llava.utils import disable_torch_init
from LLaVA.llava.mm_utils import process_images


model_path = "/root/LLaVA_backup/ShareGPT4V-7B"
disable_torch_init()
tokenizer, model, image_processor, context_len = load_pretrained_model(
    model_path=model_path,
    model_base=None,
    model_name=get_model_name_from_path(model_path)
)
model.eval()
import pickle
import requests
import torch
from transformers import AutoProcessor, CLIPVisionModel
import tqdm
import gc

from pympler import asizeof

def load_image_dict(image_dict_path):
    """Load the image dict.

    Args:
        image_dict_path (str): The path to the image dict.

    Returns:
        image_dict
    """
    try :
        with open(image_dict_path, "rb") as f:
            image_dict = pickle.load(f)
    except:
        raise FileNotFoundError("Image dict not found, which should be a url or path to a pkl file.")
    return image_dict



def calulate_clip_scores(image_dict, device="cuda:0"):
    score_dict = {}
    
    batch_size = 16
    images_batch = []
    keys = []
    for it, (k,v) in tqdm.tqdm(enumerate(image_dict.items())):
        if v is None:
            print ("None")
            continue
        # image = v["image"]
        images_batch.append(v["image"])
        keys.append(k)
        if len(images_batch) == batch_size:
            # inputs_batch = processor.preprocess(images_batch, return_tensors="pt").to(device)
            image_tensor = process_images(images_batch, image_processor, model.config)
            # import ipdb; ipdb.set_trace()
            if type(image_tensor) is list:
                image_tensor = [image.to(model.device, dtype=torch.float16) for image in image_tensor]
            else:
                image_tensor = image_tensor.to(model.device, dtype=torch.float16)
            with torch.inference_mode():
                image_features = model.encode_images(image_tensor)
                
            for i, key in enumerate(keys):
                score_dict[key] = image_features[i].cpu().numpy()
            del image_tensor, image_features, images_batch, keys
            gc.collect()
            torch.cuda.empty_cache()
            # print("image_dict: ", asizeof.asizeof(image_dict)/1024/1024/1024)

            images_batch = []
            keys = []
    # deal with the last batch
    if len(images_batch) > 0:
        image_tensor = process_images(images_batch, image_processor, model.config)
        if type(image_tensor) is list:
            image_tensor = [image.to(model.device, dtype=torch.float16) for image in image_tensor]
        else:
            image_tensor = image_tensor.to(model.device, dtype=torch.float16)
        with torch.inference_mode():
            image_features = model.encode_images(image_tensor)
        for i, key in enumerate(keys):
            score_dict[key] = image_features[i].unsqueeze(0).cpu().numpy()
        
    return score_dict


if __name__ == "__main__":
    image_dict_path = "/root/RAVLM/knowledge_base/cathedral_image_subset.pkl"
    image_dict = load_image_dict(image_dict_path)
    score_dict = calulate_clip_scores(image_dict)
    with open("/root/RAVLM/knowledge_base/cathedral_image_feature_llava_no_proj.pkl", "wb") as f:
        pickle.dump(score_dict, f)


