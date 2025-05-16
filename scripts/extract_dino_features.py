'''
Modified from: https://github.com/Kunhao-Liu/3D-OVS/blob/main/models/DINO_extractor.py
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
from odise.modeling.meta_arch.ldm import LdmFeatureExtractor
import PIL.Image as Image
import numpy as np
import os
import visdom
from termcolor import colored, cprint

import numpy as np
import os.path as osp
import os
import argparse


#os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"  # Reduces fragmentation

class VitExtractor(nn.Module):
    def __init__(self, model_name='dinov2_vitl14'):
        super().__init__()
        self.model = torch.hub.load('facebookresearch/dinov2', model_name)
        self.model.eval()
        self.patch_size = 14
        self.feature_dims = 1024
        self.preprocess = T.Compose([
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # imagenet
        ])

        self._freeze()

    def _freeze(self):
        super().train(mode=False)
        for p in self.parameters():
            p.requires_grad = False

    @torch.no_grad()
    def forward(self, input_img):
        B, C, H, W = input_img.shape
        input_img = self.preprocess(input_img)
        dino_ret = self.model.forward_features(input_img)['x_norm_patchtokens']
        dino_ret = dino_ret.transpose(1, 2).reshape([B, -1, H//self.patch_size, W//self.patch_size])    # [B, 1024, 128, 128]
        return dino_ret



def dino(vit,gt_rgb):
    d_embed=3
    dino_preprocess = T.Compose([
                T.Resize(224 * 8, antialias=True),  # must be a multiple of 14
            ])
    batched_input = dino_preprocess(gt_rgb.permute(0, 3, 1, 2))    # resize
    feature = vit(batched_input)
    gt_embed = F.interpolate(feature, size=(128, 128), mode='bilinear', align_corners=False)    # [b, 1024, 128, 128]
    return gt_embed


def parse_img_file(file_path, mask_gt_rgb=False, bg_color=[0,0,0,255]):
    """
    return np.array of RGB image with range [0, 1]
    """
    rgb = Image.open(file_path).convert('RGB')
    rgb = np.asarray(rgb).astype(np.float32) / 255.0    # [0, 1]
    return rgb

def read_batch(paths):
    batch = []
    for path in paths:
        batch.append(parse_img_file(path))
    batch = np.stack(batch, axis=0)    # [bs, h, w, c]
    batch = torch.from_numpy(batch)    # [bs, c, h, w]
    return batch

def visualise_embedding(dino_embed, env='main', title='Embeddings'):
    vis = visdom.Visdom(env=env)

    dino_embed = dino_embed[:,:3,:,:].cpu().numpy() 
    bs = dino_embed.shape[0]

    for i in range(bs):
        dino_img = (dino_embed[i].transpose(1, 2, 0) - dino_embed[i].min()) / (dino_embed[i].max() - dino_embed[i].min())

        vis.image(
            dino_img.transpose(2, 0, 1),  # Visdom expects (C, H, W)
            opts=dict(title=f'DINO Embed frame({i})', caption=title),
            win=f'dino_embed_{i}'
        )

def save_npy(dino_embed, base_dir):
    dino_embed = dino_embed.cpu().numpy() 
    bs = dino_embed.shape[0]

    for i in range(bs):
        dino_img = dino_embed[i].transpose(1, 2, 0)
        np.save(osp.join(save_dir,f"{i}.npy"), dino_img)



def test():
    vit = VitExtractor().to('cuda:0')
    paths = [f'/home/bing_TUM/bilel/MasterThesis/ManiGaussian/data/train_data/close_jar/all_variations/episodes/episode0/front_rgb/{i}.png' for i in range(10)]
    batch = read_batch(paths).to('cuda:0')
    batch.requires_grad = False
    print(batch.shape)
    gt_rgb = batch
    print('dino processing')
    dino_embed = dino(vit,gt_rgb)
    print('dino_embed done')
    visualise_embedding(dino_embed)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--img_dir",type=str)
    parser.add_argument("--save_dir",type=str)
    args = parser.parse_args()
    save_dir = os.path.join(args.save_dir,"dino_features")
    os.makedirs(save_dir,exist_ok = True)

    data_list = [os.path.join(args.img_dir, f) for f in os.listdir(args.img_dir)]
    batch = read_batch(data_list)
    batch.requires_grad = False

    vit = VitExtractor().to('cuda:0')

    bs = batch.shape[0]
    mbs = 17

    its = bs // mbs

    dino_embed = torch.zeros((bs, 1024, 128, 128)).to('cpu')

    for i in range(its):
        print(f'processing {i}/{its}')
        dino_embed[i*mbs:(i+1)*mbs] = dino(vit,batch[i*mbs:(i+1)*mbs].detach().clone().to('cuda:0')).cpu()
    dino_embed[its*mbs:] = dino(vit,batch[its*mbs:].detach().clone().to('cuda:0')).cpu()

    save_npy(dino_embed, save_dir)


