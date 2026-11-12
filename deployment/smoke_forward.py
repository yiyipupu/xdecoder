#!/usr/bin/env python3
"""Single-image CUDA forward check. No dataset, mask or metric is required."""
import argparse
import json
import logging
import os
from pathlib import Path
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--method', required=True, choices=['san', 'xdecoder'])
    parser.add_argument('--repo', required=True, type=Path)
    parser.add_argument('--weights', required=True, type=Path)
    parser.add_argument('--image', required=True, type=Path)
    parser.add_argument('--text', default='cup')
    parser.add_argument('--short-side', type=int, default=512)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    if args.short_side <= 0:
        parser.error('--short-side must be positive')
    import numpy as np
    from PIL import Image
    import torch
    logging.basicConfig(level=logging.INFO)
    if not torch.cuda.is_available():
        parser.error('CUDA unavailable; run probe_gpu.py first')
    repo, weights, image_path, output = [p.resolve() for p in
                                       (args.repo, args.weights, args.image, args.output)]
    output.mkdir(parents=True, exist_ok=True)
    image = Image.open(image_path).convert('RGB')
    sys.path.insert(0, str(repo))
    os.chdir(repo)
    torch.cuda.reset_peak_memory_stats()
    with torch.no_grad():
        if args.method == 'san':
            from predict import Predictor
            model = Predictor('configs/san_clip_vit_large_res4_coco.yaml', str(weights))
            # Official Predictor fixes the short side to 640. This deployment
            # smoke check explicitly allows smaller inputs; it is not evaluation.
            def preprocess(im):
                scale = args.short_side / min(im.size)
                resized = im.resize((round(im.width * scale), round(im.height * scale)),
                                    resample=Image.BICUBIC)
                return torch.from_numpy(np.asarray(resized).copy()).float().permute(2, 0, 1)
            model._preprocess = preprocess
            mask = model.predict(image, [args.text], augment_vocabulary='COCO-all')['sem_seg'] == 0
        else:
            from utils.arguments import load_opt_from_config_files
            from modeling.BaseModel import BaseModel
            from modeling import build_model
            from detectron2.data import MetadataCatalog
            from torchvision import transforms
            opt = load_opt_from_config_files(['configs/xdecoder/xdecoder_focalt_lang.yaml'])
            opt['MODEL']['TEXT']['PRETRAINED_TOKENIZER'] = str(repo / 'deployment/tokenizer')
            opt.update(device=torch.device('cuda'), CUDA=True, rank=0, local_rank=0,
                       world_size=1, local_size=1)
            model = BaseModel(opt, build_model(opt)).from_pretrained(str(weights)).eval().cuda()
            model.model.sem_seg_head.predictor.lang_encoder.get_text_embeddings(
                ['background', 'background'], is_eval=False)
            model.model.metadata = MetadataCatalog.get('ade20k_panoptic_train')
            resized = transforms.Resize(args.short_side, interpolation=Image.BICUBIC)(image)
            tensor = torch.from_numpy(np.asarray(resized).copy()).permute(2, 0, 1).cuda()
            inputs = [dict(image=tensor, height=image.height, width=image.width,
                           groundings={'texts': [[args.text]]})]
            logits = model.model.evaluate_grounding(inputs, None)[0]['grounding_mask'][0]
            if not torch.isfinite(logits).all():
                raise ValueError('Nonfinite prediction logits')
            mask = (logits > 0).cpu().numpy()
        torch.cuda.synchronize()
    mask = np.asarray(mask, dtype=bool)
    if mask.shape != (image.height, image.width):
        raise ValueError(f'Unexpected output shape: {mask.shape}')
    Image.fromarray(mask.astype('uint8') * 255).save(output / 'mask.png')
    overlay = np.asarray(image).copy()
    overlay[mask] = (overlay[mask] * 0.5 + np.array([255, 0, 0]) * 0.5).astype('uint8')
    Image.fromarray(overlay).save(output / 'overlay.png')
    result = dict(status='forward_passed', method=args.method, image=str(image_path),
                  text=args.text, short_side=args.short_side, output_shape=list(mask.shape),
                  foreground_pixels=int(mask.sum()), torch=torch.__version__,
                  torch_cuda=torch.version.cuda, gpu=torch.cuda.get_device_name(),
                  peak_allocated_mib=round(torch.cuda.max_memory_allocated() / 2**20, 1),
                  note='Deployment check only; no accuracy or training claim')
    (output / 'result.json').write_text(json.dumps(result, ensure_ascii=False, indent=2))
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
