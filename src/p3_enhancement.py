"""Bounded photometric enhancement, preserving image dimensions and scene geometry."""
import re
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter, ImageOps


def enhance_image(image, request, revision_index=0):
    if type(revision_index) is not int or not 0<=revision_index<=2:
        raise ValueError('Enhancement supports at most two revisions.')
    result=ImageOps.exif_transpose(image).convert('RGB')
    operations=[];q=request.casefold()
    if re.search(r'\b(?:denois\w*|noise)\b',q):
        size=3 if revision_index<2 else 5
        result=result.filter(ImageFilter.MedianFilter(size=size))
        operations.append(dict(operation='median_denoise',kernel_size=size))
    if re.search(r'\b(?:(?:color|colour)\s+(?:cast|correct\w*)|white\s+balance)\b',q):
        arr=np.asarray(result,dtype=np.float32)
        means=arr.mean(axis=(0,1));target=float(means.mean())
        gains=np.clip(target/np.maximum(means,1),0.7,1.5)
        blend=min(1.0,0.8+0.1*revision_index)
        adjusted=np.clip(arr*gains,0,255)
        result=Image.fromarray(np.rint(arr*(1-blend)+adjusted*blend).astype(np.uint8))
        operations.append(dict(operation='bounded_gray_world_white_balance',gains=gains.tolist(),blend=blend))
    if re.search(r'\b(?:brighten|exposure|lighting)\b',q):
        gamma=0.85-0.05*revision_index
        table=[round(255*((i/255)**gamma)) for i in range(256)]
        result=result.point(table*3)
        operations.append(dict(operation='shadow_brightening',gamma=gamma))
    if re.search(r'\bcontrast\b',q):
        factor=1.08+0.04*revision_index
        result=ImageEnhance.Contrast(result).enhance(factor)
        operations.append(dict(operation='contrast',factor=factor))
    if re.search(r'\bsharpen\w*\b',q):
        result=result.filter(ImageFilter.UnsharpMask(radius=1,percent=80+20*revision_index,threshold=3))
        operations.append(dict(operation='unsharp_mask',radius=1,percent=80+20*revision_index))
    if not operations:
        result=ImageOps.autocontrast(result,cutoff=0.5)
        operations.append(dict(operation='autocontrast',cutoff=0.5))
    assert result.size==image.size
    return result,operations
