import torch
import torchvision.transforms.functional as TF
import random

class JointTransform:
    """
    Applies same geometric transformations to both image and mask.
    Applies color transformations only to image.
    """
    def __init__(self, use_aug=True):
        self.use_aug = use_aug

    def __call__(self, image, mask):
        # Resize first to ensure consistency (or can be done before)
        # Using 320x320 as default (matching config)
        image = TF.resize(image, (320, 320))
        # Mask needs NN interpolation
        mask = TF.resize(mask, (320, 320), interpolation=TF.InterpolationMode.NEAREST)
        
        if self.use_aug:
            # Random Horizontal Flip
            if random.random() > 0.5:
                image = TF.hflip(image)
                mask = TF.hflip(mask)

            # Random Vertical Flip
            if random.random() > 0.5:
                image = TF.vflip(image)
                mask = TF.vflip(mask)

            # Random Rotate
            if random.random() > 0.5:
                angle = random.randint(-45, 45)
                image = TF.rotate(image, angle)
                mask = TF.rotate(mask, angle)

            # Color Jitter (Image only)
            if random.random() > 0.2:
                # TF.adjust_brightness etc. or using ColorJitter info
                # Simple jitter
                brightness = random.uniform(0.8, 1.2)
                contrast = random.uniform(0.8, 1.2)
                image = TF.adjust_brightness(image, brightness)
                image = TF.adjust_contrast(image, contrast)
                
        # Transform to Tensor
        image = TF.to_tensor(image)
        # Normalize (ImageNet stats)
        image = TF.normalize(image, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        
        # Mask to Tensor (keep as 0-1 float)
        mask = TF.to_tensor(mask) 
        
        return image, mask
