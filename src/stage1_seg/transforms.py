import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2

class JointTransform:
    """
    Applies albumentations augmentations to both image and mask.
    """
    def __init__(self, use_aug=True, size=512):
        self.use_aug = use_aug
        self.size = size
        
        if self.use_aug:
            self.transform = A.Compose([
                # Scaling & Crop (RandomResizedCrop is excellent for scale invariance)
                # Albumentations 1.4+: requires 'size' or 'height'/'width'. Error said 'size' required.
                A.RandomResizedCrop(size=(size, size), scale=(0.8, 1.0), p=1.0),
                
                # Geometric
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.Rotate(limit=45, p=0.5, border_mode=cv2.BORDER_CONSTANT), # value/mask_value defaults usually 0
                
                # Deformations (Elastic / Grid) - Good for biological structures
                A.OneOf([
                    # ElasticTransform: alpha_affine deprecated/removed. Use Affine for affine.
                    A.ElasticTransform(alpha=120, sigma=120 * 0.05, p=0.5),
                    A.GridDistortion(p=0.5),
                    # OpticalDistortion: shift_limit might be valid in some versions but 'distort_limit' is key.
                    # Removing shift_limit to be safe if warning says invalid.
                    A.OpticalDistortion(distort_limit=0.05, p=0.5),
                ], p=0.3),
                
                # Color/Noise (Image only)
                A.OneOf([
                    # GaussNoise: var_limit usually valid, but if warning, try default or simpler
                    A.GaussNoise(var_limit=(10.0, 50.0), p=0.5),
                    A.RandomBrightnessContrast(p=0.5),
                    A.ColorJitter(p=0.5),
                ], p=0.3),
                
                # Normalize & Tensor
                A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                ToTensorV2(),
            ])
        else:
            # Validation: Simple Resize & Normalize
            self.transform = A.Compose([
                A.Resize(height=size, width=size),
                A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                ToTensorV2(),
            ])

    def __call__(self, image, mask):
        # Convert PIL to Numpy
        image_np = np.array(image)
        mask_np = np.array(mask)
        
        # Albumentations expects mask as (H, W) or (H, W, C), usually uint8/float
        # Our mask is likely uint8 (0-255). 
        # Check if 2D or 3D
        if len(mask_np.shape) == 3:
            mask_np = mask_np[:, :, 0] # Take first channel if RGB/L
            
        # Apply
        transformed = self.transform(image=image_np, mask=mask_np)
        
        image_t = transformed['image']
        mask_t = transformed['mask']
        
        # Mask needs to be (1, H, W) float
        # ToTensorV2 usually keeps shape, but if mask is 2D input, output is 2D Tensor (H, W)
        if len(mask_t.shape) == 2:
            mask_t = mask_t.unsqueeze(0)
            
        return image_t, mask_t.float() / 255.0 if mask_t.max() > 1.0 else mask_t.float()
