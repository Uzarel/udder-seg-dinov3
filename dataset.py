import os
import cv2
import numpy as np

from torch.utils.data import Dataset

class UdderSegmentationDataset(Dataset):
    """
    Dataset for Udder Segmentation.
    Reads 1-channel images and their corresponding masks.

    Args:
        images_dir (str): Path to the images folder.
        masks_dir (str): Path to the masks folder.
        augmentation (albumentations.Compose, optional): Data augmentation pipeline.
    """
    def __init__(self, images_dir, masks_dir, augmentation=None):
        self.images = sorted(os.listdir(images_dir))
        self.image_paths = [os.path.join(images_dir, img) for img in self.images]
        self.mask_paths = [os.path.join(masks_dir, img) for img in self.images]
        self.augmentation = augmentation

    def __getitem__(self, index):
        img_path = self.image_paths[index]
        mask_path = self.mask_paths[index]

        # Load image as a single channel and add a channel dimension
        image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        image = np.expand_dims(image, axis=-1)

        # Load mask, threshold to create a binary mask, and add a channel dimension
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        mask = (mask > 127).astype(np.float32)
        mask = np.expand_dims(mask, axis=-1)

        # Apply augmentations if provided
        if self.augmentation:
            augmented = self.augmentation(image=image, mask=mask)
            image, mask = augmented["image"], augmented["mask"]

        # Convert images from HWC to CHW format
        image = image.transpose(2, 0, 1).astype(np.float32)
        mask = mask.transpose(2, 0, 1).astype(np.float32)
        
        return image, mask

    def __len__(self):
        return len(self.image_paths)
