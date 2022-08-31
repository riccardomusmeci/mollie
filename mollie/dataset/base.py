import os
import numpy as np
from random import sample
from mollie.io import read_rgb
from torch.utils.data import Dataset
from typing import Callable, Dict, List, Tuple

class ImageFolderDataset(Dataset):
    
    EXTENSIONS = (
        "jpg",
        "jpeg",
        "png",
        "ppm",
        "bmp",
        "pgm",
        "tif",
        "tiff",
        "webp",
    )
        
    def __init__(
        self,
        data_dir: str,
        class_map: Dict,
        transform: Callable = None,
        max_samples_per_class: int = None,
    ) -> None:
        """ImageFolderDataset init

        Args:
            data_dir (str): data dir
            class_map (Dict): class map, e.g. {0: ['class_1', 'class_2'], 1: 'class_3', ...}
            transform (Callable, optional): transform. Defaults to None.
            max_samples_per_class (int, optional): max samples per class. Defaults to None.
        """
        super().__init__()

        assert isinstance(class_map, dict), "class_map must be a Python dictionary"

        self.data_dir = data_dir
        self.class_map = class_map
        self.classes = self._find_classes()
        self.images, self.targets = self._get_samples(max_samples_per_class=max_samples_per_class)
        self.transform = transform
        
    def _find_classes(self) -> List[str]:
        """extract classes from class_map (e.g. {0: ["class_a", "class_b"], 1: ["class_c"], ..})

        Returns:
            List[str]: list of classes
        """
        classes = []
        for k, c in self.class_map.items():
            if isinstance(c, list):
                classes += c
            if isinstance(c, str):
                classes.append(c)    
        return classes
            
    def _get_samples(self, max_samples_per_class: int) -> Tuple[List[str], List[int]]:
        """finds image paths + targets for each images

        Returns:
            Tuple[List[str], List[int]]: image paths, targets
        """
        paths = []
        targets = []
        for c, c_values in self.class_map.items():
            if isinstance(c_values, str):
                c_values = [c_values]
            c_images = []
            c_targets = []
            for c_val in c_values:
                class_dir = os.path.join(self.data_dir, c_val)
                c_images += [os.path.join(class_dir, f) for f in os.listdir(class_dir) if f.split(".")[-1].lower() in self.EXTENSIONS]
            if max_samples_per_class is not None:
                c_images = sample(c_images, k=max_samples_per_class)
            c_targets += [c] * len(c_images)
            
            paths += c_images
            targets += c_targets
        
        return paths, targets
    
    def stats(self):
        """prints stats of the dataset
        """
        unique, counts = np.unique(self.targets, return_counts=True)
        num_samples = len(self.targets)
        print(f" ----------- Dataset Stats -----------")
        for k in range(len(unique)):
            classes = self.class_map[k]
            if isinstance(classes, str):
                classes = [classes]
            print(f"> {classes} : {counts[k]}/{num_samples} -> {100 * counts[k] / num_samples:.3f}%")
        print(f" -------------------------------------")
    
    def __getitem__(self, index) -> Tuple:
        
        img_path = self.images[index]
        target = self.targets[index]
        
        img = read_rgb(img_path)
        
        if self.transform is not None:
            img = self.transform(img)
        
        return img, target
            
    def __len__(self):
        return len(self.images)
        
        
    
    