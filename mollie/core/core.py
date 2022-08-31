import numpy as np
from typing import Dict, List
from mollie.trainer import Trainer
from cleanlab.filter import find_label_issues
from mollie.dataset.base import ImageFolderDataset
from cleanlab.count import estimate_confident_joint_and_cv_pred_proba

class Mollie:
    """cleanlab wrapper with PyTorch model training (cpu, cuda, mps) on image classification datasets
    """
    FILTER_BY = ['prune_by_class', 'prune_by_noise_rate', 'both', 'confident_learning', 'predicted_neq_given']
    
    def __init__(
        self,
        trainer: Trainer,
        dataset: ImageFolderDataset,
        cv_n_folds: int = 5,
        filter_by: str = "prune_by_class",
    ) -> None:
        """Cleanlab wrapper for CV classification models 

        Args:
            trainer (Trainer): trainer instance
            dataset (ImageFolderDataset): dataset
            output_dir (str): where to save output json file.
            cv_n_folds (int, optional): cross-val number of out-of-folds experiments. Defaults to 5.
            filter_by (str, optional): filter by method of cleanlab to find errors. Defaults to 'prune_by_class'.
        """
        
        assert filter_by in self.FILTER_BY, f"filter_by must be one of {self.FILTER_BY}"
        
        if hasattr(dataset, "targets") is False:
            print(f"[ERROR] Dataset must have a 'targets' field containing for each image the associate label.")
            quit()
        
        self.trainer = trainer
        self.dataset = dataset
        self.cv_n_folds = cv_n_folds
        self.filter_by = filter_by
        
        self.X = np.array(range(len(dataset)))
        self.labels = np.array(dataset.targets)
    
    def _prepare_output(
        self, 
        pred_probs: np.array,
        noisy_indices: List[int]
    ) -> List[Dict]:
        """prepares data to save

        Args:
            pred_probs (np.array): predicted probabilities from cleanlab
            noisy_indices (List[int]): noisy indice from cleanlab

        Returns:
            List[Dict]: List of dictionaries with error infos.
        """
        noisy_json = []
        for idx in noisy_indices:
            
            ground_truth_c = self.dataset.targets[idx]
            pred_label_c = np.argmax(pred_probs[idx, :])
            pred_label_score = max(pred_probs[idx, :])
            
            noisy_json.append({
                "file_name": self.dataset.images[idx],
                "ground_truth": self.dataset.class_map[ground_truth_c],
                "predicted": self.dataset.class_map[pred_label_c],
                "prediction_score": pred_label_score
            })
            
        return noisy_json
                 
    def start(self) -> List[Dict]:
        """finds errors in dataset labels

        Returns:
            List[Dict]: List of dictionaries containing labeling errors
        """
        confident_joint, pred_probs = estimate_confident_joint_and_cv_pred_proba(
            X=self.X,
            labels=self.labels,
            clf=self.trainer,
            cv_n_folds=self.cv_n_folds
        )
        
        noisy_indices_bool = find_label_issues(
            labels=self.labels,
            pred_probs=pred_probs,
            confident_joint=confident_joint,
            filter_by=self.filter_by
        )
        noisy_indices = [i for i, v in enumerate(noisy_indices_bool) if v]
        
        print(f"\n[INFO] Found {len(noisy_indices)} errors in the dataset.")
        
        return self._prepare_output(
            pred_probs=pred_probs,
            noisy_indices=noisy_indices
        )
        
    
        
        
        
        
        
        
   
        
        
        
        