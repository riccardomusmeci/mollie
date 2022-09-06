import os
import timm
import torch
import numpy as np
import torch.nn as nn
import torchvision.transforms as T
from sklearn import metrics as metrics
from sklearn.base import BaseEstimator
from torch.utils.data import DataLoader
from mollie.sampler import ImbalancedSampler
from mollie.dataset import ImageFolderDataset
from mollie.loss import LabelSmoothingCrossEntropy
from typing import Callable, Tuple, Optional, List, Union, Dict

class Trainer(BaseEstimator):
    
    def __init__(
        self,
        data_dir: str,
        model_name: str,
        class_map: Dict,
        input_size: Union[int, Tuple, List] = 224,
        val_dir: str = None,
        max_samples_per_class: int = None,
        batch_size: int = 16,
        epochs: int = 10,
        lr: float = .0001,
        drop_rate: float = 0.4,
        metric: str = "f1",
        num_workers=0,
        checkpoints_dir: str = "checkpoints",
        imbalanced: bool = False,
        seed = 42,
        verbose: bool = True
    ) -> None:
        """Trainer class extending sklearn BaseEstimator

        Args:
            data_dir (str): data dir to get cross-val datasets (train+val) from
            model_name (str): model to train
            class_map (Dict): class map, e.g. {0: ['class_1', 'class_2'], 1: ['class_3'], ..}
            input_size (Union[int, Tuple, List], optional): input size of the model. Defaults to 224.
            val_dir (str, optional): optional validation dir to show some extra metrics. Defaults to None.
            max_samples_per_class (int, optional): max samples per class during training+val. Defaults to None.
            batch_size (int, optional): batch size. Defaults to 16.
            epochs (int, optional): number of training epochs (keep it low). Defaults to 10.
            lr (float, optional): learning rate. Defaults to .0001.
            drop_rate (float, optional): drop rate for the model. Defaults to 0.4.
            metric (str, optional): metric to show in the extra validation set. Defaults to "f1".
            num_workers (int, optional): number of workers. Defaults to 0.
            checkpoints_dir (str, optional): where to save trained model if extra validation is on. Defaults to "checkpoints".
            imbalanced (bool, optional): if dataset is imbalanced. Defaults to False.
            seed (int, optional): random seed. Defaults to 42.
            verbose (bool, optional): verbose mode. Defaults to True.
        """
        super().__init__()
        
        assert metric in ["f1", "accuracy", "precision", "recall"], "metric param must be one of f1, accuracy, precision, recall"
        
        self._seed_everything(seed=seed)
        self.seed = seed
        self.data_dir = data_dir
        self.val_dir = val_dir
        self.class_map = class_map
        self.num_classes = len(class_map)
        self.batch_size = batch_size
        self.model_name = model_name
        self.max_samples_per_class = max_samples_per_class
        self.epochs = epochs
        self.lr = lr
        self.drop_rate = drop_rate
        self.metric = metric
        self.num_workers = num_workers
        self.checkpoints_dir = checkpoints_dir
        self.imbalanced = imbalanced
        self.verbose = verbose
        self._device = None
        
        if isinstance(input_size, int):
            self.input_size = (input_size, input_size)
        else:
            self.input_size = input_size
            
        self.best_model = None
    
    @property
    def device(self) -> str:
        
        if self._device is None:
            if torch.has_mps:
                self._device = "mps"
            elif torch.has_cuda:
                self._device = "cuda:0"
            else:
                self._device = "cpu"
        
        return self._device
    
    def _seed_everything(
        self, 
        seed: int
    ):
        # pseudo random determinism
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
        
    def _setup(self):
        """trainer supporting tools setup
        """
        print(f"> Trainer setup")
        self.train_transform = T.Compose([
            T.RandomRotation(20),
            T.RandomResizedCrop(self.input_size, scale=(0.85, 1.0)),
            T.RandomHorizontalFlip(0.5),
            T.ColorJitter(0.05, 0.05, 0.05, 0.025),
            T.ToTensor()
        ])
        self.val_transform = T.Compose([
            T.Resize(self.input_size), 
            T.ToTensor()
        ])
        
        if self.val_dir:
            print(f"> Setting up validation dataset..")
            self.val_dataset = ImageFolderDataset(
                data_dir=self.val_dir,
                class_map=self.class_map,
                transform=self.val_transform,
                max_samples_per_class=None
            )
            self.val_dataset.stats()
            self.val_dl = DataLoader(
                dataset=self.val_dataset,
                batch_size=self.batch_size,
                shuffle=False
            )
        
        
        print(f"> Setting up model {self.model_name}")
        self.model = timm.create_model(
            self.model_name,
            pretrained=True,
            num_classes=self.num_classes,
            drop_rate=self.drop_rate
        )
        self.model = self.model.to(self.device)
        
        print(f"> Setting up optimizer (Adam), lr_scheduler (CosineAnnealingLR), and loss (LabelSmoothingCrossEntropy)")
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.lr
        )
        self.lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer=self.optimizer,
            T_max=self.epochs,
            eta_min=self.lr*0.01
        )
        self.criterion = LabelSmoothingCrossEntropy(classes=self.num_classes)
        
        print(f"> Trainer setup done")
    
    def _save_model(
        self, 
        epoch: int, 
        score: float
    ):
        """saves best model and deletes old one

        Args:
            epoch (int): epoch
            score (float): best score
        """
        pth_name = f"{self.model_name}-epoch_{epoch}-score_{score:.5f}.pth"
        pth_path = os.path.join(self.checkpoints_dir, pth_name)
        
        if self.best_model is not None:
            print(f"> Removing previous best model at {self.best_model}")
            os.remove(self.best_model)
        
        os.makedirs(self.checkpoints_dir, exist_ok=True)
        print(f"> Saving new best model at {pth_path}")
        torch.save(
            self.model.state_dict(),
            pth_path
        )
        self.best_model = pth_path
    
    def _get_sampled_loader(
        self,
        train: bool,
        indices: np.array,
        transform: Callable,
        shuffle: bool,
        max_samples_per_class: int
    ) -> DataLoader:
        """Sampels the dataset and returns the DataLoader

        Args:
            train (bool): train mode
            indices (np.array): indices to sample
            transform (Callable): transformations
            shuffle (bool): shuffle mode DataLoader
            max_samples_per_class (int): max samples per class

        Returns:
            DataLoader: DataLoader for sampled dataset
        """
        
        if self.verbose:
            print(f"> Sampling {'train' if train else 'validation'} dataset from cross-val {'train' if train else 'validation'} indices and setting data loader.")
        
        dataset = ImageFolderDataset(
            data_dir=self.data_dir,
            class_map=self.class_map,
            transform=transform,
            max_samples_per_class=max_samples_per_class
        )
        dataset.images = [image for i, image in enumerate(dataset.images) if i in indices]
        dataset.targets = [target for i, target in enumerate(dataset.targets) if i in indices]
        
        if self.verbose: 
            dataset.stats()
        
        sampler = ImbalancedSampler(dataset=dataset) if (train and self.imbalanced) else None
        
        return DataLoader(
            dataset=dataset,
            batch_size=self.batch_size,
            shuffle=shuffle if not train else False,
            num_workers=self.num_workers,
            pin_memory=True,
            sampler=sampler
        )
    
    @torch.no_grad()
    def _eval(self) -> Tuple:
        """evaluation iteration

        Returns:
            Dict: {accuracy: val, precision: val, recall: val, f1: val}
        """
        preds_list, target_list = [], []
        self.model.eval()
        for i, batch in enumerate(self.val_dl):
            x, target = batch
            x = x.to(self.device)
            logits = self.model(x)
            preds = np.argmax(logits.cpu().numpy(), axis=1)
            preds_list += list(preds)             
            target_list += list(target.cpu().numpy())  
        
        acc = metrics.accuracy_score(target_list, preds_list)
        precision = metrics.precision_score(target_list, preds_list, average="macro")
        recall = metrics.recall_score(target_list, preds_list, average="macro")
        f1 = metrics.f1_score(target_list, preds_list, average="macro")  
        
        print(f"Validation -> accuracy: {acc*100:.4f}% - mean precision {100*precision:.4f}% - mean recall {100*recall:.4f}% - mean F1 {100*f1:.4f}% -")
        
        return {
            "accuracy": acc,
            "precision": precision,
            "recall": recall,
            "f1": f1
        }
            
    def _train(
        self,
        data_loader: DataLoader,
        log_freq: float = 0.25
    ):
        """Training iteration

        Args:
            data_loader (DataLoader): data loader
            log_freq (float, optional): logging frequency. Defaults to 0.25.
        """
        num_samples = len(data_loader.dataset)
        num_batches = num_samples // self.batch_size
        log_step = max(int(log_freq*num_batches), 1)
        best_score = 0
        print(f"> Starting training for {self.epochs} epochs..")
        for epoch in range(self.epochs):
            self.model.train()  
            for i, batch in enumerate(data_loader):
                self.optimizer.zero_grad()
                
                x, target = batch
                x = x.to(self.device)
                target = target.to(self.device)
                logits = self.model(x)
                
                loss = self.criterion(logits, target)
                loss.backward()
                self.optimizer.step()
                
                if i % log_step == 0 and self.verbose:
                    print(f"Epoch {epoch + 1} - [{i+1}/{num_batches+1}] : train/loss {loss.item()}")
                self.lr_scheduler.step()
            print("\n")
            
            if self.val_dir is not None:
                metrics = self._eval()
                if metrics[self.metric] > best_score:
                    best_score = metrics[self.metric]
                    self._save_model(
                        epoch=epoch,
                        score=best_score
                    )
                              
    def fit(
        self, 
        cv_train_indices: np.array, 
        cv_train_labels: np.array = None
    ):  
        """runs fit() of the BaseEstimator by training a model on the dataset

        Args:
            train_indices (np.array): train indices to use
            train_labels (np.array, optional): train labels to use. Defaults to None.
        """
        
        print(f"\n ================ Start Cross Val Iteration  ================ \n")
        self._setup()
        print(f"> Trainer fit on {self.device}.. ")
        
        data_loader = self._get_sampled_loader(
            train=True,
            indices=cv_train_indices,
            transform=self.train_transform,
            shuffle=True,
            max_samples_per_class=self.max_samples_per_class
        )
        
        self._train(data_loader=data_loader)
            
    @torch.no_grad()
    def predict(
        self,
        cv_val_indices: np.array
    ) -> np.array:
        """predict() implementation of the BaseEstimator

        Args:
            cv_val_indices (np.array): validation set indices to sample from dataset

        Returns:
            np.array: predictions
        """
        probs = self.predict_proba(cv_val_indices=cv_val_indices)
        return probs.argmax(axis=1)
    
    @torch.no_grad()   
    def predict_proba(
        self,
        cv_val_indices: np.array
    ) -> np.array:
        """predict_proba() implementation of the BaseEstimator

        Args:
            cv_val_indices (np.array): validation set indices to sample from dataset

        Returns:
            np.array: predictions probabilities
        """
        
        data_loader = self._get_sampled_loader(
            train=False,
            indices=cv_val_indices,
            transform=self.val_transform,
            shuffle=False,
            max_samples_per_class=self.max_samples_per_class
        )
        
        if self.val_dir is not None:
            print(f"> Loading best model from {self.best_model}")
            self.model.load_state_dict(
                state_dict=torch.load(self.best_model)
            )
        
        print(f"> Predicting cross-val validation set probabilities")
        pred_proba = []
        self.model.eval()
        for i, batch in enumerate(data_loader):
            x, target = batch
            x = x.to(self.device)
            logits = self.model(x)
            probs = torch.softmax(logits, 1)
            pred_proba += list(probs.cpu().numpy())
        
        # increasing overall iteration
        print(f"\n ================ End Cross Val Iteration ================ \n")
        return np.array(pred_proba)
            
        
        