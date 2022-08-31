# **mollie**
**mollie** (i**M**age n**O**isy **L**abe**L** f**I**nd**E**r) is a tool that leverages [cleanlab](https://github.com/cleanlab/cleanlab) to identify noisy label in your image classification dataset

## **How to use it**
This repository provides a bunch of wrappers to find noisy labels in image classification datasets with PyTorch (cpu, cuda, and mps training supported!!)

```python

from mollie.trainer import Trainer # sklearn BaseEstimator wrapper to train NN with torch (cpu, cuda, mps device support)
from mollie.core import Mollie # cleanlab wrapper
from mollie.dataset import ImageFolderDataset # image classification dataset

class_map = {
    0: 'class_1', 
    1: 'class_2', 
    ...
}, # you can also aggregate classes (e.g. {0: ['class_1', 'class_2'], 1: ['class_3'], ..})

trainer = Trainer(
    data_dir="PATH/TO/YOUR/DATA_DIR",
    model_name="resnet18", # timm model (use a tiny model - empirical)
    class_map=class_map,
    input_size=224,
    val_dir="PATH/TO/YOUR/VAL_DATA_DIR",
    epochs=10, # use few epochs (empirical result)
    device="cuda" # (cpu, mps, cuda)
)

dataset = ImageFolderDataset(
    data_dir="PATH/TO/YOUR/DATA_DIR",
    class_map=class_map
)

mollie = Mollie(
    trainer=trainer,
    dataset=dataset,
    cv_n_folds=5, # number of cross-val runs to find errors (each run is a complete model training)
    filter_by="prune_by_class" # filter_by method to find label errors
)

errors = mollie.start()

# errors = [
#    {
#       "file_name": "path/to/image", -> image path with label errors
#       "ground_truth": "class_1", -> ground truth label (noisy label)
#       "predicted": "class_2", -> suggested new label
#       "prediction_score": 0.57 -> suggested new label model score
#    },
#    {
#       ...
#    },
#    
# ]
```

### **Custom Dataset Label Errors**
You can provide your own dataset and run the *find_noisy_labels.py* script to save label errors. 

```
python find_noisy_labels.py --config config/cleanlab/config.yml --data-dir PATH/TO/YOUR/DATASET --val-dir PATH/TO/YOUR/VALIDATION_DATASET --output-dir PATH/TO/YOUR/OUTPUT_DIR
```

Please have a look at the **config.yml** provided (*config/config.yml*) to be familiar with the wrappers' params.

## **TO-DOs**

[ ] Add imagenette noisy demo notebook


