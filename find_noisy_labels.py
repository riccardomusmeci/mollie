import os
import json
import argparse
from mollie.core import Mollie
from mollie.io import load_config
from mollie.trainer import Trainer
from mollie.dataset import ImageFolderDataset

def parse_args() -> argparse.Namespace:
    
    parser = argparse.ArgumentParser("Training config")
    
    parser.add_argument(
        "--config",
        default="config/config.yml",
        type=str,
        required=False,
        help="path to the YAML configuration file"
    )
    
    parser.add_argument(
        "--output-dir",
        default="output/cleanlab",
        type=str,
        help="where to save output of the error label findings."
    )
    
    parser.add_argument(
        "--data-dir",
        metavar="N",
        help="Input data dir path to find errors into."
    )
    
    parser.add_argument(
        "--val-dir",
        metavar="N",
        default=None,
        help="Extra validation data dir"
    )
    
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    
    args = parse_args()
    config = load_config(args.config)
    
    trainer = Trainer(
        data_dir=args.data_dir,
        val_dir=args.val_dir,
        **config["trainer"]
    )
    
    dataset = ImageFolderDataset(
        data_dir=args.data_dir,
        class_map=config["trainer"]["class_map"]
    )

    mollie = Mollie(
        trainer=trainer,
        dataset=dataset,
        **config["cleanlab"]
    )
    
    errors = mollie.start()
    
    # Saving outputs
    os.makedirs(args.output_dir, exist_ok=True)
    output_filepath = os.path.join(args.output_dir, "errors.json")
    print(f"Saving output file with errors at {output_filepath}")
    with open(output_filepath, "w") as f:
        json.dump(errors, f, indent=4)
        