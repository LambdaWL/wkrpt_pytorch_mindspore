import os
import argparse
import random

# Temporarily
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import numpy as np

from config import Config
from dataloader import get_dataloaders


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--platform", choices=["PyTorch", "Mindspore"])
    parser.add_argument("--epochs", type=int, default=10, help="Number of epochs.")
    parser.add_argument("--bsz", type=int, default=8, help="Batch size.")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate.")
    parser.add_argument("--dropout", type=float, default=0.0, help="Dropout rate.")
    parser.add_argument("--device", choices=["CPU", "GPU"], help="Using CPU or GPU.")
    parser.add_argument(
        "--data_dir", type=str, default="../data/",
        help="Directory for storing datasets and vocabulary files.",
    )
    parser.add_argument("--seed", type=int, default=100)

    args = parser.parse_args()
    print(args)

    return args


def configure_platform(seed, platform, device):
    random.seed(seed)
    np.random.seed(seed)

    # GPU Checking
    if args.device == "GPU":
        if not os.getenv("CUDA_VISIBLE_DEVICES"):
            raise ValueError("No GPU device is provided. Please set CUDA_VISIBLE_DEVICES.")
        print(f"Running on GPU {os.environ['CUDA_VISIBLE_DEVICES']}.")
    else:
        print(f"Running on CPU.")

    # Platform-specific Configurations
    if platform == "PyTorch":
        from torch import manual_seed
        from model_torch import TorchModel
        from train_torch import train_torch_model, test_torch_model

        manual_seed(seed)
        model_cls = TorchModel
        train_fn = train_torch_model
        test_fn = test_torch_model
    elif platform == "Mindspore":
        from mindspore import context, set_seed
        from model_ms import MindsporeModel
        from train_ms import train_mindspore_model, test_mindspore_model

        set_seed(seed)
        context.set_context(
            mode=context.GRAPH_MODE,  # DEBUG 
            device_target=device, 
            device_id=int(os.getenv("CUDA_VISIBLE_DEVICES", "0"))
        )
        model_cls = MindsporeModel
        train_fn = train_mindspore_model
        test_fn = test_mindspore_model
    else:
        raise ValueError(f"Unknown platform: {platform}")
    
    return model_cls, train_fn, test_fn


if __name__ == "__main__":
    args = parse_arguments()
    model_cls, train_fn, test_fn = configure_platform(args.seed, args.platform, args.device)

    train_loader, test_loader, num_class, feature_dim = (
        get_dataloaders(args.platform, args.data_dir, args.bsz)
    )

    model_config = Config(num_class, feature_dim, dropout=args.dropout)
    model = model_cls(model_config)

    train_fn(model, train_loader, args.device, args.epochs, args.lr)
    test_fn(model, test_loader, args.device)
