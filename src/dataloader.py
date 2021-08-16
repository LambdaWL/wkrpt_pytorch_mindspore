
from dataset import IEMOCAP


def get_dataloaders(platform, data_dir, batch_size):
    """
    Args:
        platform: One of ["Torch", "Mindspore"]
        data_dir: Directory storing datasets and vocabulary files
        batch_size: Number of dialogues processed each time
    """
    train_set = IEMOCAP(data_dir, "train")
    train_set.pad_dialogues()
    train_set.to_numpy()
    test_set = IEMOCAP(data_dir, "test")
    test_set.pad_dialogues()
    test_set.to_numpy()

    assert train_set.num_class == test_set.num_class
    assert train_set.feature_dim == test_set.feature_dim
    num_class = train_set.num_class
    feature_dim = train_set.feature_dim

    if platform == "PyTorch":
        from torch.utils.data import DataLoader
        from dataset import TorchIEMOCAP

        train_loader = DataLoader(
            TorchIEMOCAP(train_set),
            batch_size,
            shuffle=False,
            pin_memory=False,
        )
        test_loader = DataLoader(
            TorchIEMOCAP(test_set),
            batch_size,
            shuffle=False,
            pin_memory=False,
        )
    elif platform == "Mindspore":
        from mindspore.dataset import GeneratorDataset

        train_loader = GeneratorDataset(
            train_set,
            ["features", "labels"],
            shuffle=False
        ).batch(batch_size)
        test_loader = GeneratorDataset(
            test_set,
            ["features", "labels"],
            shuffle=False,
        ).batch(batch_size)
    else:
        raise ValueError(f"Unsupported Platform {platform}")
    
    return train_loader, test_loader, num_class, feature_dim
