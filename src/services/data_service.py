"""
Data Service: orchestrates signal generation, dataset creation, and loading.
"""
import numpy as np
from typing import Dict, List, Any
import os
from .data_generator import SignalGenerator
from .data_pipeline import DatasetBuilder, DatasetSplitter, DataNormalizer, DataPersistence
from .data_loader import MLPDataset, SeqDataset, create_dataloader

class DataService:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.data_cfg = config["data"]
        self.normalizer = DataNormalizer()

    def generate_and_save_datasets(self, data_dir: str = "data"):
        os.makedirs(data_dir, exist_ok=True)
        freqs = self.data_cfg["frequencies"]
        gen = SignalGenerator(
            frequencies=freqs,
            sampling_rate=self.data_cfg["sampling_rate"],
            duration=self.data_cfg["duration_seconds"],
            noise_sigma=self.data_cfg["noise_sigma"]
        )
        
        signals = {}
        for f in freqs:
            _, clean, noisy = gen.generate_noisy(f)
            signals[f] = (clean, noisy)
            
        builder = DatasetBuilder(window_size=self.data_cfg["window_size"])
        full_dataset = builder.build_from_signals(signals, freqs)
        
        splitter = DatasetSplitter(
            val_split=self.data_cfg["val_split"],
            test_split=self.data_cfg["test_split"]
        )
        splits = splitter.split(full_dataset)
        
        # Normalize
        self.normalizer.fit(splits["train"]["noisy_samples"])
        for s in splits:
            splits[s]["noisy_samples"] = self.normalizer.transform(splits[s]["noisy_samples"])
            splits[s]["clean_samples"] = self.normalizer.transform(splits[s]["clean_samples"])
            splits[s]["target_output"] = self.normalizer.transform(splits[s]["target_output"])
            
            filepath = os.path.join(data_dir, f"{s}.npz")
            DataPersistence.save(filepath, splits[s])
            
        # Save normalizer params if needed, or simply return splits
        return splits

    def load_dataloaders(self, model_type: str, data_dir: str = "data") -> Dict[str, Any]:
        splits = {}
        for s in ["train", "val", "test"]:
            filepath = os.path.join(data_dir, f"{s}.npz")
            splits[s] = DataPersistence.load(filepath)
            
        DatasetClass = MLPDataset if model_type == "mlp" else SeqDataset
        
        dls = {}
        bs = self.config["training"]["batch_size"]
        seed = self.data_cfg["base_seed"]
        
        for s in splits:
            ds = DatasetClass(splits[s])
            dls[s] = create_dataloader(ds, batch_size=bs, shuffle=(s == "train"), seed=seed)
            
        return dls
