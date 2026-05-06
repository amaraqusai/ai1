"""
Data pipeline: building, splitting, normalization, and persistence.
"""
import numpy as np
from typing import Dict, List, Tuple
from ..shared.gatekeeper import get_gatekeeper

def one_hot_encode(freq_idx: int, num_classes: int = 4) -> np.ndarray:
    oh = np.zeros(num_classes, dtype=np.float32)
    oh[freq_idx] = 1.0
    return oh

class DatasetBuilder:
    def __init__(self, window_size: int = 10):
        self.window_size = window_size

    def build_from_signals(self, signals_dict: Dict[int, Tuple[np.ndarray, np.ndarray]], 
                           freq_list: List[float]) -> Dict[str, np.ndarray]:
        X_noisy, X_clean, y_target, labels = [], [], [], []
        
        if not freq_list:
            raise ValueError("Frequency list cannot be empty.")
            
        for idx, f in enumerate(freq_list):
            if f not in signals_dict:
                continue
            clean_sig, noisy_sig = signals_dict[f]
            label = one_hot_encode(idx, len(freq_list))
            
            n_samples = len(clean_sig)
            if self.window_size > n_samples:
                raise ValueError("window_size cannot be strictly greater than total_samples.")
                
            for i in range(n_samples - self.window_size):
                X_noisy.append(noisy_sig[i:i+self.window_size])
                X_clean.append(clean_sig[i:i+self.window_size])
                y_target.append([clean_sig[i+self.window_size]])
                labels.append(label)
                
        return {
            "noisy_samples": np.array(X_noisy, dtype=np.float32),
            "clean_samples": np.array(X_clean, dtype=np.float32),
            "target_output": np.array(y_target, dtype=np.float32),
            "frequency_label": np.array(labels, dtype=np.float32)
        }

class DatasetSplitter:
    def __init__(self, val_split: float = 0.15, test_split: float = 0.15):
        self.val_split = val_split
        self.test_split = test_split

    def split(self, dataset: Dict[str, np.ndarray]) -> Dict[str, Dict[str, np.ndarray]]:
        if len(dataset.get("frequency_label", [])) == 0:
            raise ValueError("Empty dataset list provided to split.")
            
        # Stratified by frequency label
        labels = dataset["frequency_label"]
        unique_labels = np.unique(labels, axis=0)
        
        splits = {"train": {}, "val": {}, "test": {}}
        for key in dataset:
            splits["train"][key] = []
            splits["val"][key] = []
            splits["test"][key] = []
            
        for ul in unique_labels:
            mask = np.all(labels == ul, axis=1)
            indices = np.where(mask)[0]
            np.random.shuffle(indices)
            
            n = len(indices)
            n_val = int(n * self.val_split)
            n_test = int(n * self.test_split)
            n_train = n - n_val - n_test
            
            idx_train = indices[:n_train]
            idx_val = indices[n_train:n_train+n_val]
            idx_test = indices[n_train+n_val:]
            
            for key in dataset:
                splits["train"][key].extend(dataset[key][idx_train])
                splits["val"][key].extend(dataset[key][idx_val])
                splits["test"][key].extend(dataset[key][idx_test])
                
        for s in splits:
            for k in splits[s]:
                splits[s][k] = np.array(splits[s][k])
        return splits

class DataNormalizer:
    def __init__(self):
        self.mean = 0.0
        self.std = 1.0
        
    def fit(self, noisy_samples: np.ndarray):
        self.mean = np.mean(noisy_samples)
        self.std = np.std(noisy_samples)
        if self.std < 1e-8:
            self.std = 1.0
            
    def transform(self, data: np.ndarray) -> np.ndarray:
        return (data - self.mean) / self.std

class DataPersistence:
    @staticmethod
    def save(filepath: str, data: Dict[str, np.ndarray]):
        gk = get_gatekeeper("file_io")
        gk.execute(np.savez_compressed, filepath, **data)

    @staticmethod
    def load(filepath: str) -> Dict[str, np.ndarray]:
        gk = get_gatekeeper("file_io")
        def _load():
            with np.load(filepath) as loaded:
                return {k: loaded[k] for k in loaded.files}
        return gk.execute(_load)
