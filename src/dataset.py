"""
MNIST Dataset handling for TinyDiffusion-ESP32.
"""
import os
import numpy as np
import tensorflow as tf
from typing import Tuple, Optional, Dict, Any
from src.utils import normalize_to_neg_one_one, one_hot_encode, ensure_dir


class MNISTDataset:
    """MNIST dataset loader and preprocessor for diffusion training."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.dataset_config = config.get('dataset', {})
        self.cache_dir = self.dataset_config.get('cache_dir', 'artifacts/data')
        self.normalize_range = self.dataset_config.get('normalize_range', [-1, 1])
        self.auto_download = self.dataset_config.get('auto_download', True)
        self.shuffle_buffer = self.dataset_config.get('shuffle_buffer', 10000)
        self.prefetch_buffer = self.dataset_config.get('prefetch_buffer', 2)
        
        ensure_dir(self.cache_dir)
        
        self.x_train = None
        self.y_train = None
        self.x_test = None
        self.y_test = None
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None
    
    def load(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Load MNIST dataset from Keras or download if needed."""
        print("Loading MNIST dataset...")
        
        # Use Keras built-in MNIST loader
        (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
        
        # Validate dimensions
        assert x_train.shape == (60000, 28, 28), f"Unexpected train shape: {x_train.shape}"
        assert x_test.shape == (10000, 28, 28), f"Unexpected test shape: {x_test.shape}"
        assert y_train.shape == (60000,), f"Unexpected train label shape: {y_train.shape}"
        assert y_test.shape == (10000,), f"Unexpected test label shape: {y_test.shape}"
        
        print(f"Train images: {x_train.shape}, Train labels: {y_train.shape}")
        print(f"Test images: {x_test.shape}, Test labels: {y_test.shape}")
        print(f"Classes: {np.unique(y_train)}")
        print(f"Pixel range: [{x_train.min()}, {x_train.max()}]")
        
        self.x_train = x_train
        self.y_train = y_train
        self.x_test = x_test
        self.y_test = y_test
        
        return x_train, y_train, x_test, y_test
    
    def preprocess(self, images: np.ndarray) -> np.ndarray:
        """Preprocess images: add channel dim, normalize to [-1, 1]."""
        # Add channel dimension: [N, 28, 28] -> [N, 28, 28, 1]
        images = images[..., np.newaxis].astype(np.float32)
        
        # Normalize to [-1, 1]
        images = normalize_to_neg_one_one(images)
        
        return images
    
    def prepare_datasets(self, validation_split: float = 0.1, 
                         batch_size: int = 64, seed: int = 42) -> Tuple[tf.data.Dataset, tf.data.Dataset, tf.data.Dataset]:
        """Create tf.data.Dataset pipelines for train, validation, and test."""
        if self.x_train is None:
            self.load()
        
        # Preprocess
        x_train = self.preprocess(self.x_train)
        x_test = self.preprocess(self.x_test)
        y_train = self.y_train
        y_test = self.y_test
        
        # Create validation split
        num_val = int(len(x_train) * validation_split)
        indices = np.arange(len(x_train))
        np.random.seed(seed)
        np.random.shuffle(indices)
        
        val_indices = indices[:num_val]
        train_indices = indices[num_val:]
        
        x_val = x_train[val_indices]
        y_val = y_train[val_indices]
        x_train = x_train[train_indices]
        y_train = y_train[train_indices]
        
        print(f"Train: {len(x_train)}, Val: {len(x_val)}, Test: {len(x_test)}")
        
        # Create datasets
        def create_dataset(images, labels, shuffle=True):
            dataset = tf.data.Dataset.from_tensor_slices((images, labels))
            if shuffle:
                dataset = dataset.shuffle(self.shuffle_buffer, seed=seed, reshuffle_each_iteration=True)
            dataset = dataset.batch(batch_size, drop_remainder=True)
            dataset = dataset.prefetch(self.prefetch_buffer)
            return dataset
        
        self.train_dataset = create_dataset(x_train, y_train, shuffle=True)
        self.val_dataset = create_dataset(x_val, y_val, shuffle=False)
        self.test_dataset = create_dataset(x_test, y_test, shuffle=False)
        
        return self.train_dataset, self.val_dataset, self.test_dataset
    
    def get_sample_batch(self, batch_size: int = 16, split: str = 'train') -> Tuple[np.ndarray, np.ndarray]:
        """Get a sample batch for visualization."""
        if split == 'train' and self.x_train is not None:
            indices = np.random.choice(len(self.x_train), batch_size, replace=False)
            return self.preprocess(self.x_train[indices]), self.y_train[indices]
        elif split == 'test' and self.x_test is not None:
            indices = np.random.choice(len(self.x_test), batch_size, replace=False)
            return self.preprocess(self.x_test[indices]), self.y_test[indices]
        else:
            raise ValueError(f"Invalid split: {split}")
    
    def get_class_samples(self, class_id: int, num_samples: int = 5, split: str = 'test') -> np.ndarray:
        """Get samples of a specific class."""
        if split == 'train':
            x, y = self.x_train, self.y_train
        else:
            x, y = self.x_test, self.y_test
        
        if x is None:
            self.load()
        
        indices = np.where(y == class_id)[0]
        selected = np.random.choice(indices, min(num_samples, len(indices)), replace=False)
        return self.preprocess(x[selected])
    
    def visualize_samples(self, num_samples: int = 16, save_path: Optional[str] = None) -> None:
        """Visualize sample images from the dataset."""
        import matplotlib.pyplot as plt
        
        if self.x_train is None:
            self.load()
        
        indices = np.random.choice(len(self.x_train), num_samples, replace=False)
        images = self.x_train[indices]
        labels = self.y_train[indices]
        
        rows = int(np.ceil(np.sqrt(num_samples)))
        cols = int(np.ceil(num_samples / rows))
        
        fig, axes = plt.subplots(rows, cols, figsize=(cols * 2, rows * 2))
        axes = axes.flatten() if num_samples > 1 else [axes]
        
        for i, (img, label) in enumerate(zip(images, labels)):
            axes[i].imshow(img, cmap='gray')
            axes[i].set_title(f"Label: {label}")
            axes[i].axis('off')
        
        for i in range(num_samples, len(axes)):
            axes[i].axis('off')
        
        plt.tight_layout()
        
        if save_path:
            ensure_dir(os.path.dirname(save_path))
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Saved visualization to {save_path}")
        
        plt.show()


def create_tf_dataset(images: np.ndarray, labels: np.ndarray, 
                      batch_size: int = 64, shuffle: bool = True,
                      shuffle_buffer: int = 10000, seed: int = 42) -> tf.data.Dataset:
    """Create a tf.data.Dataset from numpy arrays."""
    dataset = tf.data.Dataset.from_tensor_slices((images, labels))
    if shuffle:
        dataset = dataset.shuffle(shuffle_buffer, seed=seed, reshuffle_each_iteration=True)
    dataset = dataset.batch(batch_size, drop_remainder=True)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    return dataset


def prepare_diffusion_batch(images: np.ndarray, labels: np.ndarray,
                            timesteps: np.ndarray, num_classes: int = 10,
                            normalize_timestep: bool = True,
                            num_timesteps: int = 100) -> Tuple[np.ndarray, np.ndarray]:
    """
    Prepare a batch for diffusion training.
    
    Returns:
        model_input: [batch, 28, 28, 12] - noisy image + conditioning
        target_noise: [batch, 28, 28, 1] - ground truth noise
    """
    from src.diffusion import forward_diffusion, broadcast_conditioning
    from src.utils import one_hot_encode
    
    batch_size = images.shape[0]
    
    # Generate random noise
    noise = np.random.randn(*images.shape).astype(np.float32)
    
    # Apply forward diffusion
    noisy_images = forward_diffusion(images, timesteps, noise, num_timesteps)
    
    # Prepare conditioning
    class_onehot = one_hot_encode(labels, num_classes)
    timestep_normalized = timesteps.astype(np.float32).reshape(-1, 1)
    if normalize_timestep:
        timestep_normalized = timestep_normalized / num_timesteps
    
    conditioning = broadcast_conditioning(class_onehot, timestep_normalized, 28, 28)
    
    # Concatenate
    model_input = np.concatenate([noisy_images, conditioning], axis=-1)
    
    return model_input, noise


if __name__ == "__main__":
    # Test dataset loading
    config = {
        'dataset': {
            'cache_dir': 'artifacts/data',
            'normalize_range': [-1, 1],
            'auto_download': True,
            'shuffle_buffer': 10000,
            'prefetch_buffer': 2
        }
    }
    
    dataset = MNISTDataset(config)
    x_train, y_train, x_test, y_test = dataset.load()
    
    # Test preprocessing
    x_train_processed = dataset.preprocess(x_train[:10])
    print(f"Processed shape: {x_train_processed.shape}")
    print(f"Processed range: [{x_train_processed.min():.3f}, {x_train_processed.max():.3f}]")
    
    # Test dataset creation
    train_ds, val_ds, test_ds = dataset.prepare_datasets(validation_split=0.1, batch_size=32)
    
    for batch_images, batch_labels in train_ds.take(1):
        print(f"Batch images: {batch_images.shape}, range: [{batch_images.numpy().min():.3f}, {batch_images.numpy().max():.3f}]")
        print(f"Batch labels: {batch_labels.shape}")
    
    print("Dataset module test passed!")