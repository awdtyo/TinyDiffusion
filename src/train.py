"""
Training pipeline for TinyDiffusion-ESP32.
"""
import os
import json
import numpy as np
import tensorflow as tf
from tensorflow.keras import callbacks, optimizers
from typing import Dict, Any, Optional, Tuple, List
from src.utils import load_config, set_global_seed, ensure_dir, format_time, save_json
from src.dataset import MNISTDataset, prepare_diffusion_batch
from src.diffusion import NoiseSchedule, forward_diffusion_tf
from src.model import create_model, build_inference_model
from src.scheduler import DDIMScheduler


class DiffusionTrainer:
    """Handles the complete training pipeline for the conditional diffusion model."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.train_config = config.get('training', {})
        self.seed = self.train_config.get('seed', 42)
        
        set_global_seed(self.seed)
        
        # Initialize components
        self.noise_schedule = NoiseSchedule(config)
        self.dataset = MNISTDataset(config)
        self.model = None
        self.inference_model = None
        self.history = None
        
        # Training parameters
        self.batch_size = self.train_config.get('batch_size', 64)
        self.epochs = self.train_config.get('epochs', 50)
        self.learning_rate = self.train_config.get('learning_rate', 0.001)
        self.validation_split = self.train_config.get('validation_split', 0.1)
        
        # Paths
        self.checkpoint_dir = 'artifacts/checkpoints'
        self.log_dir = self.train_config.get('tensorboard', {}).get('log_dir', 'artifacts/logs')
        ensure_dir(self.checkpoint_dir)
        ensure_dir(self.log_dir)
    
    def prepare_data(self) -> Tuple[tf.data.Dataset, tf.data.Dataset]:
        """Load and prepare training and validation datasets."""
        train_ds, val_ds, _ = self.dataset.prepare_datasets(
            validation_split=self.validation_split,
            batch_size=self.batch_size,
            seed=self.seed
        )
        return train_ds, val_ds
    
    def build_model(self) -> None:
        """Build and compile the model."""
        # Create diffusion model
        self.model = create_model(self.config, self.noise_schedule)
        
        # Also create inference model for export
        self.inference_model = build_inference_model(self.config)
        
        # Optimizer
        optimizer = optimizers.Adam(
            learning_rate=self.learning_rate,
            beta_1=self.train_config.get('adam_beta1', 0.9),
            beta_2=self.train_config.get('adam_beta2', 0.999),
            epsilon=self.train_config.get('adam_epsilon', 1e-7)
        )
        
        # Compile with custom training step
        self.model.compile(optimizer=optimizer)
        
        print("Model built and compiled.")
        from src.utils import print_model_summary
        print_model_summary(self.inference_model)
    
    def get_callbacks(self) -> List[callbacks.Callback]:
        """Get training callbacks."""
        callback_list = []
        
        # Model checkpoint
        if self.train_config.get('checkpoint', {}).get('enabled', True):
            ckpt_config = self.train_config['checkpoint']
            callback_list.append(callbacks.ModelCheckpoint(
                filepath=os.path.join(self.checkpoint_dir, 'best_model.keras'),
                monitor=ckpt_config.get('monitor', 'val_loss'),
                mode=ckpt_config.get('mode', 'min'),
                save_best_only=ckpt_config.get('save_best_only', True),
                save_weights_only=False,
                verbose=1
            ))
            
            # Also save last epoch
            callback_list.append(callbacks.ModelCheckpoint(
                filepath=os.path.join(self.checkpoint_dir, 'last_model.keras'),
                monitor=ckpt_config.get('monitor', 'val_loss'),
                mode=ckpt_config.get('mode', 'min'),
                save_best_only=False,
                save_weights_only=False,
                verbose=0
            ))
        
        # Early stopping
        if self.train_config.get('early_stopping', {}).get('enabled', True):
            es_config = self.train_config['early_stopping']
            callback_list.append(callbacks.EarlyStopping(
                monitor=es_config.get('monitor', 'val_loss'),
                patience=es_config.get('patience', 10),
                restore_best_weights=es_config.get('restore_best_weights', True),
                verbose=1
            ))
        
        # Learning rate scheduling
        if self.train_config.get('lr_schedule', {}).get('enabled', True):
            lr_config = self.train_config['lr_schedule']
            if lr_config.get('type') == 'cosine_decay':
                callback_list.append(callbacks.LearningRateScheduler(
                    self._cosine_decay_schedule, verbose=0))
            elif lr_config.get('type') == 'reduce_on_plateau':
                callback_list.append(callbacks.ReduceLROnPlateau(
                    monitor='val_loss',
                    factor=0.5,
                    patience=5,
                    min_lr=lr_config.get('min_lr', 1e-5),
                    verbose=1
                ))
        
        # TensorBoard
        if self.train_config.get('tensorboard', {}).get('enabled', True):
            callback_list.append(callbacks.TensorBoard(
                log_dir=self.log_dir,
                histogram_freq=1,
                write_graph=True,
                update_freq='epoch'
            ))
        
        # CSV Logger
        callback_list.append(callbacks.CSVLogger(
            os.path.join(self.log_dir, 'training_log.csv')))
        
        return callback_list
    
    def _cosine_decay_schedule(self, epoch: int) -> float:
        """Cosine decay learning rate schedule with warmup."""
        lr_config = self.train_config.get('lr_schedule', {})
        warmup_epochs = lr_config.get('warmup_epochs', 5)
        min_lr = lr_config.get('min_lr', 1e-5)
        base_lr = self.learning_rate
        
        if epoch < warmup_epochs:
            # Linear warmup
            return base_lr * (epoch + 1) / warmup_epochs
        else:
            # Cosine decay
            progress = (epoch - warmup_epochs) / (self.epochs - warmup_epochs)
            return min_lr + 0.5 * (base_lr - min_lr) * (1 + np.cos(np.pi * progress))
    
    def train(self, resume_from: Optional[str] = None) -> tf.keras.callbacks.History:
        """Run the training loop."""
        if self.model is None:
            self.build_model()
        
        # Load weights if resuming
        if resume_from and os.path.exists(resume_from):
            print(f"Resuming from {resume_from}")
            self.model.load_weights(resume_from)
        
        # Prepare data
        train_ds, val_ds = self.prepare_data()
        
        # Get callbacks
        callbacks_list = self.get_callbacks()
        
        print(f"Starting training for {self.epochs} epochs...")
        print(f"Batch size: {self.batch_size}")
        print(f"Learning rate: {self.learning_rate}")
        
        # Train
        self.history = self.model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=self.epochs,
            callbacks=callbacks_list,
            verbose=1
        )
        
        # Save final model
        final_model_path = os.path.join(self.checkpoint_dir, 'final_model.keras')
        self.model.save(final_model_path)
        print(f"Final model saved to {final_model_path}")
        
        # Save inference model
        inference_path = os.path.join(self.checkpoint_dir, 'inference_model.keras')
        self.inference_model.save(inference_path)
        print(f"Inference model saved to {inference_path}")
        
        # Save training history
        self.save_history()
        
        return self.history
    
    def save_history(self) -> None:
        """Save training history to JSON."""
        if self.history is None:
            return
        
        history_dict = {}
        for key, values in self.history.history.items():
            history_dict[key] = [float(v) for v in values]
        
        history_path = os.path.join(self.log_dir, 'training_history.json')
        save_json(history_dict, history_path)
        print(f"Training history saved to {history_path}")
    
    def load_history(self, history_path: str) -> Dict[str, List[float]]:
        """Load training history from JSON."""
        with open(history_path, 'r') as f:
            return json.load(f)


def train_diffusion_model(config_path: str = "configs/default.yaml",
                          resume_from: Optional[str] = None) -> Tuple[tf.keras.Model, Dict]:
    """
    Main training entry point.
    
    Returns:
        Trained inference model and training history
    """
    config = load_config(config_path)
    trainer = DiffusionTrainer(config)
    history = trainer.train(resume_from=resume_from)
    return trainer.inference_model, trainer.history.history


def load_trained_model(checkpoint_path: str, config_path: str = "configs/default.yaml") -> tf.keras.Model:
    """Load a trained model from checkpoint."""
    config = load_config(config_path)
    model = build_inference_model(config)
    model.load_weights(checkpoint_path)
    return model


if __name__ == "__main__":
    # Test training setup (without actual training)
    config = load_config("configs/default.yaml")
    trainer = DiffusionTrainer(config)
    
    # Test data preparation
    train_ds, val_ds = trainer.prepare_data()
    print(f"Train dataset: {train_ds}")
    print(f"Val dataset: {val_ds}")
    
    # Test model building
    trainer.build_model()
    
    print("Training module test passed!")