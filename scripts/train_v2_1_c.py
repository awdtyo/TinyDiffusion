#!/usr/bin/env python3
"""Phase 3: Train Tiny U-Net v2.1 Configuration C [6,8,8,6]"""

import sys, os, json, numpy as np, tensorflow as tf
from datetime import datetime
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.utils import load_config, set_global_seed, ensure_dir, save_json
from src.v2_1.model_v2_1 import build_tiny_unet_v2_1
from src.diffusion import NoiseSchedule
from src.dataset import MNISTDataset

# Diffusion constants
betas = np.linspace(0.0001, 0.02, 1000, dtype=np.float32)
alphas = 1 - betas
alphas_cumprod = np.cumprod(1 - betas)
sqrt_alphas_cumprod = np.sqrt(alphas_cumprod)
sqrt_one_minus_alphas_cumprod = np.sqrt(1 - alphas_cumprod)

def prepare_batch(x_batch, y_batch):
    bs = len(x_batch)
    t = np.random.randint(0, 1000, bs, dtype=np.int32)
    n = np.random.randn(*x_batch.shape).astype(np.float32)
    
    sqrt_ab = sqrt_alphas_cumprod[t].reshape(bs, 1, 1, 1)
    sqrt_1mab = sqrt_one_minus_alphas_cumprod[t].reshape(bs, 1, 1, 1)
    x_t = sqrt_ab * x_batch + sqrt_one_minus_alphas_cumprod[t].reshape(bs, 1, 1, 1) * n
    v = sqrt_ab * n - sqrt_1mab * x_batch
    
    lo = np.eye(10, dtype=np.float32)[y_batch]
    tn = (t.astype(np.float32) / 1000).reshape(bs, 1)
    
    # Ensure correct shapes for tiling
    cm = np.repeat(lo[:, None, None, :], 28*28, axis=1).reshape(bs, 28, 28, 10)
    tm = np.repeat(tn[:, None], 28*28, axis=1).reshape(bs, 28, 28, 1)
    cond = np.concatenate([cm, tm], axis=-1)
    mi = np.concatenate([x_t, cond], axis=-1)
    return mi, v

def make_gen(x_data, y_data):
    def gen():
        idx = np.arange(len(x_data))
        while True:
            np.random.shuffle(idx)
            for i in range(0, len(idx), 64):
                bidx = idx[i:i+64]
                x_b = x_data[bidx]; y_b = y_data[bidx]
                bs = len(x_b)
                t = np.random.randint(0, 1000, bs, dtype=np.int32)
                n = np.random.randn(*x_b.shape).astype(np.float32)
                
                sqrt_ab = sqrt_alphas_cumprod[t].reshape(bs, 1, 1, 1)
                sqrt_1mab = sqrt_one_minus_alphas_cumprod[t].reshape(bs, 1, 1, 1)
                x_t = sqrt_ab * x_b + sqrt_one_minus_alphas_cumprod[t].reshape(bs, 1, 1, 1) * n
                v = sqrt_ab * n - sqrt_1mab * x_b
                
                lo = np.eye(10, dtype=np.float32)[y_b]
                tn = (t.astype(np.float32) / 1000).reshape(-1, 1)
                
                cm = np.repeat(lo[:, None, :], 28*28, axis=1).reshape(-1, 28, 28, 10)
                tm = np.repeat(tn, 28*28, axis=1).reshape(-1, 28, 28, 1)
                cond = np.concatenate([cm, tm], axis=-1)
                mi = np.concatenate([x_t, cond], axis=-1)
                
                yield mi, v
    return gen

def main():
    # Setup
    import numpy as np, tensorflow as tf
    from datetime import datetime
    import sys, os, json
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
    
    from src.utils import load_config, set_global_seed, ensure_dir, save_json
    from src.v2_1.model_v2_1 import build_tiny_unet_v2_1
    from src.diffusion import NoiseSchedule
    from src.dataset import MNISTDataset
    
    np.random.seed(42); tf.random.set_seed(42)
    
    # Diffusion constants
    betas = np.linspace(0.0001, 0.02, 1000, dtype=np.float32)
    alphas = 1 - betas
    alphas_cumprod = np.cumprod(1 - betas)
    sqrt_alphas_cumprod = np.sqrt(np.cumprod(1 - betas))
    sqrt_one_minus_alphas_cumprod = np.sqrt(1 - np.cumprod(1 - betas))
    
    # Load data
    dataset = MNISTDataset(load_config("configs/tiny_unet_v2_1.yaml"))
    x_train, y_train, x_test, y_test = dataset.load()
    x_train_proc = dataset.preprocess(x_train)
    
    indices = np.random.permutation(len(x_train_proc))
    val_idx, train_idx = indices[:6000], indices[6000:]
    x_val_proc, y_val = x_train_proc[val_idx], y_train[val_idx]
    x_train_final, y_train_final = x_train_proc[train_idx], y_train[train_idx]
    
    print("="*60)
    print("Phase 3: Training Tiny U-Net v2.1 Configuration C")
    print("="*60)
    print("Config: [6,8,8,6] — 8,011 params")
    print(f"Train: {len(x_train_final)}, Val: {len(x_val_proc)}")
    
    # Build model with correct config
    config = {
        'model': {
            'input_height': 28, 'input_width': 28, 'total_input_channels': 12,
            'num_classes': 10, 'architecture': 'tiny_unet_v2_1',
            'tiny_unet_v2_1': {
                'channels': [6, 8, 8, 6],
                'initial_conv_kernel': 3,
                'encoder': [
                    {'resolution': [28, 28], 'channels': 6, 'num_blocks': 1},
                    {'resolution': [14, 14], 'channels': 8, 'num_blocks': 1}
                ],
                'bottleneck': {'resolution': [7, 7], 'channels': 8, 'num_blocks': 1},
                'decoder': [
                    {'resolution': [14, 14], 'channels': 8, 'num_blocks': 1},
                    {'resolution': [28, 28], 'channels': 6, 'num_blocks': 1}
                ],
                'output_channels': 1,
                'residual_block': {'kernel_size': 3, 'activation': 'relu', 'use_bias': True, 'padding': 'same'},
                'downsample': {'type': 'strided_conv', 'kernel_size': 3, 'stride': 2},
                'upsample': {'type': 'bilinear_conv', 'kernel_size': 3}
            }
        }
    }
    
    cnn = build_tiny_unet_v2_1(config)
    cnn(tf.random.normal([1,28,28,12]), training=False)
    cnn.compile(optimizer=tf.keras.optimizers.Adam(0.001,0.9,0.999,1e-7), loss='mse')
    print(f"Model: {cnn.name}, Params: {cnn.count_params():,}")
    
    # Data generators
    def make_gen(x_data, y_data):
        def gen():
            idx = np.arange(len(x_data))
            while True:
                np.random.shuffle(idx)
                for i in range(0, len(idx), 64):
                    bidx = idx[i:i+64]
                    x_b = x_data[bidx]; y_b = y_data[bidx]
                    bs = len(x_b)
                    t = np.random.randint(0, 1000, bs, dtype=np.int32)
                    n = np.random.randn(*x_b.shape).astype(np.float32)
                    
                    sqrt_ab = sqrt_alphas_cumprod[t].reshape(bs, 1, 1, 1)
                    sqrt_1mab = sqrt_one_minus_alphas_cumprod[t].reshape(bs, 1, 1, 1)
                    x_t = sqrt_ab * x_b + sqrt_one_minus_alphas_cumprod[t].reshape(bs, 1, 1, 1) * n
                    v = sqrt_ab * n - sqrt_1mab * x_b
                    
                    lo = np.eye(10, dtype=np.float32)[y_b]
                    tn = (t.astype(np.float32)/1000).reshape(-1, 1)
                    
                    cm = np.repeat(lo[:, None, :], 28*28, axis=1).reshape(-1, 28, 28, 10)
                    tm = np.repeat(tn, 28*28, axis=1).reshape(-1, 28, 28, 1)
                    cond = np.concatenate([cm, tm], axis=-1)
                    mi = np.concatenate([x_t, cond], axis=-1)
                    
                    yield mi, v
        return gen
    
    train_gen = make_gen(x_train_final, y_train_final)
    val_gen = make_gen(x_val_proc, y_val)
    
    train_ds = tf.data.Dataset.from_generator(
        train_gen, output_signature=(
            tf.TensorSpec((None,28,28,12), tf.float32), tf.TensorSpec((None,28,28,1), tf.float32))
    ).prefetch(tf.data.AUTOTUNE)
    
    val_ds = tf.data.Dataset.from_generator(
        val_gen, output_signature=(
            tf.TensorSpec((None,28,28,12), tf.float32), tf.TensorSpec((None,28,28,1), tf.float32))
    ).prefetch(tf.data.AUTOTUNE)
    
    # Model
    config = {
        'model': {
            'input_height': 28, 'input_width': 28, 'total_input_channels': 12,
            'num_classes': 10, 'architecture': 'tiny_unet_v2_1',
            'tiny_unet_v2_1': {
                'channels': [6, 8, 8, 6],
                'initial_conv_kernel': 3,
                'encoder': [
                    {'resolution': [28, 28], 'channels': 6, 'num_blocks': 1},
                    {'resolution': [14, 14], 'channels': 8, 'num_blocks': 1}
                ],
                'bottleneck': {'resolution': [7, 7], 'channels': 8, 'num_blocks': 1},
                'decoder': [
                    {'resolution': [14, 14], 'channels': 8, 'num_blocks': 1},
                    {'resolution': [28, 28], 'channels': 6, 'num_blocks': 1}
                ],
                'output_channels': 1,
                'residual_block': {'kernel_size': 3, 'activation': 'relu', 'use_bias': True, 'padding': 'same'},
                'downsample': {'type': 'strided_conv', 'kernel_size': 3, 'stride': 2},
                'upsample': {'type': 'bilinear_conv', 'kernel_size': 3}
            }
        }
    }
    
    cnn = build_tiny_unet_v2_1(config)
    cnn(tf.random.normal([1,28,28,12]), training=False)
    cnn.compile(optimizer=tf.keras.optimizers.Adam(0.001,0.9,0.999,1e-7), loss='mse')
    print(f"Model: {cnn.name}, Params: {cnn.count_params():,}")
    
    # Callbacks
    callbacks = [
        tf.keras.callbacks.ModelCheckpoint("artifacts/checkpoints_tiny_unet_v2_1_c/best_model.keras", monitor='val_loss', save_best_only=True, verbose=1),
        tf.keras.callbacks.ModelCheckpoint("artifacts/checkpoints_tiny_unet_v2_1_c/last_model.keras", monitor='val_loss', save_best_only=False, verbose=1),
        tf.keras.callbacks.EarlyStopping('val_loss', patience=10, restore_best_weights=True, verbose=1),
        tf.keras.callbacks.CSVLogger("artifacts/logs_tiny_unet_v2_1_c/training_log.csv"),
        tf.keras.callbacks.LearningRateScheduler(lambda e, lr: 1e-3*(e+1)/5 if e<5 else 1e-5+(1e-3-1e-5)*0.5*(1+np.cos(np.pi*(e-5)/45)), verbose=0)
    ]
    
    hist = cnn.fit(
        tf.data.Dataset.from_generator(
            train_gen, output_signature=(
                tf.TensorSpec((None,28,28,12), tf.float32), tf.TensorSpec((None,28,28,1), tf.float32))
        ).prefetch(tf.data.AUTOTUNE),
        validation_data=tf.data.Dataset.from_generator(
            val_gen, output_signature=(
                tf.TensorSpec((None,28,28,12), tf.float32), tf.TensorSpec((None,28,28,1), tf.float32))
        ).prefetch(tf.data.AUTOTUNE),
        epochs=50,
        steps_per_epoch=54000//64,
        validation_steps=6000//64,
        callbacks=callbacks,
        verbose=1
    )
    
    # Save
    cnn.save("artifacts/checkpoints_tiny_unet_v2_1_c/final_model.keras")
    cnn.save("artifacts/checkpoints_tiny_unet_v2_1_c/inference_model.keras")
    save_json({'loss': hist.history['loss'], 'val_loss': hist.history['val_loss'], 'params': 8011}, "artifacts/logs_tiny_unet_v2_1_c/training_history.json")
    print(f"Best val_loss: {min(hist.history['val_loss']):.6f}")


if __name__ == "__main__":
    import numpy as np, tensorflow as tf
    from datetime import datetime
    import sys, os, json
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
    
    from src.utils import load_config, set_global_seed, ensure_dir, save_json
    from src.v2_1.model_v2_1 import build_tiny_unet_v2_1
    from src.diffusion import NoiseSchedule
    from src.dataset import MNISTDataset
    
    np.random.seed(42); tf.random.set_seed(42)
    
    # Diffusion constants
    betas = np.linspace(0.0001, 0.02, 1000, dtype=np.float32)
    alphas = 1 - betas
    alphas_cumprod = np.cumprod(1 - betas)
    sqrt_alphas_cumprod = np.sqrt(np.cumprod(1 - betas))
    sqrt_one_minus_alphas_cumprod = np.sqrt(1 - np.cumprod(1 - betas))
    
    # Load data
    dataset = MNISTDataset(load_config("configs/tiny_unet_v2_1.yaml"))
    x_train, y_train, x_test, y_test = dataset.load()
    x_train_proc = dataset.preprocess(x_train)
    
    indices = np.random.permutation(len(x_train_proc))
    val_idx, train_idx = indices[:6000], indices[6000:]
    x_val_proc, y_val = x_train_proc[val_idx], y_train[val_idx]
    x_train_final, y_train_final = x_train_proc[train_idx], y_train[train_idx]
    
    print("="*60)
    print("Phase 3: Training Tiny U-Net v2.1 Configuration C")
    print("="*60)
    print("Config: [6,8,8,6] — 8,011 params")
    print(f"Train: {len(x_train_final)}, Val: {len(x_val_proc)}")
    
    main()