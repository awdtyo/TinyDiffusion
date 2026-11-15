"""
Conditional CNN model for TinyDiffusion-ESP32.
Implements the tiny noise prediction network with class and timestep conditioning.
"""
import tensorflow as tf
from tensorflow.keras import layers, Model, Input
from typing import Dict, Any, List, Tuple, Optional
from src.utils import count_parameters, print_model_summary


def build_conditional_cnn(config: Dict[str, Any]) -> Model:
    """
    Build the conditional CNN noise prediction network.
    
    Architecture:
    Input: 28x28x12 (noisy image 1ch + 10 class maps + 1 timestep map)
    
    1. Conv2D: 8 filters, 3x3, SAME, ReLU
    2. Conv2D: 8 filters, 3x3, SAME, ReLU
    3. Conv2D: 12 filters, 3x3, SAME, ReLU
    4. Conv2D: 8 filters, 3x3, SAME, ReLU
    5. Conv2D: 1 filter, 3x3, SAME, Linear
    
    Output: 28x28x1 (predicted noise)
    """
    model_config = config.get('model', {})
    
    input_height = model_config.get('input_height', 28)
    input_width = model_config.get('input_width', 28)
    total_input_channels = model_config.get('total_input_channels', 12)
    conv_layers_config = model_config.get('conv_layers', [
        {'filters': 8, 'kernel_size': 3, 'activation': 'relu'},
        {'filters': 8, 'kernel_size': 3, 'activation': 'relu'},
        {'filters': 12, 'kernel_size': 3, 'activation': 'relu'},
        {'filters': 8, 'kernel_size': 3, 'activation': 'relu'},
        {'filters': 1, 'kernel_size': 3, 'activation': 'linear'}
    ])
    use_bias = model_config.get('use_bias', True)
    padding = model_config.get('padding', 'same')
    
    # Input layer
    inputs = Input(shape=(input_height, input_width, total_input_channels), name='model_input')
    
    x = inputs
    
    # Build convolutional layers
    for i, layer_config in enumerate(conv_layers_config):
        filters = layer_config['filters']
        kernel_size = layer_config['kernel_size']
        activation = layer_config['activation']
        
        x = layers.Conv2D(
            filters=filters,
            kernel_size=kernel_size,
            padding=padding,
            use_bias=use_bias,
            activation=None,  # We'll apply activation separately for the last layer
            name=f'conv_{i+1}'
        )(x)
        
        # Apply activation (skip for last layer which is linear)
        if activation != 'linear':
            x = layers.Activation(activation, name=f'{activation}_{i+1}')(x)
    
    # Output is predicted noise
    outputs = x
    
    model = Model(inputs=inputs, outputs=outputs, name='TinyConditionalCNN')
    
    return model


class ConditionalDiffusionModel(Model):
    """
    Wrapper model that handles the diffusion training logic.
    Combines the noise prediction network with diffusion forward process.
    """
    
    def __init__(self, config: Dict[str, Any], noise_schedule, **kwargs):
        super().__init__(**kwargs)
        self.config = config
        self.noise_schedule = noise_schedule
        self.cnn = build_conditional_cnn(config)
        
        # Get schedule arrays for TF operations
        from src.diffusion import get_schedule_arrays
        self.schedule_arrays = get_schedule_arrays(
            noise_schedule.num_timesteps,
            noise_schedule.schedule_type,
            noise_schedule.beta_start,
            noise_schedule.beta_end,
            noise_schedule.cosine_s
        )
        
        # Convert to TF constants
        self.sqrt_alphas_cumprod = tf.constant(
            self.schedule_arrays['sqrt_alphas_cumprod'], dtype=tf.float32)
        self.sqrt_one_minus_alphas_cumprod = tf.constant(
            self.schedule_arrays['sqrt_one_minus_alphas_cumprod'], dtype=tf.float32)
    
    def call(self, inputs, training=False):
        """Forward pass: predict noise from noisy image + conditioning."""
        return self.cnn(inputs, training=training)
    
    def compute_loss(self, x_0, class_labels, timesteps, noise):
        """
        Compute diffusion training loss.
        
        Args:
            x_0: Clean images [batch, 28, 28, 1]
            class_labels: Integer labels [batch]
            timesteps: Integer timesteps [batch]
            noise: Ground truth noise [batch, 28, 28, 1]
            
        Returns:
            MSE loss between predicted and actual noise
        """
        from src.diffusion import forward_diffusion_tf
        from src.utils import prepare_model_input
        import numpy as np
        
        # Forward diffusion to get x_t
        x_t = forward_diffusion_tf(
            x_0, timesteps, noise,
            self.sqrt_alphas_cumprod,
            self.sqrt_one_minus_alphas_cumprod
        )
        
        # Prepare conditioning input
        # This needs to be done in TF for graph compatibility
        batch_size = tf.shape(x_0)[0]
        
        # One-hot encode labels
        class_onehot = tf.one_hot(class_labels, depth=10)
        
        # Normalize timesteps
        t_normalized = tf.cast(timesteps, tf.float32) / self.noise_schedule.num_timesteps
        t_normalized = tf.reshape(t_normalized, [-1, 1])
        
        # Broadcast to spatial maps
        # Class maps: [batch, 10] -> [batch, 28, 28, 10]
        class_maps = tf.zeros((batch_size, 28, 28, 10), dtype=tf.float32)
        for c in range(10):
            class_maps = tf.tensor_scatter_nd_update(
                class_maps,
                tf.where(tf.ones((batch_size, 28, 28, 1), dtype=tf.bool)),
                tf.tile(class_onehot[:, c:c+1], [1, 28*28])
            )
        
        # Simpler approach: use tf.repeat and tf.reshape
        class_maps = tf.repeat(class_onehot[:, tf.newaxis, tf.newaxis, :], 28*28, axis=1)
        class_maps = tf.reshape(class_maps, (batch_size, 28, 28, 10))
        
        # Timestep map: [batch, 1] -> [batch, 28, 28, 1]
        timestep_maps = tf.repeat(t_normalized[:, tf.newaxis, tf.newaxis, :], 28*28, axis=1)
        timestep_maps = tf.reshape(timestep_maps, (batch_size, 28, 28, 1))
        
        # Concatenate conditioning
        conditioning = tf.concat([class_maps, timestep_maps], axis=-1)
        
        # Concatenate noisy image + conditioning
        model_input = tf.concat([x_t, conditioning], axis=-1)
        
        # Predict noise
        noise_pred = self.cnn(model_input, training=training)
        
        # MSE loss
        loss = tf.reduce_mean(tf.square(noise - noise_pred))
        return loss
    
    def train_step(self, data):
        """Custom training step for diffusion."""
        x_0, class_labels = data
        
        batch_size = tf.shape(x_0)[0]
        
        # Sample random timesteps
        timesteps = tf.random.uniform(
            (batch_size,), 0, self.noise_schedule.num_timesteps, dtype=tf.int32)
        
        # Sample noise
        noise = tf.random.normal(tf.shape(x_0), dtype=tf.float32)
        
        with tf.GradientTape() as tape:
            loss = self.compute_loss(x_0, class_labels, timesteps, noise)
        
        gradients = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))
        
        return {'loss': loss}


def create_model(config: Dict[str, Any], noise_schedule) -> ConditionalDiffusionModel:
    """Factory function to create the diffusion model."""
    return ConditionalDiffusionModel(config, noise_schedule, name='TinyDiffusionModel')


def build_inference_model(config: Dict[str, Any]) -> Model:
    """Build the standalone CNN for inference (export)."""
    return build_conditional_cnn(config)


def get_model_config_summary(config: Dict[str, Any]) -> Dict[str, Any]:
    """Get model configuration summary for logging."""
    model_config = config.get('model', {})
    conv_layers = model_config.get('conv_layers', [])
    
    total_params = 0
    # Estimate parameters
    in_channels = model_config.get('total_input_channels', 12)
    for layer in conv_layers:
        filters = layer['filters']
        kernel_size = layer['kernel_size']
        use_bias = model_config.get('use_bias', True)
        params = filters * (in_channels * kernel_size * kernel_size + (1 if use_bias else 0))
        total_params += params
        in_channels = filters
    
    return {
        'input_shape': (model_config.get('input_height', 28), 
                       model_config.get('input_width', 28), 
                       model_config.get('total_input_channels', 12)),
        'output_shape': (model_config.get('input_height', 28), 
                        model_config.get('input_width', 28), 1),
        'conv_layers': len(conv_layers),
        'estimated_params': total_params,
        'architecture': 'Conditional CNN'
    }


if __name__ == "__main__":
    # Test model building
    config = {
        'model': {
            'input_height': 28,
            'input_width': 28,
            'input_channels': 1,
            'num_classes': 10,
            'conditioning_channels': 11,
            'total_input_channels': 12,
            'conv_layers': [
                {'filters': 8, 'kernel_size': 3, 'activation': 'relu'},
                {'filters': 8, 'kernel_size': 3, 'activation': 'relu'},
                {'filters': 12, 'kernel_size': 3, 'activation': 'relu'},
                {'filters': 8, 'kernel_size': 3, 'activation': 'relu'},
                {'filters': 1, 'kernel_size': 3, 'activation': 'linear'}
            ],
            'use_bias': True,
            'padding': 'same'
        }
    }
    
    # Build inference model
    model = build_inference_model(config)
    print_model_summary(model)
    
    # Test forward pass
    test_input = tf.random.normal((1, 28, 28, 12))
    output = model(test_input)
    print(f"Output shape: {output.shape}")
    
    # Count parameters
    param_count = count_parameters(model)
    print(f"Total parameters: {param_count}")
    
    # Expected: ~3,277
    print("Model module test passed!")