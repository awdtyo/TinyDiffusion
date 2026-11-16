"""
Conditional CNN and Tiny U-Net models for TinyDiffusion-ESP32.
Implements noise/v prediction networks with class and timestep conditioning.
Supports both epsilon-prediction and v-prediction.
"""
import tensorflow as tf
from tensorflow.keras import layers, Model, Input
from typing import Dict, Any, List, Tuple, Optional
from src.utils import count_parameters, print_model_summary


def build_conditional_cnn(config: Dict[str, Any]) -> Model:
    """
    Build the conditional CNN noise prediction network (legacy architecture).
    
    Architecture:
    Input: 28x28x12 (noisy image 1ch + 10 class maps + 1 timestep map)
    
    1. Conv2D: 8 filters, 3x3, SAME, ReLU
    2. Conv2D: 8 filters, 3x3, SAME, ReLU
    3. Conv2D: 12 filters, 3x3, SAME, ReLU
    4. Conv2D: 8 filters, 3x3, SAME, ReLU
    5. Conv2D: 1 filter, 3x3, SAME, Linear
    
    Output: 28x28x1 (predicted noise or v)
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
            activation=None,
            name=f'conv_{i+1}'
        )(x)
        
        if activation != 'linear':
            x = layers.Activation(activation, name=f'{activation}_{i+1}')(x)
    
    outputs = x
    
    model = Model(inputs=inputs, outputs=outputs, name='TinyConditionalCNN')
    
    return model


class SinusoidalTimestepEmbedding(layers.Layer):
    """Sinusoidal timestep embedding for diffusion models."""
    
    def __init__(self, embedding_dim: int = 32, max_frequency: float = 10000.0, **kwargs):
        super().__init__(**kwargs)
        self.embedding_dim = embedding_dim
        self.max_frequency = max_frequency
        self.half_dim = embedding_dim // 2
        
    def build(self, input_shape):
        # Pre-compute frequency bands
        frequencies = tf.exp(
            tf.range(self.half_dim, dtype=tf.float32) * 
            (-tf.math.log(float(self.max_frequency)) / (self.half_dim - 1))
        )
        self.frequencies = tf.Variable(frequencies, trainable=False, name='frequencies')
        super().build(input_shape)
        
    def call(self, timesteps: tf.Tensor) -> tf.Tensor:
        """
        Args:
            timesteps: [batch] or [batch, 1] normalized timesteps in [0, 1]
        Returns:
            embeddings: [batch, embedding_dim]
        """
        # Ensure timesteps is [batch, 1]
        if timesteps.shape.rank == 1:
            timesteps = tf.expand_dims(timesteps, -1)
        
        # Scale to [0, 1000] for frequency computation
        timesteps_scaled = timesteps * 1000.0
        
        # Compute sinusoidal embeddings
        args = timesteps_scaled * self.frequencies
        embeddings = tf.concat([tf.sin(args), tf.cos(args)], axis=-1)
        
        # Handle odd embedding_dim
        if self.embedding_dim % 2 == 1:
            embeddings = embeddings[:, :self.embedding_dim]
            
        return embeddings
    
    def get_config(self):
        config = super().get_config()
        config.update({
            'embedding_dim': self.embedding_dim,
            'max_frequency': self.max_frequency,
        })
        return config


class ClassEmbedding(layers.Layer):
    """Learnable class embedding layer."""
    
    def __init__(self, num_classes: int = 10, embedding_dim: int = 32, **kwargs):
        super().__init__(**kwargs)
        self.num_classes = num_classes
        self.embedding_dim = embedding_dim
        
    def build(self, input_shape):
        self.embedding = self.add_weight(
            name='class_embedding',
            shape=(self.num_classes, self.embedding_dim),
            initializer='random_normal',
            trainable=True
        )
        super().build(input_shape)
        
    def call(self, class_labels: tf.Tensor) -> tf.Tensor:
        """
        Args:
            class_labels: [batch] integer class labels 0-9
        Returns:
            embeddings: [batch, embedding_dim]
        """
        return tf.nn.embedding_lookup(self.embedding, class_labels)
    
    def get_config(self):
        config = super().get_config()
        config.update({
            'num_classes': self.num_classes,
            'embedding_dim': self.embedding_dim,
        })
        return config


class FiLMConditioning(layers.Layer):
    """FiLM (Feature-wise Linear Modulation) conditioning layer.
    
    Modulates feature maps using scale and shift parameters derived from
    timestep and class embeddings.
    """
    
    def __init__(self, 
                 feature_channels: int,
                 conditioning_dim: int,
                 hidden_dim: int = 64,
                 **kwargs):
        super().__init__(**kwargs)
        self.feature_channels = feature_channels
        self.conditioning_dim = conditioning_dim
        self.hidden_dim = hidden_dim
        
    def build(self, input_shape):
        # MLP to generate scale and shift from conditioning
        self.film_mlp = tf.keras.Sequential([
            layers.Dense(self.hidden_dim, activation='relu', name='film_dense_1'),
            layers.Dense(self.feature_channels * 2, name='film_dense_2')  # scale + shift
        ], name='film_mlp')
        super().build(input_shape)
        
    def call(self, features: tf.Tensor, conditioning: tf.Tensor) -> tf.Tensor:
        """
        Args:
            features: [batch, H, W, C] feature maps to modulate
            conditioning: [batch, conditioning_dim] conditioning vector
        Returns:
            modulated: [batch, H, W, C] modulated features
        """
        # Generate scale and shift
        film_params = self.film_mlp(conditioning)  # [batch, 2*C]
        scale, shift = tf.split(film_params, 2, axis=-1)  # each [batch, C]
        
        # Reshape for broadcasting: [batch, 1, 1, C]
        scale = tf.reshape(scale, [-1, 1, 1, self.feature_channels])
        shift = tf.reshape(shift, [-1, 1, 1, self.feature_channels])
        
        # Apply FiLM: scale * features + shift
        modulated = scale * features + shift
        
        return modulated
    
    def get_config(self):
        config = super().get_config()
        config.update({
            'feature_channels': self.feature_channels,
            'conditioning_dim': self.conditioning_dim,
            'hidden_dim': self.hidden_dim,
        })
        return config


class ResidualBlock(layers.Layer):
    """Residual convolutional block with FiLM conditioning."""
    
    def __init__(self, 
                 channels: int,
                 kernel_size: int = 3,
                 activation: str = 'relu',
                 use_bias: bool = True,
                 padding: str = 'same',
                 conditioning_dim: int = 64,
                 film_hidden_dim: int = 64,
                 **kwargs):
        super().__init__(**kwargs)
        self.channels = channels
        self.kernel_size = kernel_size
        self.activation_name = activation
        self.use_bias = use_bias
        self.padding = padding
        self.conditioning_dim = conditioning_dim
        self.film_hidden_dim = film_hidden_dim
        
    def build(self, input_shape):
        # First conv
        self.conv1 = layers.Conv2D(
            filters=self.channels,
            kernel_size=self.kernel_size,
            padding=self.padding,
            use_bias=self.use_bias,
            name='conv1'
        )
        self.norm1 = layers.BatchNormalization(name='bn1')
        self.act1 = layers.Activation(self.activation_name, name='act1')
        
        # Second conv
        self.conv2 = layers.Conv2D(
            filters=self.channels,
            kernel_size=self.kernel_size,
            padding=self.padding,
            use_bias=self.use_bias,
            name='conv2'
        )
        self.norm2 = layers.BatchNormalization(name='bn2')
        
        # FiLM conditioning for both conv layers
        self.film1 = FiLMConditioning(
            feature_channels=self.channels,
            conditioning_dim=self.conditioning_dim,
            hidden_dim=self.film_hidden_dim,
            name='film1'
        )
        self.film2 = FiLMConditioning(
            feature_channels=self.channels,
            conditioning_dim=self.conditioning_dim,
            hidden_dim=self.film_hidden_dim,
            name='film2'
        )
        
        # Residual connection (identity if channels match)
        in_channels = input_shape[-1]
        if in_channels != self.channels:
            self.residual_conv = layers.Conv2D(
                filters=self.channels,
                kernel_size=1,
                padding='same',
                use_bias=self.use_bias,
                name='residual_conv'
            )
        else:
            self.residual_conv = None
            
        self.final_act = layers.Activation(self.activation_name, name='final_act')
        super().build(input_shape)
        
    def call(self, inputs: Tuple[tf.Tensor, tf.Tensor], training: bool = False) -> tf.Tensor:
        """
        Args:
            inputs: tuple of (features, conditioning)
                features: [batch, H, W, C]
                conditioning: [batch, conditioning_dim]
            training: whether in training mode
        Returns:
            output: [batch, H, W, channels]
        """
        features, conditioning = inputs
        residual = features
        
        # First conv block
        x = self.conv1(features)
        x = self.norm1(x, training=training)
        x = self.film1(x, conditioning)
        x = self.act1(x)
        
        # Second conv block
        x = self.conv2(x)
        x = self.norm2(x, training=training)
        x = self.film2(x, conditioning)
        
        # Residual connection
        if self.residual_conv is not None:
            residual = self.residual_conv(residual)
            
        x = x + residual
        x = self.final_act(x)
        
        return x
    
    def get_config(self):
        config = super().get_config()
        config.update({
            'channels': self.channels,
            'kernel_size': self.kernel_size,
            'activation': self.activation_name,
            'use_bias': self.use_bias,
            'padding': self.padding,
            'conditioning_dim': self.conditioning_dim,
            'film_hidden_dim': self.film_hidden_dim,
        })
        return config


class DownsampleBlock(layers.Layer):
    """Downsampling block with strided convolution."""
    
    def __init__(self, 
                 out_channels: int,
                 kernel_size: int = 3,
                 stride: int = 2,
                 activation: str = 'relu',
                 use_bias: bool = True,
                 padding: str = 'same',
                 **kwargs):
        super().__init__(**kwargs)
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.activation_name = activation
        self.use_bias = use_bias
        self.padding = padding
        
    def build(self, input_shape):
        self.conv = layers.Conv2D(
            filters=self.out_channels,
            kernel_size=self.kernel_size,
            strides=self.stride,
            padding=self.padding,
            use_bias=self.use_bias,
            name='downsample_conv'
        )
        self.norm = layers.BatchNormalization(name='downsample_bn')
        self.act = layers.Activation(self.activation_name, name='downsample_act')
        super().build(input_shape)
        
    def call(self, inputs: tf.Tensor, training: bool = False) -> tf.Tensor:
        x = self.conv(inputs)
        x = self.norm(x, training=training)
        x = self.act(x)
        return x
    
    def get_config(self):
        config = super().get_config()
        config.update({
            'out_channels': self.out_channels,
            'kernel_size': self.kernel_size,
            'stride': self.stride,
            'activation': self.activation_name,
            'use_bias': self.use_bias,
            'padding': self.padding,
        })
        return config


class UpsampleBlock(layers.Layer):
    """Upsampling block with transposed convolution."""
    
    def __init__(self, 
                 out_channels: int,
                 kernel_size: int = 3,
                 stride: int = 2,
                 activation: str = 'relu',
                 use_bias: bool = True,
                 padding: str = 'same',
                 **kwargs):
        super().__init__(**kwargs)
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.activation_name = activation
        self.use_bias = use_bias
        self.padding = padding
        
    def build(self, input_shape):
        self.conv_transpose = layers.Conv2DTranspose(
            filters=self.out_channels,
            kernel_size=self.kernel_size,
            strides=self.stride,
            padding=self.padding,
            use_bias=self.use_bias,
            name='upsample_conv_transpose'
        )
        self.norm = layers.BatchNormalization(name='upsample_bn')
        self.act = layers.Activation(self.activation_name, name='upsample_act')
        super().build(input_shape)
        
    def call(self, inputs: tf.Tensor, training: bool = False) -> tf.Tensor:
        x = self.conv_transpose(inputs)
        x = self.norm(x, training=training)
        x = self.act(x)
        return x
    
    def get_config(self):
        config = super().get_config()
        config.update({
            'out_channels': self.out_channels,
            'kernel_size': self.kernel_size,
            'stride': self.stride,
            'activation': self.activation_name,
            'use_bias': self.use_bias,
            'padding': self.padding,
        })
        return config


def build_tiny_unet_v1(config: Dict[str, Any]) -> Model:
    """
    Build the conditional residual Tiny U-Net for diffusion.
    
    Architecture:
    Input: 28x28x12 (noisy image 1ch + 10 class maps + 1 timestep map)
    
    Stage                    Resolution    Channels
    -------------------------------------------------
    Input                    28x28         12
    Initial Conv             28x28         16
    Encoder Level 1          28x28         16  (1 residual block)
    Downsample 1             14x14         32
    Encoder Level 2          14x14         32  (1 residual block)
    Downsample 2             7x7           48
    Bottleneck               7x7           48  (2 residual blocks)
    Upsample 1               14x14         32
    Decoder Level 1          14x14         32  (1 residual block + skip)
    Upsample 2               28x28         16
    Decoder Level 2          28x28         16  (1 residual block + skip)
    Output Conv              28x28         1
    
    Output: 28x28x1 (predicted v)
    
    Conditioning: FiLM modulation injected into each residual block
    """
    model_config = config.get('model', {})
    unet_config = model_config.get('tiny_unet_v1', {})
    
    input_height = model_config.get('input_height', 28)
    input_width = model_config.get('input_width', 28)
    total_input_channels = model_config.get('total_input_channels', 12)
    num_classes = model_config.get('num_classes', 10)
    
    # Architecture parameters
    initial_conv_filters = unet_config.get('initial_conv_filters', 16)
    initial_conv_kernel = unet_config.get('initial_conv_kernel', 3)
    
    encoder_config = unet_config.get('encoder', [
        {'resolution': [28, 28], 'channels': 16, 'num_blocks': 1},
        {'resolution': [14, 14], 'channels': 32, 'num_blocks': 1}
    ])
    
    bottleneck_config = unet_config.get('bottleneck', {
        'resolution': [7, 7], 'channels': 48, 'num_blocks': 2
    })
    
    decoder_config = unet_config.get('decoder', [
        {'resolution': [14, 14], 'channels': 32, 'num_blocks': 1},
        {'resolution': [28, 28], 'channels': 16, 'num_blocks': 1}
    ])
    
    output_channels = unet_config.get('output_channels', 1)
    
    residual_config = unet_config.get('residual_block', {
        'kernel_size': 3,
        'activation': 'relu',
        'use_bias': True,
        'padding': 'same',
        'film_embedding_dim': 32,
        'film_hidden_dim': 64
    })
    
    downsample_config = unet_config.get('downsample', {
        'type': 'strided_conv', 'kernel_size': 3, 'stride': 2
    })
    
    upsample_config = unet_config.get('upsample', {
        'type': 'transposed_conv', 'kernel_size': 3, 'stride': 2
    })
    
    timestep_emb_config = unet_config.get('timestep_embedding', {
        'type': 'sinusoidal', 'embedding_dim': 32, 'max_frequency': 10000.0
    })
    
    class_emb_config = unet_config.get('class_embedding', {
        'embedding_dim': 32
    })
    
    # Derived parameters
    kernel_size = residual_config['kernel_size']
    activation = residual_config['activation']
    use_bias = residual_config['use_bias']
    padding = residual_config['padding']
    film_embedding_dim = residual_config['film_embedding_dim']
    film_hidden_dim = residual_config['film_hidden_dim']
    
    # Total conditioning dimension = timestep_emb + class_emb
    conditioning_dim = timestep_emb_config['embedding_dim'] + class_emb_config['embedding_dim']
    
    # Input
    inputs = Input(shape=(input_height, input_width, total_input_channels), name='model_input')
    
    # Split input: noisy image (channel 0) and conditioning (channels 1-11)
    # But we'll use the full input for initial conv, and extract class/timestep for embeddings
    # Actually, the conditioning is already in the input. For embeddings, we need to extract
    # class label and timestep. Since they're broadcast spatially, we can extract from input.
    
    # For the embedding path, we'll create separate inputs for class and timestep
    # But the model only receives the concatenated 12-channel input.
    # We need to extract class and timestep from the conditioning channels.
    # Channel 0: noisy image
    # Channels 1-10: class one-hot maps (broadcast)
    # Channel 11: timestep map (broadcast)
    
    # Extract class label from input (argmax over channels 1-10 at any spatial location)
    # Extract timestep from channel 11
    # This is a bit awkward. Let's instead modify the approach:
    # The model will take the full 12-channel input, and we'll also provide
    # separate class and timestep inputs for the embedding path.
    
    # Actually, looking at the existing code, the model only gets the 12-channel input.
    # For the U-Net, we need to extract class and timestep from the conditioning channels.
    # Let's do this with Lambda layers.
    
    # Initial convolution on full input
    x = layers.Conv2D(
        filters=initial_conv_filters,
        kernel_size=initial_conv_kernel,
        padding=padding,
        use_bias=use_bias,
        name='initial_conv'
    )(inputs)
    x = layers.BatchNormalization(name='initial_bn')(x)
    x = layers.Activation(activation, name='initial_act')(x)
    
    # Build conditioning embeddings from input
    # Extract class maps (channels 1-10) and timestep map (channel 11)
    class_maps = layers.Lambda(lambda t: t[:, :, :, 1:11], name='extract_class_maps')(inputs)
    timestep_map = layers.Lambda(lambda t: t[:, :, :, 11:12], name='extract_timestep_map')(inputs)
    
    # Get class label (argmax over class channels at position 0,0)
    class_label = layers.Lambda(lambda t: tf.cast(tf.argmax(t[:, 0, 0, :], axis=-1), tf.int32), 
                                name='extract_class_label', dtype=tf.int32)(class_maps)
    
    # Get timestep (value at position 0,0 in timestep channel)
    timestep_normalized = layers.Lambda(lambda t: t[:, 0, 0, 0], name='extract_timestep')(timestep_map)
    
    # Timestep embedding
    timestep_embedding = SinusoidalTimestepEmbedding(
        embedding_dim=timestep_emb_config['embedding_dim'],
        max_frequency=timestep_emb_config['max_frequency'],
        name='timestep_embedding'
    )(timestep_normalized)
    
    # Class embedding
    class_embedding = ClassEmbedding(
        num_classes=num_classes,
        embedding_dim=class_emb_config['embedding_dim'],
        name='class_embedding'
    )(class_label)
    
    # Combine conditioning
    conditioning = layers.Concatenate(name='conditioning_concat')([timestep_embedding, class_embedding])
    # conditioning shape: [batch, conditioning_dim]
    
    # Encoder with skip connections
    skip_connections = []
    
    # Encoder Level 1: 28x28, 16 channels
    for i in range(encoder_config[0]['num_blocks']):
        x = ResidualBlock(
            channels=encoder_config[0]['channels'],
            kernel_size=kernel_size,
            activation=activation,
            use_bias=use_bias,
            padding=padding,
            conditioning_dim=conditioning_dim,
            film_hidden_dim=film_hidden_dim,
            name=f'encoder1_block{i+1}'
        )((x, conditioning))
    skip_connections.append(x)  # 28x28x16
    
    # Downsample 1: 28x28 -> 14x14, 16 -> 32
    x = DownsampleBlock(
        out_channels=encoder_config[1]['channels'],
        kernel_size=downsample_config['kernel_size'],
        stride=downsample_config['stride'],
        activation=activation,
        use_bias=use_bias,
        padding=padding,
        name='downsample1'
    )(x)
    
    # Encoder Level 2: 14x14, 32 channels
    for i in range(encoder_config[1]['num_blocks']):
        x = ResidualBlock(
            channels=encoder_config[1]['channels'],
            kernel_size=kernel_size,
            activation=activation,
            use_bias=use_bias,
            padding=padding,
            conditioning_dim=conditioning_dim,
            film_hidden_dim=film_hidden_dim,
            name=f'encoder2_block{i+1}'
        )((x, conditioning))
    skip_connections.append(x)  # 14x14x32
    
    # Downsample 2: 14x14 -> 7x7, 32 -> 48
    x = DownsampleBlock(
        out_channels=bottleneck_config['channels'],
        kernel_size=downsample_config['kernel_size'],
        stride=downsample_config['stride'],
        activation=activation,
        use_bias=use_bias,
        padding=padding,
        name='downsample2'
    )(x)
    
    # Bottleneck: 7x7, 48 channels
    for i in range(bottleneck_config['num_blocks']):
        x = ResidualBlock(
            channels=bottleneck_config['channels'],
            kernel_size=kernel_size,
            activation=activation,
            use_bias=use_bias,
            padding=padding,
            conditioning_dim=conditioning_dim,
            film_hidden_dim=film_hidden_dim,
            name=f'bottleneck_block{i+1}'
        )((x, conditioning))
    
    # Decoder with skip connections
    # Upsample 1: 7x7 -> 14x14, 48 -> 32
    x = UpsampleBlock(
        out_channels=decoder_config[0]['channels'],
        kernel_size=upsample_config['kernel_size'],
        stride=upsample_config['stride'],
        activation=activation,
        use_bias=use_bias,
        padding=padding,
        name='upsample1'
    )(x)
    
    # Skip connection from encoder level 2 (14x14x32)
    skip2 = skip_connections.pop()  # 14x14x32
    x = layers.Concatenate(name='skip_concat1')([x, skip2])
    
    # Decoder Level 1: 14x14, 32 channels
    for i in range(decoder_config[0]['num_blocks']):
        x = ResidualBlock(
            channels=decoder_config[0]['channels'],
            kernel_size=kernel_size,
            activation=activation,
            use_bias=use_bias,
            padding=padding,
            conditioning_dim=conditioning_dim,
            film_hidden_dim=film_hidden_dim,
            name=f'decoder1_block{i+1}'
        )((x, conditioning))
    
    # Upsample 2: 14x14 -> 28x28, 32 -> 16
    x = UpsampleBlock(
        out_channels=decoder_config[1]['channels'],
        kernel_size=upsample_config['kernel_size'],
        stride=upsample_config['stride'],
        activation=activation,
        use_bias=use_bias,
        padding=padding,
        name='upsample2'
    )(x)
    
    # Skip connection from encoder level 1 (28x28x16)
    skip1 = skip_connections.pop()  # 28x28x16
    x = layers.Concatenate(name='skip_concat2')([x, skip1])
    
    # Decoder Level 2: 28x28, 16 channels
    for i in range(decoder_config[1]['num_blocks']):
        x = ResidualBlock(
            channels=decoder_config[1]['channels'],
            kernel_size=kernel_size,
            activation=activation,
            use_bias=use_bias,
            padding=padding,
            conditioning_dim=conditioning_dim,
            film_hidden_dim=film_hidden_dim,
            name=f'decoder2_block{i+1}'
        )((x, conditioning))
    
    # Output convolution
    outputs = layers.Conv2D(
        filters=output_channels,
        kernel_size=3,
        padding=padding,
        use_bias=use_bias,
        activation='linear',
        name='output_conv'
    )(x)
    
    model = Model(inputs=inputs, outputs=outputs, name='TinyUNetV1')
    
    return model


@tf.keras.utils.register_keras_serializable()
class ConditionalDiffusionModel(Model):
    """
    Wrapper model that handles the diffusion training logic.
    Combines the noise/v prediction network with diffusion forward process.
    Supports both epsilon-prediction and v-prediction.
    """
    
    def __init__(self, config: Dict[str, Any], noise_schedule, **kwargs):
        super().__init__(**kwargs)
        self.config = config
        self.noise_schedule = noise_schedule
        # Use architecture selection
        self.cnn = build_inference_model(config)
        
        # Get prediction type from config
        self.prediction_type = config.get('diffusion', {}).get('prediction_type', 'epsilon')
        
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
    
    def get_config(self):
        """Return config for serialization."""
        config = super().get_config()
        config.update({
            'model_config': self.config,
            'num_timesteps': self.noise_schedule.num_timesteps,
            'schedule_type': self.noise_schedule.schedule_type,
            'beta_start': self.noise_schedule.beta_start,
            'beta_end': self.noise_schedule.beta_end,
            'cosine_s': self.noise_schedule.cosine_s,
            'normalize_timestep': self.noise_schedule.normalize_timestep,
        })
        return config
    
    @classmethod
    def from_config(cls, config):
        """Recreate model from config."""
        # Extract model config and noise schedule params
        model_config = config.pop('model_config', {})
        noise_schedule_params = {
            'num_timesteps': config.pop('num_timesteps', 100),
            'schedule_type': config.pop('schedule_type', 'linear'),
            'beta_start': config.pop('beta_start', 0.0001),
            'beta_end': config.pop('beta_end', 0.02),
            'cosine_s': config.pop('cosine_s', 0.008),
            'normalize_timestep': config.pop('normalize_timestep', True),
        }
        
        # Create noise schedule object
        from src.diffusion import NoiseSchedule
        noise_schedule = NoiseSchedule({
            'diffusion': noise_schedule_params
        })
        
        return cls(model_config, noise_schedule, **config)
    
    def call(self, inputs, training=False):
        """Forward pass: predict noise/v from noisy image + conditioning."""
        return self.cnn(inputs, training=training)
    
    def compute_loss(self, x_0, class_labels, timesteps, noise, training=False):
        """
        Compute diffusion training loss.
        
        Args:
            x_0: Clean images [batch, 28, 28, 1]
            class_labels: Integer labels [batch]
            timesteps: Integer timesteps [batch]
            noise: Ground truth noise [batch, 28, 28, 1]
            training: Whether in training mode
            
        Returns:
            MSE loss between predicted and actual target (epsilon or v)
        """
        from src.diffusion import forward_diffusion_tf, forward_diffusion_v_tf
        from src.utils import prepare_model_input
        import numpy as np
        
        if self.prediction_type == 'v':
            # V-prediction forward diffusion
            x_t, v_true = forward_diffusion_v_tf(
                x_0, timesteps, noise,
                self.sqrt_alphas_cumprod,
                self.sqrt_one_minus_alphas_cumprod
            )
            target = v_true
        else:
            # Epsilon-prediction forward diffusion
            x_t = forward_diffusion_tf(
                x_0, timesteps, noise,
                self.sqrt_alphas_cumprod,
                self.sqrt_one_minus_alphas_cumprod
            )
            target = noise
        
        # Prepare conditioning input
        # This needs to be done in TF for graph compatibility
        batch_size = tf.shape(x_0)[0]
        
        # One-hot encode labels
        class_onehot = tf.one_hot(class_labels, depth=10)
        
        # Normalize timesteps
        t_normalized = tf.cast(timesteps, tf.float32) / self.noise_schedule.num_timesteps
        t_normalized = tf.reshape(t_normalized, [-1, 1])
        
        # Broadcast class maps: [batch, 10] -> [batch, 28, 28, 10]
        class_maps = tf.repeat(class_onehot[:, tf.newaxis, tf.newaxis, :], 28*28, axis=1)
        class_maps = tf.reshape(class_maps, (batch_size, 28, 28, 10))
        
        # Broadcast timestep map: [batch, 1] -> [batch, 28, 28, 1]
        timestep_maps = tf.repeat(t_normalized[:, tf.newaxis, tf.newaxis, :], 28*28, axis=1)
        timestep_maps = tf.reshape(timestep_maps, (batch_size, 28, 28, 1))
        
        # Concatenate conditioning
        conditioning = tf.concat([class_maps, timestep_maps], axis=-1)
        
        # Concatenate noisy image + conditioning
        model_input = tf.concat([x_t, conditioning], axis=-1)
        
        # Predict noise or v
        pred = self.cnn(model_input, training=training)
        
        # MSE loss
        loss = tf.reduce_mean(tf.square(target - pred))
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
            loss = self.compute_loss(x_0, class_labels, timesteps, noise, training=True)
        
        gradients = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))
        
        # Update metrics (including loss tracking)
        for metric in self.metrics:
            if hasattr(metric, 'update_state') and metric.__class__.__name__ != 'CompileMetrics':
                metric.update_state(loss)
        
        # Return metrics results
        return {m.name: m.result() for m in self.metrics if m.__class__.__name__ != 'CompileMetrics'}
    
    def test_step(self, data):
        """Custom validation step for diffusion."""
        x_0, class_labels = data
        
        batch_size = tf.shape(x_0)[0]
        
        # Sample random timesteps for validation
        timesteps = tf.random.uniform(
            (batch_size,), 0, self.noise_schedule.num_timesteps, dtype=tf.int32)
        
        # Sample noise
        noise = tf.random.normal(tf.shape(x_0), dtype=tf.float32)
        
        # Compute loss (no gradient tape needed)
        loss = self.compute_loss(x_0, class_labels, timesteps, noise, training=False)
        
        # Update metrics
        for metric in self.metrics:
            if hasattr(metric, 'update_state') and metric.__class__.__name__ != 'CompileMetrics':
                metric.update_state(loss)
        
        return {m.name: m.result() for m in self.metrics if m.__class__.__name__ != 'CompileMetrics'}


def create_model(config: Dict[str, Any], noise_schedule) -> ConditionalDiffusionModel:
    """Factory function to create the diffusion model."""
    return ConditionalDiffusionModel(config, noise_schedule, name='TinyDiffusionModel')


def build_inference_model(config: Dict[str, Any]) -> Model:
    """Build the standalone model for inference (export).
    
    Selects architecture based on config['model']['architecture'].
    """
    model_config = config.get('model', {})
    architecture = model_config.get('architecture', 'conditional_cnn')
    
    if architecture == 'tiny_unet_v1':
        return build_tiny_unet_v1(config)
    else:
        # Default to legacy CNN
        return build_conditional_cnn(config)


def get_model_config_summary(config: Dict[str, Any]) -> Dict[str, Any]:
    """Get model configuration summary for logging."""
    model_config = config.get('model', {})
    architecture = model_config.get('architecture', 'conditional_cnn')
    
    if architecture == 'tiny_unet_v1':
        return get_tiny_unet_v1_config_summary(config)
    else:
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


def get_tiny_unet_v1_config_summary(config: Dict[str, Any]) -> Dict[str, Any]:
    """Get Tiny U-Net v1 configuration summary for logging."""
    model_config = config.get('model', {})
    unet_config = model_config.get('tiny_unet_v1', {})
    
    # Build model to get actual parameter count
    model = build_tiny_unet_v1(config)
    actual_params = count_parameters(model)
    
    encoder_config = unet_config.get('encoder', [
        {'resolution': [28, 28], 'channels': 16, 'num_blocks': 1},
        {'resolution': [14, 14], 'channels': 32, 'num_blocks': 1}
    ])
    
    bottleneck_config = unet_config.get('bottleneck', {
        'resolution': [7, 7], 'channels': 48, 'num_blocks': 2
    })
    
    decoder_config = unet_config.get('decoder', [
        {'resolution': [14, 14], 'channels': 32, 'num_blocks': 1},
        {'resolution': [28, 28], 'channels': 16, 'num_blocks': 1}
    ])
    
    # Count residual blocks
    num_residual_blocks = (encoder_config[0]['num_blocks'] + 
                          encoder_config[1]['num_blocks'] + 
                          bottleneck_config['num_blocks'] + 
                          decoder_config[0]['num_blocks'] + 
                          decoder_config[1]['num_blocks'])
    
    return {
        'input_shape': (model_config.get('input_height', 28), 
                       model_config.get('input_width', 28), 
                       model_config.get('total_input_channels', 12)),
        'output_shape': (model_config.get('input_height', 28), 
                        model_config.get('input_width', 28), 1),
        'residual_blocks': num_residual_blocks,
        'actual_params': actual_params,
        'architecture': 'Tiny U-Net v1'
    }