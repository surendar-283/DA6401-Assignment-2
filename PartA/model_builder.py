import torch
import torch.nn as nn
import functools

def generate_activation_layer(activation_type):
    """Generate and return the specified activation layer."""
    activation_options = {
        'relu': nn.ReLU(),
        'leaky_relu': nn.LeakyReLU(0.1),
        'gelu': nn.GELU()
    }
    return activation_options.get(activation_type, nn.ReLU())

def assemble_conv_layer(input_dims, output_dims, filter_size, 
                       normalize_batch, drop_probability, activation_layer):
    """Assemble a convolutional layer with optional batch normalization and dropout."""
    components = [
        nn.Conv2d(input_dims, output_dims, filter_size, padding='same'),
        activation_layer
    ]

    if normalize_batch:
        components.append(nn.BatchNorm2d(output_dims))

    components.append(nn.MaxPool2d(kernel_size=2, stride=2))

    if drop_probability > 0:
        components.append(nn.Dropout(drop_probability))

    return nn.Sequential(*components)

def compute_flatten_dimensions(conv_stack, img_dimensions):
    """Compute the flattened dimensions after convolutional layers."""
    with torch.no_grad():
        sample_tensor = torch.zeros(1, img_dimensions[2], img_dimensions[0], img_dimensions[1])
        for layer in conv_stack:
            sample_tensor = layer(sample_tensor)
        return sample_tensor.view(1, -1).shape[1]

def assemble_dense_stack(input_size, hidden_dims, drop_probability, activation_layer, output_classes):
    """Assemble dense layers with optional dropout."""
    components = [nn.Flatten()]
    prev_dim = input_size

    for dim in hidden_dims:
        components.extend([
            nn.Linear(prev_dim, dim),
            activation_layer
        ])
        if drop_probability > 0:
            components.append(nn.Dropout(drop_probability))
        prev_dim = dim

    components.append(nn.Linear(prev_dim, output_classes))
    return nn.Sequential(*components)

def process_inputs(x, conv_stack, dense_stack):
    """Process inputs through the network."""
    for layer in conv_stack:
        x = layer(x)
    return dense_stack(x)

def construct_vision_network(params, img_dimensions=(224, 224, 3), output_classes=10):
    """Construct a vision network based on configuration parameters."""
    # Extract parameters
    channel_maps = params.get('conv_filters', [32, 32, 32, 32, 32])
    filter_dimensions = params.get('kernel_sizes', [3, 3, 3, 3, 3])
    hidden_dims = params.get('dense_units', [128])
    drop_probability = params.get('dropout_rate', 0.2)
    normalize_batch = params.get('use_batchnorm', True)
    activation_type = params.get('activation', 'relu')

    # Generate activation layer
    activation_layer = generate_activation_layer(activation_type)

    # Build convolutional stack
    conv_stack = []
    input_channels = img_dimensions[2]

    for i in range(len(channel_maps)):
        apply_dropout = (i < len(channel_maps) - 1) and (drop_probability > 0)
        layer = assemble_conv_layer(
            input_channels,
            channel_maps[i],
            filter_dimensions[i],
            normalize_batch,
            drop_probability if apply_dropout else 0,
            activation_layer
        )
        conv_stack.append(layer)
        input_channels = channel_maps[i]

    # Calculate flattened dimensions
    flattened_size = compute_flatten_dimensions(conv_stack, img_dimensions)

    # Build dense stack
    dense_stack = assemble_dense_stack(
        flattened_size,
        hidden_dims,
        drop_probability,
        activation_layer,
        output_classes
    )

    # Create complete model
    network = nn.Module()
    network.conv_stack = nn.ModuleList(conv_stack)
    network.dense_stack = dense_stack
    network.forward = functools.partial(process_inputs, conv_stack=network.conv_stack, dense_stack=network.dense_stack)

    return network