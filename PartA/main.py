import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader, random_split
import wandb
from tqdm import tqdm
import os
from model_builder import construct_vision_network

def generate_transformation_pipelines(apply_augmentation=False):
    """Generate image transformation pipelines for training and testing."""
    base_transforms = [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ]

    if apply_augmentation:
        train_pipeline = transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            *base_transforms
        ])
    else:
        train_pipeline = transforms.Compose(base_transforms)

    test_pipeline = transforms.Compose(base_transforms)

    return train_pipeline, test_pipeline

def load_and_partition_data(data_directory, train_transform, test_transform, val_split=0.2):
    """Load and partition dataset into train, validation, and test sets."""
    # Load the datasets
    train_dataset = ImageFolder(os.path.join(data_directory, 'train'), transform=train_transform)
    test_dataset = ImageFolder(os.path.join(data_directory, 'val'), transform=test_transform)

    # Partition training data to include validation
    val_count = int(val_split * len(train_dataset))
    train_count = len(train_dataset) - val_count
    train_subset, val_subset = random_split(train_dataset, [train_count, val_count])

    return train_subset, val_subset, test_dataset, train_dataset.classes

def create_data_loaders(train_data, val_data, test_data, batch_size=32):
    """Create data loaders for training, validation, and testing."""
    train_loader = DataLoader(
        train_data,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_data,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )

    test_loader = DataLoader(
        test_data,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )

    return train_loader, val_loader, test_loader

def setup_datasets(data_directory, batch_size=32, val_split=0.2, apply_augmentation=False):
    """Setup and prepare datasets and loaders."""
    # Generate transform pipelines
    train_transform, test_transform = generate_transformation_pipelines(apply_augmentation)

    # Load and partition datasets
    train_data, val_data, test_data, class_list = load_and_partition_data(
        data_directory, train_transform, test_transform, val_split
    )

    # Create data loaders
    train_loader, val_loader, test_loader = create_data_loaders(
        train_data, val_data, test_data, batch_size
    )

    return train_loader, val_loader, test_loader, class_list

def evaluate_model(model, data_loader, loss_fn, device):
    """Evaluate model performance on the given data loader."""
    # Set model to evaluation mode
    model.eval()
    total_loss = 0.0
    correct_count = 0
    total_count = 0

    # Disable gradient calculation for evaluation
    with torch.no_grad():
        for data, target in data_loader:
            # Transfer data to appropriate device
            data, target = data.to(device), target.to(device)

            # Compute model output
            output = model(data)
            
            # Compute batch loss
            batch_loss = loss_fn(output, target)
            total_loss += batch_loss.item()
            
            # Determine predictions and calculate accuracy
            predictions = torch.argmax(output, dim=1)
            total_count += target.shape[0]
            correct_count += (predictions == target).sum().item()

    # Compute final metrics
    mean_loss = total_loss / len(data_loader)
    accuracy_pct = (correct_count / total_count) * 100

    return mean_loss, accuracy_pct

def run_training_epoch(model, data_loader, optimizer, loss_fn, device):
    """Run one training epoch."""
    # Set model to training mode
    model.train()
    total_loss = 0.0
    correct_count = 0
    total_count = 0

    # Iterate through batches with progress visualization
    with tqdm(data_loader, unit="batch") as progress_bar:
        for data, target in progress_bar:
            # Move batch to device
            data, target = data.to(device), target.to(device)

            # Clear accumulated gradients
            optimizer.zero_grad()
            
            # Forward computation
            output = model(data)
            batch_loss = loss_fn(output, target)
            
            # Backward computation and parameter update
            batch_loss.backward()
            optimizer.step()
            
            # Track statistics
            total_loss += batch_loss.item()
            predictions = torch.argmax(output, dim=1)
            total_count += target.shape[0]
            correct_count += (predictions == target).sum().item()
            
            # Update progress information
            progress_bar.set_postfix({
                'loss': f'{batch_loss.item():.4f}',
                'acc': f'{100.0 * correct_count / total_count:.1f}%'
            })

    # Calculate final epoch metrics
    mean_loss = total_loss / len(data_loader)
    accuracy_pct = (correct_count / total_count) * 100

    return mean_loss, accuracy_pct

def execute_training_workflow(config=None):
    """Execute training and validation workflow with the given configuration."""
    with wandb.init(config=config) as run:
        config = wandb.config

        # Set descriptive run name
        run.name = (f"filters_{'-'.join(map(str, config.conv_filters))}_"
                   f"dense_{'-'.join(map(str, config.dense_units))}_"
                   f"lr_{config.learning_rate:.0e}_"
                   f"bs_{config.batch_size}")

        # Setup datasets
        train_loader, val_loader, _, class_names = setup_datasets(
            data_directory='/kaggle/input/nature-12k/inaturalist_12K',
            batch_size=config.batch_size,
            apply_augmentation=config.data_augmentation
        )

        # Construct network
        model = construct_vision_network(
            {
                'conv_filters': config.conv_filters,
                'kernel_sizes': config.kernel_sizes,
                'dense_units': config.dense_units,
                'dropout_rate': config.dropout_rate,
                'use_batchnorm': config.use_batchnorm,
                'activation': config.activation
            },
            output_classes=len(class_names)
        )

        # Configure device
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = model.to(device)

        # Initialize optimizer and loss function
        optimizer = optim.Adam(
            model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay
        )
        loss_fn = nn.CrossEntropyLoss()

        # Training loop
        for epoch in range(1, 21):  # Fixed 20 epochs
            # Train for one epoch
            train_loss, train_acc = run_training_epoch(
                model, train_loader, optimizer, loss_fn, device
            )

            # Validate
            val_loss, val_acc = evaluate_model(
                model, val_loader, loss_fn, device
            )

            # Log metrics
            wandb.log({
                'epoch': epoch,
                'train_loss': train_loss,
                'train_acc': train_acc,
                'val_loss': val_loss,
                'val_acc': val_acc
            })

def configure_hyperparameter_sweep():
    """Configure hyperparameter sweep settings."""
    return {
        'method': 'bayes',
        'metric': {'name': 'val_acc', 'goal': 'maximize'},
        'parameters': {
            'conv_filters': {
                'values': [
                    [32, 32, 32, 32, 32],
                    [64, 64, 64, 64, 64],
                    [16, 32, 64, 128, 256],
                    [256, 128, 64, 32, 16]
                ]
            },
            'kernel_sizes': {
                'values': [
                    [3, 3, 3, 3, 3],
                    [5, 5, 5, 5, 5],
                    [3, 5, 3, 5, 3]
                ]
            },
            'dense_units': {
                'values': [
                    [64],
                    [128],
                    [64, 128],
                    [256, 128]
                ]
            },
            'learning_rate': {
                'values': [1e-3, 1e-4]
            },
            'weight_decay': {
                'values': [0, 0.0001, 0.001, 0.01]
            },
            'dropout_rate': {
                'values': [0.0, 0.2, 0.3, 0.5]
            },
            'use_batchnorm': {
                'values': [True, False]
            },
            'batch_size': {
                'values': [32, 64, 128]
            },
            'data_augmentation': {
                'values': [True, False]
            },
            'activation': {
                'values': ['relu', 'leaky_relu', 'gelu']
            }
        }
    }

def main():
    """Main function to execute the hyperparameter sweep."""
    # Initialize wandb
    wandb.login(key="49f8f505158ee3693f0cacf0a82118bd4e636e8c")

    # Configure hyperparameter sweep
    sweep_config = configure_hyperparameter_sweep()

    # Create and run sweep
    sweep_id = wandb.sweep(sweep_config, project='DA6401_A2')
    wandb.agent(sweep_id, function=execute_training_workflow, count=30)

if __name__ == '__main__':
    main()
