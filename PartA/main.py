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

def create_transform_pipelines(use_augmentation=False):
    """Create image transformation pipelines for training and testing."""
    standard_transforms = [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ]

    if use_augmentation:
        training_pipeline = transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            *standard_transforms
        ])
    else:
        training_pipeline = transforms.Compose(standard_transforms)

    testing_pipeline = transforms.Compose(standard_transforms)

    return training_pipeline, testing_pipeline

def prepare_and_split_datasets(dataset_path, training_transforms, testing_transforms, validation_portion=0.2):
    """Load and split datasets into train, validation, and test sets."""
    # Load datasets
    training_dataset = ImageFolder(os.path.join(dataset_path, 'train'), transform=training_transforms)
    testing_dataset = ImageFolder(os.path.join(dataset_path, 'val'), transform=testing_transforms)

    # Split training into training and validation
    validation_size = int(validation_portion * len(training_dataset))
    adjusted_training_size = len(training_dataset) - validation_size
    training_subset, validation_subset = random_split(training_dataset, [adjusted_training_size, validation_size])

    return training_subset, validation_subset, testing_dataset, training_dataset.classes

def setup_data_loaders(training_data, validation_data, testing_data, batch_size=32):
    """Set up data loaders for training, validation, and testing."""
    training_loader = DataLoader(
        training_data,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )

    validation_loader = DataLoader(
        validation_data,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )

    testing_loader = DataLoader(
        testing_data,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )

    return training_loader, validation_loader, testing_loader

def initialize_datasets(dataset_path, batch_size=32, validation_portion=0.2, use_augmentation=False):
    """Initialize and prepare datasets and loaders."""
    # Create transform pipelines
    training_transforms, testing_transforms = create_transform_pipelines(use_augmentation)

    # Load and split datasets
    training_data, validation_data, testing_data, class_names = prepare_and_split_datasets(
        dataset_path, training_transforms, testing_transforms, validation_portion
    )

    # Set up data loaders
    training_loader, validation_loader, testing_loader = setup_data_loaders(
        training_data, validation_data, testing_data, batch_size
    )

    return training_loader, validation_loader, testing_loader, class_names

def assess_model_performance(network, data_loader, criterion, device):
    """Assess model performance on the given data loader."""
    network.eval()
    cumulative_loss = 0.0
    correct_predictions = 0
    total_samples = 0

    with torch.no_grad():
        for inputs, labels in data_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            # Forward pass
            outputs = network(inputs)
            loss = criterion(outputs, labels)

            # Calculate metrics
            cumulative_loss += loss.item()
            _, predicted = outputs.max(1)
            total_samples += labels.size(0)
            correct_predictions += predicted.eq(labels).sum().item()

    # Calculate average loss and accuracy
    avg_loss = cumulative_loss / len(data_loader)
    accuracy = 100.0 * correct_predictions / total_samples

    return avg_loss, accuracy

def execute_training_epoch(network, train_loader, optimizer, criterion, device):
    """Execute one training epoch."""
    network.train()
    cumulative_loss = 0.0
    correct_predictions = 0
    total_samples = 0

    with tqdm(train_loader, unit="batch") as progress_bar:
        for inputs, labels in progress_bar:
            inputs, labels = inputs.to(device), labels.to(device)

            # Reset gradients
            optimizer.zero_grad()

            # Forward pass
            outputs = network(inputs)
            loss = criterion(outputs, labels)

            # Backward pass and optimization
            loss.backward()
            optimizer.step()

            # Calculate metrics
            cumulative_loss += loss.item()
            _, predicted = outputs.max(1)
            total_samples += labels.size(0)
            correct_predictions += predicted.eq(labels).sum().item()

            # Update progress display
            progress_bar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'acc': f'{100.0 * correct_predictions / total_samples:.1f}%'
            })

    # Calculate average loss and accuracy
    avg_loss = cumulative_loss / len(train_loader)
    accuracy = 100.0 * correct_predictions / total_samples

    return avg_loss, accuracy

def execute_training_workflow(config=None):
    """Execute training and validation workflow with the given configuration."""
    with wandb.init(config=config) as run:
        config = wandb.config

        # Set descriptive run name
        run.name = (f"filters_{'-'.join(map(str, config.conv_filters))}_"
                   f"dense_{'-'.join(map(str, config.dense_units))}_"
                   f"lr_{config.learning_rate:.0e}_"
                   f"bs_{config.batch_size}")

        # Initialize datasets
        training_loader, validation_loader, _, class_names = initialize_datasets(
            dataset_path='/kaggle/input/nature-12k/inaturalist_12K',
            batch_size=config.batch_size,
            use_augmentation=config.data_augmentation
        )

        # Construct network
        network = construct_vision_network(
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
        network = network.to(device)

        # Initialize optimizer and loss function
        optimizer = optim.Adam(
            network.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay
        )
        criterion = nn.CrossEntropyLoss()

        # Training loop
        for epoch in range(1, 21):  # Fixed 20 epochs
            # Train for one epoch
            train_loss, train_acc = execute_training_epoch(
                network, training_loader, optimizer, criterion, device
            )

            # Validate
            val_loss, val_acc = assess_model_performance(
                network, validation_loader, criterion, device
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