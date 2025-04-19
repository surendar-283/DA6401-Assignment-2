import torch
import torchvision
import numpy as np
from torchvision import transforms
from torch.utils.data import DataLoader, Subset
from torchvision.datasets import ImageFolder
import torch.nn as nn

def create_image_transforms(use_augmentation):
    """Create image transformations with optional augmentation."""
    standard_transforms = [
        transforms.Resize((299, 299)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]
    
    if use_augmentation == 'yes':
        augmentation_transforms = [
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(20),
            transforms.ColorJitter(0.2, 0.2, 0.2, 0.2)
        ]
        return transforms.Compose(augmentation_transforms + standard_transforms)
    else:
        return transforms.Compose(standard_transforms)

def prepare_data_loaders(batch_size, use_augmentation):
    """Prepare data loaders for training and validation."""
    image_transform = create_image_transforms(use_augmentation)
    # Using Kaggle path
    complete_dataset = ImageFolder('/kaggle/input/nature-12k/inaturalist_12K/train', transform=image_transform)
    validation_percentage = 0.2

    # Create stratified split for validation
    labels = np.array(complete_dataset.targets)
    all_indices = np.arange(len(labels))
    training_idx, validation_idx = [], []
    
    # Distribute indices for stratified sampling
    for label_value in np.unique(labels):
        indices_for_class = all_indices[labels == label_value]
        # Shuffle the indices for this class
        np.random.seed(42)  # Add deterministic behavior
        np.random.shuffle(indices_for_class)
        # Calculate partition point 
        partition_idx = int((1 - validation_percentage) * len(indices_for_class))
        # Assign indices to respective sets
        training_idx += indices_for_class[:partition_idx].tolist()
        validation_idx += indices_for_class[partition_idx:].tolist()

    # Create dataset subsets using partitioned indices
    train_partition = Subset(complete_dataset, sorted(training_idx))
    validation_partition = Subset(complete_dataset, sorted(validation_idx))

    # Configure data loaders with appropriate settings
    train_loader = DataLoader(
        dataset=train_partition, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=2,
        pin_memory=True  # Added performance enhancement
    )
    validation_loader = DataLoader(
        dataset=validation_partition, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=2,
        pin_memory=True  # Added performance enhancement
    )

    # Return configured data loaders
    return train_loader, validation_loader

def construct_inception_model(fc_neurons, dropout_rate, unfreeze_count):
    """Construct and customize InceptionV3 model."""
    inception_model = torchvision.models.inception_v3(pretrained=True, aux_logits=True)
    inception_model.aux_logits = False
    inception_model.AuxLogits = None  # Remove auxiliary classifier

    # Freeze all parameters initially
    for param in inception_model.parameters():
        param.requires_grad = False

    # Selectively unfreeze layers if specified
    if unfreeze_count is not None:
        all_layers = list(inception_model.children())
        for layer in all_layers[-unfreeze_count:]:
            for param in layer.parameters():
                param.requires_grad = True

    # Create custom classifier
    input_features = inception_model.fc.in_features
    classifier_components = []
    
    for neuron_count in fc_neurons:
        classifier_components.append(nn.Linear(input_features, neuron_count))
        classifier_components.append(nn.ReLU())
        if dropout_rate > 0:
            classifier_components.append(nn.Dropout(dropout_rate))
        input_features = neuron_count
    
    # Add final classification layer (10 classes)
    classifier_components.append(nn.Linear(input_features, 10))
    inception_model.fc = nn.Sequential(*classifier_components)
    
    return inception_model

def execute_training_epoch(model, data_loader, optimizer, device):
    """Execute one training epoch."""
    model.train()
    samples_count, correct_predictions, running_loss = 0, 0, 0.0
    criterion = nn.CrossEntropyLoss()
    
    for inputs, targets in data_loader:
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        
        # Handle potential tuple output from inception
        if isinstance(outputs, tuple):
            outputs = outputs[0]
            
        batch_loss = criterion(outputs, targets)
        batch_loss.backward()
        optimizer.step()
        
        running_loss += batch_loss.item() * inputs.size(0)
        predictions = torch.argmax(outputs, 1)
        correct_predictions += (predictions == targets).sum().item()
        samples_count += inputs.size(0)
        
    return running_loss / samples_count, correct_predictions / samples_count

def evaluate_model(model, data_loader, device):
    """Evaluate model on validation data."""
    model.eval()
    samples_count, correct_predictions, running_loss = 0, 0, 0.0
    criterion = nn.CrossEntropyLoss()
    
    with torch.no_grad():
        for inputs, targets in data_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            
            # Handle potential tuple output from inception
            if isinstance(outputs, tuple):
                outputs = outputs[0]
                
            batch_loss = criterion(outputs, targets)
            running_loss += batch_loss.item() * inputs.size(0)
            predictions = torch.argmax(outputs, 1)
            correct_predictions += (predictions == targets).sum().item()
            samples_count += inputs.size(0)
            
    return running_loss / samples_count, correct_predictions / samples_count
