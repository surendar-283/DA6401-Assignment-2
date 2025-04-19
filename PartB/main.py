import os
import torch
import torch.optim as optim
import wandb
from model_utils import (
    prepare_data_loaders,
    construct_inception_model,
    execute_training_epoch,
    evaluate_model
)

# Hyperparameter search configuration
hyperparameter_config = {
    'method': 'bayes',
    'metric': {'name': 'validation_accuracy', 'goal': 'maximize'},
    'parameters': {
        'fc_neurons': {'values': [[256], [128], [64]]},
        'unfreeze_count': {'values': [None, 20, 30]},
        'dropout_rate': {'values': [0, 0.2, 0.3]},
        'use_augmentation': {'values': ['yes', 'no']},
        'batch_size': {'values': [128, 64, 256]},
        'training_epochs': {'values': [5, 10]}
    }
}

def hyperparameter_optimization_run():
    """Training function for hyperparameter sweep."""
    wandb.init(project='NatureClassifier_Project')
    config = wandb.config
    
    # Create meaningful run name
    experiment_name = (
        f"batch{config.batch_size}_"
        f"fc{config.fc_neurons}_"
        f"drop{config.dropout_rate}_"
        f"unfrozen{config.unfreeze_count}_"
        f"augment{config.use_augmentation}_"
        f"epochs{config.training_epochs}"
    )
    wandb.run.name = experiment_name

    # Setup device, data, and model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    train_loader, val_loader = prepare_data_loaders(config.batch_size, config.use_augmentation)
    model = construct_inception_model(config.fc_neurons, config.dropout_rate, config.unfreeze_count)
    model.to(device)
    
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()))
    best_validation_accuracy = 0.0
    
    # Training loop
    for epoch in range(config.training_epochs):
        train_loss, train_accuracy = execute_training_epoch(model, train_loader, optimizer, device)
        val_loss, val_accuracy = evaluate_model(model, val_loader, device)
        
        print(f"Epoch {epoch+1}/{config.training_epochs} | "
              f"Train Loss: {train_loss:.4f} | Train Acc: {train_accuracy:.4f} | "
              f"Val Loss: {val_loss:.4f} | Val Acc: {val_accuracy:.4f}")
              
        wandb.log({
            'epoch': epoch+1,
            'training_loss': train_loss,
            'training_accuracy': train_accuracy,
            'validation_loss': val_loss,
            'validation_accuracy': val_accuracy
        })
        
        if val_accuracy > best_validation_accuracy:
            best_validation_accuracy = val_accuracy
            
    wandb.log({'best_validation_accuracy': best_validation_accuracy})
    
    # Save the best model
    torch.save(model.state_dict(), '/kaggle/working/best_model.pth')
    print(f"Training completed. Best validation accuracy: {best_validation_accuracy:.4f}")
    
    return best_validation_accuracy

def main():
    """Main function to initialize and run the training process."""
    try:
        print("Setting up WandB and starting hyperparameter optimization...")
        
        # Set up WandB with API key
        wandb.login(key="49f8f505158ee3693f0cacf0a82118bd4e636e8c")
        
        # Create sweep and agent
        sweep_id = wandb.sweep(hyperparameter_config, project='NatureClassifier_Project')
        print(f"Sweep created with ID: {sweep_id}")
        
        # Start sweep agent
        wandb.agent(sweep_id, function=hyperparameter_optimization_run)
        
        print("Hyperparameter optimization completed successfully.")
        
    except Exception as e:
        print(f"An error occurred during training: {str(e)}")
        import traceback
        traceback.print_exc()

# This part is commented out since we'll call main() externally
# if __name__ == '__main__':
#     main()