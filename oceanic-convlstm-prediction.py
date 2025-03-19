import os
import glob
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.colors import LinearSegmentedColormap
from datetime import datetime, timedelta
import rasterio
from sklearn.model_selection import train_test_split
import torchvision.transforms as transforms
from tqdm import tqdm

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# Configuration
class Config:
    # Data paths and parameters
    DATA_DIR = "info"  # Directory where your GeoTIFF files are stored
    VARIABLES = ['SST', 'SSS', 'CHL', 'TURB']  # Variables to predict
    SEQUENCE_LENGTH = 5  # Number of time steps to use as input (4 for history, 1 for current)
    FORECAST_HORIZON = 4  # Number of time steps to predict
    BATCH_SIZE = 4
    IMAGE_SIZE = (128, 128)  # Resize images to this size for model input
    TEST_SIZE = 0.2  # Fraction of data to use for testing
    
    # Model parameters
    HIDDEN_DIM = 64
    KERNEL_SIZE = (3, 3)
    NUM_LAYERS = 3
    DROPOUT = 0.2
    
    # Training parameters
    LEARNING_RATE = 1e-3
    EPOCHS = 15
    EARLY_STOPPING_PATIENCE = 10
    
    # Visualization
    CMAP = {
        'SST': 'coolwarm',
        'SSS': 'viridis',
        'CHL': 'YlGn',
        'TURB': 'YlOrBr'
    }

config = Config()

# Custom Conv-LSTM Cell 
class ConvLSTMCell(nn.Module):
    def __init__(self, input_dim, hidden_dim, kernel_size, bias=True):
        super(ConvLSTMCell, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.kernel_size = kernel_size
        self.padding = kernel_size[0] // 2, kernel_size[1] // 2
        self.bias = bias
        
        self.conv = nn.Conv2d(
            in_channels=self.input_dim + self.hidden_dim,
            out_channels=4 * self.hidden_dim,  # for the four gates
            kernel_size=self.kernel_size,
            padding=self.padding,
            bias=self.bias
        )
    
    def forward(self, input_tensor, cur_state):
        h_cur, c_cur = cur_state
        
        # Concatenate along channel axis
        combined = torch.cat([input_tensor, h_cur], dim=1)
        
        # Convolution
        conv_output = self.conv(combined)
        
        # Split the convolution output into the four gates
        cc_i, cc_f, cc_o, cc_g = torch.split(conv_output, self.hidden_dim, dim=1)
        
        # Apply activations
        i = torch.sigmoid(cc_i)
        f = torch.sigmoid(cc_f)
        o = torch.sigmoid(cc_o)
        g = torch.tanh(cc_g)
        
        # Update cell state and hidden state
        c_next = f * c_cur + i * g
        h_next = o * torch.tanh(c_next)
        
        return h_next, c_next
    
    def init_hidden(self, batch_size, image_size):
        height, width = image_size
        return (torch.zeros(batch_size, self.hidden_dim, height, width, device=self.conv.weight.device),
                torch.zeros(batch_size, self.hidden_dim, height, width, device=self.conv.weight.device))

# Conv-LSTM Model
class ConvLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, kernel_size, num_layers, batch_first=True, dropout=0.0):
        super(ConvLSTM, self).__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.kernel_size = kernel_size
        self.num_layers = num_layers
        self.batch_first = batch_first
        self.dropout = dropout
        
        cell_list = []
        for i in range(self.num_layers):
            cur_input_dim = self.input_dim if i == 0 else self.hidden_dim
            cell_list.append(ConvLSTMCell(cur_input_dim, self.hidden_dim, self.kernel_size))
            
        self.cell_list = nn.ModuleList(cell_list)
        self.dropout_layer = nn.Dropout(self.dropout)
    
    def forward(self, input_tensor, hidden_state=None):
        """
        input_tensor: (batch, seq_len, channels, height, width)
        hidden_state: list of (h, c) for each layer
        """
        b, seq_len, _, h, w = input_tensor.size()
        
        # Initialize hidden state if not provided
        if hidden_state is None:
            hidden_state = self._init_hidden(batch_size=b, image_size=(h, w))
            
        layer_output_list = []
        last_state_list = []
        
        # For each layer
        for layer_idx in range(self.num_layers):
            h, c = hidden_state[layer_idx]
            output_inner = []
            
            # Process sequence
            for t in range(seq_len):
                if layer_idx == 0:
                    # First layer gets the input
                    x = input_tensor[:, t, :, :, :]
                else:
                    # Subsequent layers get output from previous layer
                    x = self.dropout_layer(layer_output_list[-1][:, t, :, :, :])
                
                h, c = self.cell_list[layer_idx](x, (h, c))
                output_inner.append(h)
                
            # Stack the outputs along sequence dimension
            layer_output = torch.stack(output_inner, dim=1)
            layer_output_list.append(layer_output)
            last_state_list.append((h, c))
            
        return layer_output_list[-1], last_state_list
    
    def _init_hidden(self, batch_size, image_size):
        init_states = []
        for i in range(self.num_layers):
            init_states.append(self.cell_list[i].init_hidden(batch_size, image_size))
        return init_states

# Full prediction model
class OceanicVariablePredictor(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim, kernel_size, num_layers, seq_length, forecast_horizon, dropout=0.0):
        super(OceanicVariablePredictor, self).__init__()
        
        self.input_dim = input_dim  # Number of variables (4 for SST, SSS, CHL, TURB)
        self.output_dim = output_dim  # Same as input_dim for our case
        self.hidden_dim = hidden_dim
        self.kernel_size = kernel_size
        self.num_layers = num_layers
        self.seq_length = seq_length
        self.forecast_horizon = forecast_horizon
        
        # Encoder (processes input sequence)
        self.encoder = ConvLSTM(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            kernel_size=kernel_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout
        )
        
        # Decoder (generates predictions)
        self.decoder = ConvLSTM(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            kernel_size=kernel_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout
        )
        
        # Output projection to get final predictions
        self.output_conv = nn.Conv2d(hidden_dim, output_dim, kernel_size=1)
    
    def forward(self, x):
        """
        x: Input tensor of shape (batch_size, seq_length, channels, height, width)
        Returns predictions of shape (batch_size, forecast_horizon, channels, height, width)
        """
        batch_size, _, channels, height, width = x.size()
        
        # Encode the input sequence
        _, encoder_hidden_states = self.encoder(x)
        
        # Initialize decoder input (use the last input frame as initial decoder input)
        decoder_input = x[:, -1:, :, :, :]  # Shape: (batch_size, 1, channels, height, width)
        
        # Initialize list to store predictions
        outputs = []
        
        # Generate predictions auto-regressively
        for t in range(self.forecast_horizon):
            # Pass through decoder
            decoder_output, encoder_hidden_states = self.decoder(decoder_input, encoder_hidden_states)
            
            # Project to get prediction
            current_pred = self.output_conv(decoder_output[:, 0])  # Shape: (batch_size, channels, height, width)
            
            # Reshape for next input
            decoder_input = current_pred.unsqueeze(1)  # Shape: (batch_size, 1, channels, height, width)
            
            # Store prediction
            outputs.append(current_pred.unsqueeze(1))  # Add a time dimension
        
        # Concatenate along time dimension
        outputs = torch.cat(outputs, dim=1)  # Shape: (batch_size, forecast_horizon, channels, height, width)
        
        return outputs

# Dataset class for oceanic variables
class OceanicVariableDataset(Dataset):
    def __init__(self, data_dir, variables, sequence_length, forecast_horizon, transform=None):
        """
        data_dir: Directory containing the GeoTIFF files
        variables: List of variables to use (e.g., ['SST', 'SSS', 'CHL', 'TURB'])
        sequence_length: Number of time steps to use as input
        forecast_horizon: Number of time steps to predict
        transform: Optional transformations to apply to the images
        """
        self.data_dir = data_dir
        self.variables = variables
        self.sequence_length = sequence_length
        self.forecast_horizon = forecast_horizon
        self.transform = transform
        
        # Get all dates from filenames
        all_files = glob.glob(os.path.join(data_dir, f"{variables[0]}*.tif"))
        self.dates = sorted([os.path.basename(f).split('_')[1].split('.')[0] for f in all_files])
        
        # Create date to index mapping
        self.date_to_idx = {date: i for i, date in enumerate(self.dates)}
        
        # Create valid sequences
        self.sequences = []
        for i in range(len(self.dates) - (sequence_length + forecast_horizon - 1)):
            input_dates = self.dates[i:i+sequence_length]
            target_dates = self.dates[i+sequence_length:i+sequence_length+forecast_horizon]
            self.sequences.append((input_dates, target_dates))
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        input_dates, target_dates = self.sequences[idx]
        
        # Load input sequence
        inputs = []
        for date in input_dates:
            # Load all variables for this date
            var_data = []
            for var in self.variables:
                file_path = os.path.join(self.data_dir, f"{var}_{date}.tif")
                with rasterio.open(file_path) as src:
                    # Read the normalized band (index 1)
                    data = src.read(2)  # Assuming normalized data is in band 2
                    if self.transform:
                        data = self.transform(data)
                    var_data.append(data)
            
            # Stack variables along channel dimension
            date_data = np.stack(var_data)
            inputs.append(date_data)
        
        # Load target sequence
        targets = []
        for date in target_dates:
            var_data = []
            for var in self.variables:
                file_path = os.path.join(self.data_dir, f"{var}_{date}.tif")
                with rasterio.open(file_path) as src:
                    data = src.read(2)  # Assuming normalized data is in band 2
                    if self.transform:
                        data = self.transform(data)
                    var_data.append(data)
            
            date_data = np.stack(var_data)
            targets.append(date_data)
        
        # Convert to tensors
        inputs = torch.tensor(np.stack(inputs), dtype=torch.float32)
        targets = torch.tensor(np.stack(targets), dtype=torch.float32)
        
        return inputs, targets

# Function to load and preprocess data
def load_data(config):
    print("Loading and preprocessing data...")
    
    # Define transformations (resize, normalize, etc.)
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize(config.IMAGE_SIZE),
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.numpy())
    ])
    
    # Create dataset
    dataset = OceanicVariableDataset(
        data_dir=config.DATA_DIR,
        variables=config.VARIABLES,
        sequence_length=config.SEQUENCE_LENGTH,
        forecast_horizon=config.FORECAST_HORIZON,
        transform=transform
    )
    
    # Split into train and test sets
    train_size = int((1 - config.TEST_SIZE) * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        num_workers=4
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=4
    )
    
    return train_loader, test_loader, dataset

# Training function
def train_model(model, train_loader, test_loader, config):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    model = model.to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=config.LEARNING_RATE)
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5, verbose=True
    )
    
    # For early stopping
    best_loss = float('inf')
    patience_counter = 0
    
    # Training history
    history = {
        'train_loss': [],
        'val_loss': []
    }
    
    print(f"Starting training for {config.EPOCHS} epochs...")
    for epoch in range(config.EPOCHS):
        # Training
        model.train()
        train_loss = 0.0
        train_batches = 0
        
        for inputs, targets in tqdm(train_loader, desc=f"Epoch {epoch+1}/{config.EPOCHS}"):
            inputs = inputs.to(device)
            targets = targets.to(device)
            
            optimizer.zero_grad()
            
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            train_batches += 1
        
        avg_train_loss = train_loss / train_batches
        history['train_loss'].append(avg_train_loss)
        
        # Validation
        model.eval()
        val_loss = 0.0
        val_batches = 0
        
        with torch.no_grad():
            for inputs, targets in test_loader:
                inputs = inputs.to(device)
                targets = targets.to(device)
                
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                
                val_loss += loss.item()
                val_batches += 1
        
        avg_val_loss = val_loss / val_batches
        history['val_loss'].append(avg_val_loss)
        
        # Learning rate scheduling
        scheduler.step(avg_val_loss)
        
        # Print epoch results
        print(f"Epoch {epoch+1}/{config.EPOCHS}, Train Loss: {avg_train_loss:.6f}, Val Loss: {avg_val_loss:.6f}")
        
        # Early stopping
        if avg_val_loss < best_loss:
            best_loss = avg_val_loss
            patience_counter = 0
            # Save the best model
            torch.save(model.state_dict(), 'best_model.pth')
            print(f"Saved best model with validation loss: {best_loss:.6f}")
        else:
            patience_counter += 1
            if patience_counter >= config.EARLY_STOPPING_PATIENCE:
                print(f"Early stopping triggered after {epoch+1} epochs!")
                break
    
    # Load the best model
    model.load_state_dict(torch.load('best_model.pth'))
    
    return model, history

# Visualization functions
def plot_training_history(history):
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 1, 1)
    plt.plot(history['train_loss'], label='Training Loss')
    plt.plot(history['val_loss'], label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('training_history.png')
    plt.close()

def visualize_predictions(model, dataset, date_str, config, save_dir='predictions'):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()
    
    # Find the sequence that includes the given date
    for idx, (input_dates, target_dates) in enumerate(dataset.sequences):
        if date_str in input_dates:
            # Get the model inputs and targets for this sequence
            inputs, targets = dataset[idx]
            inputs = inputs.unsqueeze(0).to(device)  # Add batch dimension
            
            # Get model predictions
            with torch.no_grad():
                outputs = model(inputs)
            
            # Create the output directory if it doesn't exist
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            
            # Plot the input sequence, ground truth, and predictions
            fig, axs = plt.subplots(len(config.VARIABLES), config.SEQUENCE_LENGTH + config.FORECAST_HORIZON, 
                                    figsize=(20, 12))
            
            # Find index of the specified date in the input sequence
            date_idx = input_dates.index(date_str)
            
            # Set the title for the entire figure
            fig.suptitle(f"Predictions starting from {date_str}", fontsize=16)
            
            # Plot inputs
            for var_idx, var_name in enumerate(config.VARIABLES):
                for t in range(config.SEQUENCE_LENGTH):
                    ax = axs[var_idx, t]
                    im = ax.imshow(inputs[0, t, var_idx].cpu().numpy(), 
                                   cmap=config.CMAP[var_name], vmin=0, vmax=1)
                    ax.set_title(f"Input: {input_dates[t]}")
                    ax.set_ylabel(var_name)
                    ax.axis('off')
                    
                    # Highlight the specified date
                    if t == date_idx:
                        ax.set_title(f"Input: {input_dates[t]} (Selected)", color='red')
                        for spine in ax.spines.values():
                            spine.set_edgecolor('red')
                            spine.set_linewidth(2)
            
            # Plot ground truth and predictions side by side
            for var_idx, var_name in enumerate(config.VARIABLES):
                for t in range(config.FORECAST_HORIZON):
                    ax = axs[var_idx, config.SEQUENCE_LENGTH + t]
                    
                    # Plot predictions
                    im = ax.imshow(outputs[0, t, var_idx].cpu().numpy(), 
                                   cmap=config.CMAP[var_name], vmin=0, vmax=1)
                    ax.set_title(f"Pred: {target_dates[t]}")
                    ax.axis('off')
                    
                    # Add colorbar for each variable
                    if t == config.FORECAST_HORIZON - 1:
                        fig.colorbar(im, ax=ax)
            
            plt.tight_layout(rect=[0, 0, 1, 0.95])  # Make room for the title
            plt.savefig(os.path.join(save_dir, f"prediction_{date_str}.png"), dpi=200)
            plt.close()
            
            # Also save individual predictions for each variable
            for var_idx, var_name in enumerate(config.VARIABLES):
                fig, axs = plt.subplots(1, config.FORECAST_HORIZON, figsize=(16, 4))
                fig.suptitle(f"{var_name} Predictions starting from {date_str}", fontsize=16)
                
                for t in range(config.FORECAST_HORIZON):
                    ax = axs[t]
                    im = ax.imshow(outputs[0, t, var_idx].cpu().numpy(), 
                                   cmap=config.CMAP[var_name], vmin=0, vmax=1)
                    ax.set_title(f"Day {t+1}: {target_dates[t]}")
                    ax.axis('off')
                
                plt.colorbar(im, ax=axs, fraction=0.046, pad=0.04)
                plt.tight_layout(rect=[0, 0, 1, 0.95])
                plt.savefig(os.path.join(save_dir, f"{var_name}_{date_str}.png"), dpi=200)
                plt.close()
            
            print(f"Saved predictions for date {date_str}")
            return
    
    print(f"Date {date_str} not found in the dataset.")

# Main function to run everything
def main():
    # Check if CUDA is available
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA device: {torch.cuda.get_device_name(0)}")
    
    # Load data
    train_loader, test_loader, full_dataset = load_data(config)
    print(f"Dataset contains {len(full_dataset)} sequences")
    print(f"Train set: {len(train_loader.dataset)} sequences")
    print(f"Test set: {len(test_loader.dataset)} sequences")
    
    # Get input shape from a sample
    sample_inputs, _ = next(iter(train_loader))
    _, seq_len, channels, height, width = sample_inputs.shape
    print(f"Input shape: (batch_size, {seq_len}, {channels}, {height}, {width})")
    
    # Initialize model
    model = OceanicVariablePredictor(
        input_dim=len(config.VARIABLES),
        output_dim=len(config.VARIABLES),
        hidden_dim=config.HIDDEN_DIM,
        kernel_size=config.KERNEL_SIZE,
        num_layers=config.NUM_LAYERS,
        seq_length=config.SEQUENCE_LENGTH,
        forecast_horizon=config.FORECAST_HORIZON,
        dropout=config.DROPOUT
    )
    
    print(f"Model initialized with {sum(p.numel() for p in model.parameters())} parameters")
    
    # Train model
    model, history = train_model(model, train_loader, test_loader, config)
    
    # Plot training history
    plot_training_history(history)
    
    # Visualize predictions for a sample date from the dataset
    sample_date = full_dataset.dates[config.SEQUENCE_LENGTH]
    visualize_predictions(model, full_dataset, sample_date, config)
    
    print(f"Training complete. Model saved as 'best_model.pth'")
    print(f"Example prediction visualization saved in 'predictions' directory")

# Prediction function for a specific date
def predict_for_date(date_str, model_path='best_model.pth'):
    # Load the trained model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = OceanicVariablePredictor(
        input_dim=len(config.VARIABLES),
        output_dim=len(config.VARIABLES),
        hidden_dim=config.HIDDEN_DIM,
        kernel_size=config.KERNEL_SIZE,
        num_layers=config.NUM_LAYERS,
        seq_length=config.SEQUENCE_LENGTH,
        forecast_horizon=config.FORECAST_HORIZON,
        dropout=config.DROPOUT
    )
    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device)
    model.eval()
    
    # Create dataset
    dataset = OceanicVariableDataset(
        data_dir=config.DATA_DIR,
        variables=config.VARIABLES,
        sequence_length=config.SEQUENCE_LENGTH,
        forecast_horizon=config.FORECAST_HORIZON
    )
    
    # Visualize predictions
    visualize_predictions(model, dataset, date_str, config)

if __name__ == "__main__":
    # Run the main function
    main()
    
    # Uncomment to predict for a specific date after training
    predict_for_date('2019-06-04')
