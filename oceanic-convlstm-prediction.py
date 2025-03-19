import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os
import rasterio
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
from tqdm import tqdm

# Configuration
class Config:
    DATA_DIR = 'info'
    INPUT_SEQUENCE = 5  # Number of input time steps
    FUTURE_STEPS = 4    # Number of steps to predict
    BATCH_SIZE = 8
    EPOCHS = 15
    LEARNING_RATE = 0.001
    HIDDEN_DIM = 64
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    VARIABLES = ['SST', 'SSS', 'CHL', 'TURB']
    # Normalization ranges for each variable
    RANGES = {
        'SST': (0, 35),    # Temperature in Celsius
        'SSS': (30, 40),   # Salinity in PSU
        'CHL': (0, 10),    # Chlorophyll concentration
        'TURB': (0, 30)    # Turbidity
    }

class OceanDataset(Dataset):
    def __init__(self, data_dir, input_sequence, future_steps):
        self.data_dir = data_dir
        self.input_sequence = input_sequence
        self.future_steps = future_steps
        self.variables = Config.VARIABLES
        
        # Get all dates from filenames
        self.dates = self._get_dates()
        self.sequences = self._create_sequences()
        
    def _get_dates(self):
        dates = set()
        for file in os.listdir(self.data_dir):
            if file.endswith('.tif'):
                date = file.split('_')[1].replace('.tif', '')
                dates.add(date)
        return sorted(list(dates))
    
    def _load_and_normalize(self, date, var):
        filename = f"{var}_{date}.tif"
        filepath = os.path.join(self.data_dir, filename)
        
        if os.path.exists(filepath):
            with rasterio.open(filepath) as src:
                img = src.read(1).astype(np.float32)
                # Normalize based on variable type
                min_val, max_val = Config.RANGES[var]
                img = np.clip(img, min_val, max_val)
                img = (img - min_val) / (max_val - min_val)
                return img
        return None
    
    def _create_sequences(self):
        sequences = []
        for i in range(len(self.dates) - self.input_sequence - self.future_steps + 1):
            input_dates = self.dates[i:i + self.input_sequence]
            target_dates = self.dates[i + self.input_sequence:
                                    i + self.input_sequence + self.future_steps]
            
            # Load input sequence
            inputs = []
            for date in input_dates:
                date_data = []
                for var in self.variables:
                    img = self._load_and_normalize(date, var)
                    if img is None:
                        break
                    date_data.append(img)
                if len(date_data) == len(self.variables):
                    inputs.append(np.stack(date_data))
                    
            # Load target sequence
            targets = []
            for date in target_dates:
                date_data = []
                for var in self.variables:
                    img = self._load_and_normalize(date, var)
                    if img is None:
                        break
                    date_data.append(img)
                if len(date_data) == len(self.variables):
                    targets.append(np.stack(date_data))
            
            if len(inputs) == self.input_sequence and len(targets) == self.future_steps:
                sequences.append((np.stack(inputs), np.stack(targets)))
        
        return sequences
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        return torch.FloatTensor(self.sequences[idx][0]), torch.FloatTensor(self.sequences[idx][1])

class ConvLSTM(nn.Module):
    def __init__(self, input_channels, hidden_dim, kernel_size, num_layers):
        super(ConvLSTM, self).__init__()
        
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        # Encoder (ConvLSTM)
        self.conv_lstm = nn.ModuleList([
            nn.Conv3d(
                input_channels if i == 0 else hidden_dim,
                hidden_dim,
                kernel_size=(3, 3, 3),
                padding=(1, 1, 1)
            ) for i in range(num_layers)
        ])
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.Conv3d(hidden_dim, 32, 3, padding=1),
            nn.ReLU(),
            nn.Conv3d(32, input_channels, 3, padding=1)
        )
        
    def forward(self, x):
        batch_size, seq_len, channels, height, width = x.size()
        h = [torch.zeros(batch_size, self.hidden_dim, height, width).to(x.device)
             for _ in range(self.num_layers)]
        
        # Process input sequence
        for t in range(seq_len):
            for layer in range(self.num_layers):
                if layer == 0:
                    inp = x[:, t]
                else:
                    inp = h[layer-1]
                
                h[layer] = torch.tanh(self.conv_lstm[layer](
                    torch.cat([inp.unsqueeze(2), h[layer].unsqueeze(2)], dim=2)
                )[:, :, -1])
        
        # Generate predictions
        outputs = []
        current_input = h[-1]
        
        for _ in range(Config.FUTURE_STEPS):
            current_input = self.decoder(current_input.unsqueeze(2))[:, :, 0]
            outputs.append(current_input)
        
        return torch.stack(outputs, dim=1)

def train_model(model, train_loader, val_loader):
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=Config.LEARNING_RATE)
    
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    
    for epoch in range(Config.EPOCHS):
        # Training
        model.train()
        epoch_loss = 0
        for x, y in tqdm(train_loader, desc=f'Epoch {epoch+1}/{Config.EPOCHS}'):
            x, y = x.to(Config.DEVICE), y.to(Config.DEVICE)
            
            optimizer.zero_grad()
            output = model(x)
            loss = criterion(output, y)
            
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
        
        train_loss = epoch_loss / len(train_loader)
        train_losses.append(train_loss)
        
        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(Config.DEVICE), y.to(Config.DEVICE)
                output = model(x)
                val_loss += criterion(output, y).item()
        
        val_loss /= len(val_loader)
        val_losses.append(val_loss)
        
        print(f'Epoch {epoch+1}: Train Loss = {train_loss:.6f}, Val Loss = {val_loss:.6f}')
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), 'best_model.pth')
    
    return train_losses, val_losses

def visualize_predictions(model, test_loader):
    model.eval()
    x, y = next(iter(test_loader))
    x, y = x.to(Config.DEVICE), y.to(Config.DEVICE)
    
    with torch.no_grad():
        pred = model(x)
    
    # Plot results
    sample_idx = 0
    fig, axes = plt.subplots(Config.FUTURE_STEPS, len(Config.VARIABLES), 
                            figsize=(4*len(Config.VARIABLES), 4*Config.FUTURE_STEPS))
    
    for step in range(Config.FUTURE_STEPS):
        for var_idx, var_name in enumerate(Config.VARIABLES):
            # True vs Predicted
            true = y[sample_idx, step, var_idx].cpu()
            predicted = pred[sample_idx, step, var_idx].cpu()
            
            ax = axes[step, var_idx]
            im = ax.imshow(predicted, cmap='viridis')
            ax.set_title(f'{var_name} Step {step+1}\nTrue vs Pred')
            plt.colorbar(im, ax=ax)
    
    plt.tight_layout()
    plt.savefig('predictions.png')
    plt.close()

def main():
    print(f"Using device: {Config.DEVICE}")
    
    # Create dataset
    dataset = OceanDataset(Config.DATA_DIR, Config.INPUT_SEQUENCE, Config.FUTURE_STEPS)
    print(f"Created dataset with {len(dataset)} sequences")
    
    # Split dataset
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=Config.BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=Config.BATCH_SIZE)
    
    # Create model
    model = ConvLSTM(
        input_channels=len(Config.VARIABLES),
        hidden_dim=Config.HIDDEN_DIM,
        kernel_size=3,
        num_layers=2
    ).to(Config.DEVICE)
    
    # Train model
    train_losses, val_losses = train_model(model, train_loader, val_loader)
    
    # Plot training curves
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig('training_curves.png')
    plt.close()
    
    # Visualize predictions
    visualize_predictions(model, val_loader)
    
    print("Training completed and visualizations saved!")

if __name__ == "__main__":
    main()
