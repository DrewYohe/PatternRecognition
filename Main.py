import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import soundfile as sf
import sys

# Define the feedforward neural network
class SimpleCNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers):
        super(SimpleCNN, self).__init__()
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)
        self.flatten = nn.Flatten(start_dim=0)
    def forward(self, x):
        # Initialize hidden state with zeros
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)

        # Forward propagate LSTM
        out, _ = self.lstm(x, (h0, c0))  # out: tensor of shape (batch_size, seq_length, hidden_size)

        # Decode the hidden state of the last time step
        out = self.fc(out[:, -1, :])
        return out


sampleSound, fs = sf.read("SampleMusic.wav")
sampleSound = list(map(lambda x: x[0], sampleSound))
# Define hyperparameters
input_size = int(fs/2) # Input feature size
hidden_size = 256
output_size = 1  # Output size
num_epochs = 10
batch_size = 32
learning_rate = 0.001
numInputs = input_size 
numOutputs = 1
train = True

if train:
    x_train = []
    y_train = []
    for i in range(0,len(sampleSound)- input_size-2,500):
        x_train.append(sampleSound[i:(i+numInputs)])
        y_train.append((sampleSound[(i+numInputs):(i+numInputs+1):]))
    x_train = torch.FloatTensor(x_train)
    y_train = torch.FloatTensor(y_train)
    # Create DataLoader
    train_dataset = TensorDataset(x_train, y_train)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    # Initialize the model
    model = SimpleCNN(input_size, hidden_size, 1)

    # Define loss function and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)



    # Training loop
    for epoch in range(num_epochs):
        running_loss = 0.0
        for inputs, targets in train_loader:
            # Forward pass
            outputs = model(torch.unsqueeze(torch.FloatTensor(inputs),1))
            loss = criterion(outputs, targets)
            
            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * inputs.size(0)
        
        epoch_loss = running_loss / len(train_dataset)
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}')

    print('Training finished!')
    torch.save(model, "model.pt")
else:
    model = torch.load('model.pt')

model.eval()
currentSound = (sampleSound[105*fs:(105*fs+numInputs)])
soundLength = 4*input_size
for i in range(soundLength):
    prediction = model(torch.unsqueeze(torch.unsqueeze(torch.FloatTensor(currentSound[i:]),0),0))
    currentSound.append(prediction.item())
    if(i %100 == 0):
        print(f"generating sound {i/soundLength*100}%")
        print(currentSound[-1])

print('Audio Generated!')

sf.write("OutputMusic.wav",currentSound,fs)