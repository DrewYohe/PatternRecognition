import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import sys
import os
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.animation as animation
import numpy as np
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import pandas as pd
import geopandas as gpd
from scipy.interpolate import griddata
from scipy.interpolate import RegularGridInterpolator
from shapely.geometry import Point
from torchviz import make_dot

num_weather_centers = 130;
lon_size = 280
lat_size = 100
param_scale = 50.0
input_size = 2
hidden_size = 256
output_size = 1
num_epochs = 256
batch_size = 2
learning_rate = 0.001
tolerance = 3
all_data = []
all_data_gt = []
all_lats = []
all_lons = []
ax = []
cbar = 1
#Per run params
pull_data = False
clean_data = False
train_net = False

class CNN3D(nn.Module):
    def __init__(self):
        super(CNN3D, self).__init__()
        
        # Define 3D convolutional layers
        self.conv1 = nn.Conv3d(in_channels=1, out_channels=32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv3d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        
        # Define max pooling layers
        self.pool = nn.MaxPool3d(kernel_size=2, stride=2)
        
        # Define fully connected layers
        self.fc1 = nn.Linear(64 * 6 * 2 * 70, 512)  # Adjust input size based on the data dimensions
        self.fc2 = nn.Linear(512, 256)  # Output size: 20*56
        self.fc3 = nn.Linear(256, 20*56)  # Output size: 20*56
        # Define activation function
        self.relu = nn.ReLU()

    def forward(self, x):
        # Apply 3D convolution, activation, and pooling
        x = self.pool(F.leaky_relu(self.conv1(x)))
        x = self.pool(F.leaky_relu(self.conv2(x)))
        
        # Reshape the tensor for fully connected layers
        x = x.view(-1,64 * 6 * 2 * 70)
        
        # Apply fully connected layers
        x = F.leaky_relu(self.fc1(x))
        x = F.leaky_relu(self.fc2(x))
        x = self.fc3(x)
        return x


class Densify(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(Densify, self).__init__()
        self.fc1 = nn.Linear(input_size, int(hidden_size/2))
        self.fc2 = nn.Linear(int(hidden_size/2), hidden_size)
        self.fc3 = nn.Linear(hidden_size, int(hidden_size/2))
        self.out = nn.Linear(int(hidden_size/2), output_size)
    def forward(self, x):
        x = F.leaky_relu(self.fc1(x))
        x = F.leaky_relu(self.fc2(x))
        x = F.leaky_relu(self.fc3(x))
        x = self.out(x)
        return x

def DisplayMap(lons, lats, data, this_df, date_time, param = "T_CALC", title = "",check_border = False):

    # Load the shapefile of the US border
    us_border = gpd.read_file("geo/cb_2022_us_nation_20m.shx")

    # Create a 2D grid for longitude and latitude
    lon_grid, lat_grid = np.meshgrid(np.linspace(-130, -60, lon_size), np.linspace(25, 50, lat_size))

    # Interpolate data onto the 2D grid
    data_grid = griddata((lons, lats), data, (lon_grid, lat_grid), method='nearest')
    data_mask = np.copy(data_grid)

    # Iterate over the grid and set values to NaN if they fall outside the US border
    if check_border:
        for i in range(len(lon_grid)):
            for j in range(len(lon_grid[0])):
                lon = lon_grid[i][j]
                lat = lat_grid[i][j]
                if not us_border.geometry.contains(Point(lon, lat)).any():
                    data_mask[i][j] = False
                else:
                    data_mask[i][j] = True
        np.save("geo/border_mask.npy", data_mask)
    else:
        data_mask = np.load("geo/border_mask.npy")

    # Apply data mask
    for i in range(len(lon_grid)):
        for j in range(len(lon_grid[0])):
            lon = lon_grid[i][j]
            lat = lat_grid[i][j]
            if not data_mask[i][j]:
                data_grid[i][j] = np.nan

    # Create a figure and axis with Cartopy projection
    fig = plt.figure(figsize=(10, 6))
    ax = fig.add_subplot(1, 1, 1, projection=ccrs.LambertConformal())

    # Add state borders
    ax.add_feature(cfeature.STATES.with_scale('10m'), edgecolor='gray', linewidth=0.5)

    # Add ocean background outside the US map
    ax.add_feature(cfeature.OCEAN, edgecolor='none', facecolor='dimgray')

    # Plot the data as a colormap
    # Create a custom colormap with transparency
    colors = plt.cm.turbo(np.linspace(0, 1, 256))
    colors[:, 3] = 0.7  # Set alpha value (transparency) to 0.5
    transparent_cmap = LinearSegmentedColormap.from_list('transparent_cmap', colors)
    sc = ax.contourf(lon_grid, lat_grid, data_grid, cmap=transparent_cmap, levels = 20, transform=ccrs.PlateCarree())

    # Plot scatter plot points of weather stations
    filtered_df = this_df[(this_df["DATETIME"]==date_time) & (this_df[param]>-9999)]
    scatter = ax.scatter(filtered_df["LONGITUDE"], filtered_df["LATTITUDE"], c=filtered_df[param], cmap='turbo', transform=ccrs.PlateCarree())

    # Add map features
    ax.coastlines()
    ax.gridlines()

    # Add colorbar
    cbar = plt.colorbar(sc, ax=ax, orientation='vertical', shrink=0.5)
    cbar.set_label('Temperature °C')

    # Set plot extent to the United States
    ax.set_extent([-125, -65, 20, 50], ccrs.PlateCarree())

    error = 0
    stations = 0
    interp_func = RegularGridInterpolator((np.linspace(-130, -60, lon_size), np.linspace(25, 50, lat_size)), np.transpose(data_grid))
    for index, row in filtered_df.iterrows():
        tempGuess = interp_func([row["LONGITUDE"],row["LATTITUDE"]])
        stations += 1
        if str(tempGuess[0]) != 'nan':
            error += np.abs(row[param] - tempGuess[0])*np.abs(row[param] - tempGuess[0])
        else:
            #Unable to classify
            error += np.abs(row[param])*np.abs(row[param])

    error = error/stations
    error = np.round(error, 2)
    print(f"The error of {title} is {error}")
    # Show the plot
    plt.title(title)
    plt.show()

def DisplayLinearInterpolatedMap(this_df, date_time, param = "T_CALC"):
    filtered_df = this_df[(this_df['DATETIME'] == date_time) & (this_df[param] > -9999) & (this_df['LONGITUDE'] > -130) & (this_df['LONGITUDE'] < -60) & (this_df['LATTITUDE'] > 25) & (this_df['LATTITUDE'] < 50)]
    lon = filtered_df['LONGITUDE']
    lat = filtered_df['LATTITUDE']
    data = filtered_df[param] 
    DisplayMap(lon, lat, data, this_df, date_time, param)

def DisplayDensifiedMap(this_df, date_time, param = "T_CALC"):
    filtered_df = this_df[(this_df["DATETIME"]==date_time) & (this_df[param]>-9999)]
    densification_net = TrainDensificationModel(filtered_df,date_time,param)
    lon = np.linspace(-130, -60, lon_size)
    lat = np.linspace(25, 50, lat_size)
    lons = []
    lats = []
    data = []
    densification_net.eval()
    for i in range(len(lon)):
        for j in range(len(lat)):
            lons.append(lon[i])
            lats.append(lat[j])
            feature_tensor = []
            feature_tensor.append((lon[i]+130.0)/35.0)
            feature_tensor.append((lat[j]-37.5)/12.5)
            data.append(densification_net(torch.FloatTensor(feature_tensor)).item()*param_scale)
    DisplayMap(lons, lats, data, this_df, date_time, param)

def TrainDensificationModel(this_df,date_time, param = "T_CALC"):
    x_train = []
    y_train = []
    df_one_date_time = this_df[(this_df["DATETIME"] == date_time) & (this_df[param]>-9999)]
    for index, row in df_one_date_time.iterrows():
        x_train.append([(row['LONGITUDE']+130.0)/35, (row['LATTITUDE']-37.5)/12.5])
        y_train.append(row[param]/param_scale)
    x_train = torch.FloatTensor(x_train)
    y_train = torch.FloatTensor(y_train)
    # Create DataLoader
    train_dataset = TensorDataset(x_train, y_train)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    # Initialize the model
    model = Densify(input_size, hidden_size, output_size)

    # Define loss function and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Training loop
    for epoch in range(num_epochs):
        running_loss = 0.0
        for inputs, targets in train_loader:
            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, targets.unsqueeze(1))
            
            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * inputs.size(0)
        
        epoch_loss = running_loss / len(train_dataset)
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}')

    print('Training finished!')
    #torch.save(model, "models/"+date_time+"_densification_model.pt")
    return model

def TrainConvolutionModel(x_train,y_train, param = "T_CALC"):
    x_train = torch.FloatTensor(x_train)
    y_train = torch.FloatTensor(y_train)
    # Create DataLoader
    train_dataset = TensorDataset(x_train, y_train)
    train_loader = DataLoader(train_dataset, batch_size=20, shuffle=True)

    # Initialize the model
    model = CNN3D()

    # Define loss function and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=.0005)

    # Training loop
    for epoch in range(50):
        running_loss = 0.0
        for inputs, targets in train_loader:
            # Forward pass
            outputs = model(inputs.view(20,1,48,56,20))
            loss = criterion(outputs, targets.view(20,56*20))
            
            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * inputs.size(0)
        
        epoch_loss = running_loss / len(train_dataset)
        print(f'Epoch [{epoch+1}/{50}], Loss: {epoch_loss:.4f}')

    print('Training finished!')
    torch.save(model, "models/ConvolutionNet.pt")
    return model

def GetMatrixFromNet(date_time):
    true_model = torch.load("models/T_CALC/"+date_time+"_densification_model.pt")
    lon = np.linspace(-130, -60, 56)
    lat = np.linspace(25, 50, 20)
    data = []
    for i in range(len(lon)):
        for j in range(len(lat)):
            feature_tensor = []
            feature_tensor.append((lon[i]+130.0)/35.0)
            feature_tensor.append((lat[j]-37.5)/12.5)
            data.append(true_model(torch.FloatTensor(feature_tensor)).item())
    return data

def MakePrediction(start_date, dates, hours, model, this_df, display = True):
    start_index = np.where(dates==start_date)[0][0]
    input_vector = []
    model.eval()
    for i in range(48):
        input_vector.append(np.load("outputs/T_CALC/"+dates[start_index-47+i]+"_data_grid_adjusted.npy"))
    
    for i in range(hours):
        x = torch.FloatTensor(input_vector).view(1,48,56,20)
        raw_output = model(x)
        raw_output = raw_output.view(1120)
        input_vector.pop(0)
        input_vector.append(raw_output.detach().numpy())

    date_time = dates[start_index+hours]
    output = raw_output.view(56,20)
    lon = np.linspace(-130, -60, 56)
    lat = np.linspace(25, 50, 20)
    lons = []
    lats = []
    data = []
    true_data = []
    true_data_loaded = []
    date_time_array = []
    for i in range(len(lon)):
        for j in range(len(lat)):
            feature_tensor = []
            lons.append(lon[i])
            lats.append(lat[j])
            data.append(output[i][j].item()*param_scale)
            date_time_array.append(date_time)
    global num_epochs
    num_epochs = 32
    reduced_df = pd.DataFrame({"LONGITUDE": lons, "LATTITUDE": lats, "T_CALC": data, "DATETIME": date_time_array})
    file_path = "models/animations/predictions/convolution_"+date_time+".pt"
    if os.path.exists(file_path):
        expansion_net = torch.load(file_path)
    else:
        expansion_net = TrainDensificationModel(reduced_df,date_time,"T_CALC")
        torch.save(expansion_net, file_path)

    lon = np.linspace(-130, -60, lon_size)
    lat = np.linspace(25, 50, lat_size)
    lons = []
    lats = []
    data = []
    expansion_net.eval()
    for i in range(len(lon)):
        for j in range(len(lat)):
            lons.append(lon[i])
            lats.append(lat[j])
            feature_tensor = []
            feature_tensor.append((lon[i]+130.0)/35.0)
            feature_tensor.append((lat[j]-37.5)/12.5)
            data.append(expansion_net(torch.FloatTensor(feature_tensor)).item()*param_scale)
    if display:
        DisplayMap(lons, lats, data, this_df, date_time, "T_CALC", str(hours)+" Hour Feed-Forward Prediction")

    #Compute from weather stations
    filtered_df = this_df[(this_df["DATETIME"]==date_time) & (this_df["T_CALC"]>-9999)]
    num_epochs = 256
    
    file_path = "models/animations/ground_truths/"+date_time+".pt"
    if os.path.exists(file_path):
        densification_net  = torch.load(file_path)
    else:
        densification_net = TrainDensificationModel(filtered_df,date_time,"T_CALC")
        torch.save(densification_net, file_path)

    lon = np.linspace(-130, -60, lon_size)
    lat = np.linspace(25, 50, lat_size)
    lons = []
    lats = []
    data = []
    densification_net.eval()
    for i in range(len(lon)):
        for j in range(len(lat)):
            lons.append(lon[i])
            lats.append(lat[j])
            feature_tensor = []
            feature_tensor.append((lon[i]+130.0)/35.0)
            feature_tensor.append((lat[j]-37.5)/12.5)
            data.append(densification_net(torch.FloatTensor(feature_tensor)).item()*param_scale)

    if display:
        DisplayMap(lons, lats, data, this_df, date_time, "T_CALC", str(hours)+" Hour Ground Truth")

def GetModelAccuracy(start_date, dates, hours, model, this_df):
    start_index = np.where(dates==start_date)[0][0]
    input_vector = []
    model.eval()
    for i in range(48):
        input_vector.append(np.load("outputs/T_CALC/"+dates[start_index-47+i]+"_data_grid_adjusted.npy"))
    
    for i in range(hours):
        x = torch.FloatTensor(input_vector).view(1,48,56,20)
        raw_output = model(x)
        raw_output = raw_output.view(1120)
        input_vector.pop(0)
        input_vector.append(raw_output.detach().numpy())

    date_time = dates[start_index+hours]
    output = raw_output.view(56,20)
    lon = np.linspace(-130, -60, 56)
    lat = np.linspace(25, 50, 20)
    lons = []
    lats = []
    data = []
    true_data = []
    true_data_loaded = []
    date_time_array = []
    for i in range(len(lon)):
        for j in range(len(lat)):
            feature_tensor = []
            lons.append(lon[i])
            lats.append(lat[j])
            data.append(output[i][j].item()*param_scale)
            date_time_array.append(date_time)
    global num_epochs
    num_epochs = 16
    reduced_df = pd.DataFrame({"LONGITUDE": lons, "LATTITUDE": lats, "T_CALC": data, "DATETIME": date_time_array})
    file_path = "models/animations/predictions/convolution_"+date_time+".pt"
    if os.path.exists(file_path):
        expansion_net = torch.load(file_path)
    else:
        expansion_net = TrainDensificationModel(reduced_df,date_time,"T_CALC")
        torch.save(expansion_net, file_path)
        
    #Compute from weather stations
    filtered_df = this_df[(this_df["DATETIME"]==date_time) & (this_df["T_CALC"]>-9999)]
    
    correctPoints = 0
    totalPoints = 0
    for index, row in filtered_df.iterrows():
        feature_tensor = []
        feature_tensor.append((row["LONGITUDE"]+130.0)/35.0)
        feature_tensor.append((row["LATTITUDE"]-37.5)/12.5)
        if abs(expansion_net(torch.FloatTensor(feature_tensor)).item()*param_scale - row["T_CALC"]) <= tolerance:
            correctPoints += 1.0
        totalPoints += 1.0
    print((correctPoints/totalPoints))
    
    num_epochs = 256
    file_path = "models/animations/ground_truths/"+date_time+".pt"
    if os.path.exists(file_path):
        densification_net  = torch.load(file_path)
    else:
        densification_net = TrainDensificationModel(filtered_df,date_time,"T_CALC")
        torch.save(densification_net, file_path)

    correctPoints = 0
    totalPoints = 0
    for index, row in filtered_df.iterrows():
        feature_tensor = []
        feature_tensor.append((row["LONGITUDE"]+130.0)/35.0)
        feature_tensor.append((row["LATTITUDE"]-37.5)/12.5)
        if abs(densification_net(torch.FloatTensor(feature_tensor)).item()*param_scale - row["T_CALC"]) <= tolerance:
            correctPoints += 1.0
        totalPoints += 1.0
    #print("Ground truth accuracy: "+str(correctPoints/totalPoints))


def AnimateModel(start_date, dates, hours, model, this_df, truths = False):
    start_index = np.where(dates==start_date)[0][0]
    input_vector = []
    model.eval()
    print(sum(p.numel() for p in model.parameters() if p.requires_grad))
    valid_hours = []
    for i in np.linspace(1,240,60):
        valid_hours.append(int(np.floor(i)))
    for i in range(48):
        input_vector.append(np.load("outputs/T_CALC/"+dates[start_index-47+i]+"_data_grid_adjusted.npy"))

    global all_data
    global all_data_gt
    global all_lons
    global all_lats

    for k in range(hours):
        x = torch.FloatTensor(input_vector).view(1,48,56,20)
        raw_output = model(x)
        raw_output = raw_output.view(1120)
        input_vector.pop(0)
        input_vector.append(raw_output.detach().numpy())
        if k in valid_hours:
            date_time = dates[start_index+k]
            output = raw_output.view(56,20)
            if not truths:
                lon = np.linspace(-130, -60, 56)
                lat = np.linspace(25, 50, 20)
                lons = []
                lats = []
                data = []
                true_data = []
                true_data_loaded = []
                date_time_array = []
                for i in range(len(lon)):
                    for j in range(len(lat)):
                        feature_tensor = []
                        lons.append(lon[i])
                        lats.append(lat[j])
                        data.append(output[i][j].item()*param_scale)
                        date_time_array.append(date_time)
                global num_epochs
                num_epochs = 32
                reduced_df = pd.DataFrame({"LONGITUDE": lons, "LATTITUDE": lats, "T_CALC": data, "DATETIME": date_time_array})
                file_path = "models/animations/predictions/convolution_"+date_time+".pt"
                print(file_path)
                if os.path.exists(file_path):
                    expansion_net = torch.load(file_path)
                else:
                    expansion_net = TrainDensificationModel(reduced_df,date_time,"T_CALC")
                    torch.save(file_path)

                lon = np.linspace(-130, -60, lon_size)
                lat = np.linspace(25, 50, lat_size)
                lons = []
                lats = []
                data = []
                expansion_net.eval()
                for i in range(len(lon)):
                    for j in range(len(lat)):
                        lons.append(lon[i])
                        lats.append(lat[j])
                        feature_tensor = []
                        feature_tensor.append((lon[i]+130.0)/35.0)
                        feature_tensor.append((lat[j]-37.5)/12.5)
                        data.append(expansion_net(torch.FloatTensor(feature_tensor)).item()*param_scale)
                all_data.append(data)
            else:
                #Compute from weather stations
                filtered_df = this_df[(this_df["DATETIME"]==date_time) & (this_df["T_CALC"]>-9999)]
                num_epochs = 256
                
                file_path = "models/animations/ground_truths/"+date_time+".pt"
                if os.path.exists(file_path):
                    densification_net  = torch.load(file_path)
                else:
                    densification_net = TrainDensificationModel(filtered_df,date_time,"T_CALC")
                    torch.save(file_path)

                lon = np.linspace(-130, -60, lon_size)
                lat = np.linspace(25, 50, lat_size)
                lons = []
                lats = []
                data = []
                densification_net.eval()
                for i in range(len(lon)):
                    for j in range(len(lat)):
                        lons.append(lon[i])
                        lats.append(lat[j])
                        feature_tensor = []
                        feature_tensor.append((lon[i]+130.0)/35.0)
                        feature_tensor.append((lat[j]-37.5)/12.5)
                        data.append(densification_net(torch.FloatTensor(feature_tensor)).item()*param_scale)
                all_data.append(data)

    # Create a 2D grid for longitude and latitude
    
    all_lons = lons
    all_lats = lats

    global ax
    # Create a figure and axis with Cartopy projection
    fig = plt.figure(figsize=(10, 6))
    ax = fig.add_subplot(1, 1, 1, projection=ccrs.LambertConformal())

    anim = animation.FuncAnimation(fig, update, frames=hours,interval=int(np.floor(1000/6.0)))
    plt.show()
    anim.save('animation_convolution.mp4', fps=6, writer='ffmpeg')

def update(frame):
    if frame >= 59:
        return None
    lon_grid, lat_grid = np.meshgrid(np.linspace(-130, -60, lon_size), np.linspace(25, 50, lat_size))

    # Interpolate data onto the 2D grid
    data_grid = griddata((all_lons, all_lats), all_data[frame], (lon_grid, lat_grid), method='nearest')
    data_mask = np.load("geo/border_mask.npy")

    # Apply data mask
    for i in range(len(lon_grid)):
        for j in range(len(lon_grid[0])):
            lon = lon_grid[i][j]
            lat = lat_grid[i][j]
            if not data_mask[i][j]:
                data_grid[i][j] = np.nan
    global cbar
    ax.clear()
    # Plot the data as a colormap
    # Create a custom colormap with transparency
    colors = plt.cm.turbo(np.linspace(0, 1, 256))
    colors[:, 3] = 0.7  # Set alpha value (transparency) to 0.5
    transparent_cmap = LinearSegmentedColormap.from_list('transparent_cmap', colors)
    sc = ax.contourf(lon_grid, lat_grid, data_grid, cmap=transparent_cmap, levels = 20, transform=ccrs.PlateCarree())
        # Add state borders
    ax.add_feature(cfeature.STATES.with_scale('10m'), edgecolor='gray', linewidth=0.5)

    # Add ocean background outside the US map
    ax.add_feature(cfeature.OCEAN, edgecolor='none', facecolor='dimgray')
    valid_hours = []
    for i in np.linspace(1,240,60):
        valid_hours.append(int(np.floor(i)))
    ax.set_title("3D Convolution Prediction, Hour "+str(valid_hours[int(frame)]))
    # Add map features
    ax.coastlines()
    ax.gridlines()
    # Set plot extent to the United States
    ax.set_extent([-125, -65, 20, 50], ccrs.PlateCarree())
    # Add colorbar    global cbar
    if (type(cbar) == type(1)):
        cbar = plt.colorbar(sc, ax=ax, orientation='vertical', shrink=0.5)
        cbar.set_label('Temperature °C')

    sc.set_clim(vmin=-20, vmax=25)



#####################################################################################################################################################
if clean_data:
    print("Starting DB Cleaning")
    df = PrepDataSet(pd.read_csv("db/minidb.csv"))
else:
    df = pd.read_csv("db/minidb.csv")
    df = df[((df['LONGITUDE'] > -130) & (df['LONGITUDE'] < -60) & (df['LATTITUDE'] > 25) & (df['LATTITUDE'] < 50))]
    df["DATETIME"] = df["UTC_DATE"].astype(str)+df["UTC_TIME"].astype(str)

dates = df['DATETIME'].unique()

if pull_data:
    #Generates a very large set of spatial-temporal training tensors 1000x48x56x20
    x_train = []
    y_train = []
    for i in np.linspace(0,len(dates)-150,10):
        for j in range(100):
            x = []
            #print("Starting From: "+dates[int(np.floor(i))+j])
            prev_hour = int(dates[int(np.floor(i))+j][8:])-100
            for k in range(48):
                if int(dates[int(np.floor(i))+j+k][8:]) == prev_hour+100:
                    print(dates[int(np.floor(i))+j+k])
                    x.append(np.load("outputs/T_CALC/"+dates[int(np.floor(i))+j+k]+"_data_grid_adjusted.npy"))
                else:
                    print("Continuity Error")
                prev_hour = prev_hour+100
                if prev_hour == 2300:
                    prev_hour = -100
            y = np.load("outputs/T_CALC/"+dates[int(np.floor(i))+j+48]+"_data_grid_adjusted.npy")
            x_train.append(x)
            y_train.append(y)
    np.save("outputs/x_train.npy",x_train)
    np.save("outputs/y_train.npy",y_train)
else:
    x_train = np.load("outputs/x_train.npy")
    y_train = np.load("outputs/y_train.npy")

if train_net:
    model = TrainConvolutionModel(x_train,y_train)
else:
    model = torch.load("models/ConvolutionNet.pt")

#Possible visualization tool
#dot = make_dot(model(x), params=dict(model.named_parameters()))
#dot.render("network_structure", format="png")
#MakePrediction("202301030", dates, 24, model, df, display = True)
#for i in np.linspace(1,240,60):
#    GetModelAccuracy("202301030", dates, int(np.floor(i)), model, df)
#for i in np.linspace(1,240,60):
#    print(int(np.floor(i)))
AnimateModel("202301030", dates, 240, model, df)