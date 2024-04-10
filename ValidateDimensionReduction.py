import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import sys
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import numpy as np
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import pandas as pd
import geopandas as gpd
from scipy.interpolate import griddata
from scipy.interpolate import RegularGridInterpolator
from shapely.geometry import Point

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

#Per run params
train_densification = True
clean_data = False

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

def DisplayDenseAndReducedMaps(this_df, date_time, param = "T_CALC"):
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
    DisplayMap(lons, lats, data, this_df, date_time, param, "Initial Dense Map (280x100)")
    lon = np.linspace(-130, -60, 56)
    lat = np.linspace(25, 50, 20)
    lons = []
    lats = []
    data = []
    date_time_array = []
    densification_net.eval()
    for i in range(len(lon)):
        for j in range(len(lat)):
            lons.append(lon[i])
            lats.append(lat[j])
            date_time_array.append(date_time)
            feature_tensor = []
            feature_tensor.append((lon[i]+130.0)/35.0)
            feature_tensor.append((lat[j]-37.5)/12.5)
            data.append(densification_net(torch.FloatTensor(feature_tensor)).item()*param_scale)
    DisplayMap(lons, lats, data, this_df, date_time, param, "Reduced Dimension Map (56x20)")

    #Train a new network on the reduced data to evaluate expansion
    reduced_df = pd.DataFrame({"LONGITUDE": lons, "LATTITUDE": lats, param: data, "DATETIME": date_time_array})
    expansion_net = TrainDensificationModel(reduced_df,date_time,param)
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
    DisplayMap(lons, lats, data, this_df, date_time, param, "Expanded Dense Map (280x100)")
    


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
    print("Starting Training")
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
        if epoch_loss < .0002:
            print("Very low loss value, stopping training")
            break

    print('Training finished!')
    torch.save(model, "models/"+date_time+"_densification_model.pt")
    return model


#####################################################################################################################################################
if clean_data:
    print("Starting DB Cleaning")
    df = PrepDataSet(pd.read_csv("db/minidb.csv"))
else:
    df = pd.read_csv("db/minidb.csv")
    df = df[((df['LONGITUDE'] > -130) & (df['LONGITUDE'] < -60) & (df['LATTITUDE'] > 25) & (df['LATTITUDE'] < 50))]
    df["DATETIME"] = df["UTC_DATE"].astype(str)+df["UTC_TIME"].astype(str)

DisplayDenseAndReducedMaps(df, "20230301"+"1200","T_CALC")