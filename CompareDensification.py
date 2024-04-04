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
tempScale = 50.0
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

def DisplayMapsAndComputeError(this_df, date_time, param = "T_CALC", check_border = False):
    filtered_df = this_df[(this_df["DATETIME"]==date_time) & (this_df[param]!=-9999)]
    training_df = filtered_df.sample(5)
    densification_net = TrainDensificationModel(training_df,date_time)
    lon = np.linspace(-130, -60, lon_size)
    lat = np.linspace(25, 50, lat_size)
    lons = []
    lats = []
    data_nn = []
    densification_net.eval()
    for i in range(len(lon)):
        for j in range(len(lat)):
            lons.append(lon[i])
            lats.append(lat[j])
            feature_tensor = []
            feature_tensor.append((lon[i]+130.0)/35.0)
            feature_tensor.append((lat[j]-37.5)/12.5)
            data_nn.append(densification_net(torch.FloatTensor(feature_tensor)).item()*tempScale)

    # Load the shapefile of the US border
    us_border = gpd.read_file("geo/cb_2022_us_nation_20m.shx")

    # Create a 2D grid for longitude and latitude
    lon_grid, lat_grid = np.meshgrid(np.linspace(-130, -60, lon_size), np.linspace(25, 50, lat_size))

    # Interpolate data onto the 2D grid
    data_grid_nearest = griddata((training_df["LONGITUDE"], training_df["LATTITUDE"]), training_df[param], (lon_grid, lat_grid), method='nearest')
    data_grid_linear = griddata((training_df["LONGITUDE"], training_df["LATTITUDE"]), training_df[param], (lon_grid, lat_grid), method='linear')
    data_grid_nn = griddata((lons, lats), data_nn, (lon_grid, lat_grid), method='nearest')

    data_grids = [data_grid_nearest, data_grid_linear, data_grid_nn]
    titles = ["Nearest Interpolation", "Linear Interpolation", "Feed-Forward Densification"]

    for k in range(3):
        data_grid = data_grids[k]
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
        fig = plt.figure(figsize=(15, 9))
        ax = fig.add_subplot(1, 1, 1, projection=ccrs.LambertConformal())

        # Add state borders
        ax.add_feature(cfeature.STATES.with_scale('10m'), edgecolor='gray', linewidth=0.5)

        # Add ocean background outside the US map
        ax.add_feature(cfeature.OCEAN, edgecolor='none', facecolor='dimgray')

        # Plot the data as a colormap
        # Create a custom colormap with transparency
        colors = plt.cm.turbo(np.linspace(0, 1, 256))
        colors[:, 3] = 0.7  # Set alpha value (transparency) to 0.7
        transparent_cmap = LinearSegmentedColormap.from_list('transparent_cmap', colors)
        sc = ax.contourf(lon_grid, lat_grid, data_grid, cmap=transparent_cmap, levels = 20, transform=ccrs.PlateCarree())

        # Plot scatter plot points of weather stations
        filtered_df = this_df[(this_df["DATETIME"]==date_time) & (this_df[param]!=-9999)]
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
        training_stations = training_df["WBANNO"]
        testing_df = filtered_df[~filtered_df["WBANNO"].isin(training_stations )]
        for index, row in testing_df.iterrows():
            tempGuess = interp_func([row["LONGITUDE"],row["LATTITUDE"]])
            stations += 1
            if str(tempGuess[0]) != 'nan':
                error += np.abs(row[param] - tempGuess[0])
            else:
                #Unable to classify
                error += np.abs(row[param])

        error = error/stations
        error = np.round(error, 2)
        # Show the plot
        plt.title(titles[k]+" Average Temperature Error: "+ str(error))
        plt.show()

    titles = ["Complete Sparse Set", "Training Subset (Dropout)","Validation of Generalization Subset"]
    #Visualize testing data
    for k in range(3):
        # Create a figure and axis with Cartopy projection
        fig = plt.figure(figsize=(15, 9))
        ax = fig.add_subplot(1, 1, 1, projection=ccrs.LambertConformal())

        # Add state borders
        ax.add_feature(cfeature.STATES.with_scale('10m'), edgecolor='gray', linewidth=0.5)

        # Add ocean background outside the US map
        ax.add_feature(cfeature.OCEAN, edgecolor='none', facecolor='dimgray')
        # Plot scatter plot points of weather stations
        if k == 0:
            filtered_df = this_df[(this_df["DATETIME"]==date_time) & (this_df[param]!=-9999)]
            scatter = ax.scatter(filtered_df["LONGITUDE"], filtered_df["LATTITUDE"], c=filtered_df[param], cmap='turbo', transform=ccrs.PlateCarree())
        if k == 1:
            scatter = ax.scatter(training_df["LONGITUDE"], training_df["LATTITUDE"], c=training_df[param], cmap='turbo', transform=ccrs.PlateCarree())
        if k == 2:
            scatter = ax.scatter(testing_df["LONGITUDE"], testing_df["LATTITUDE"], c=testing_df[param], cmap='turbo', transform=ccrs.PlateCarree())
        # Add map features
        ax.coastlines()
        ax.gridlines()

        # Add colorbar
        cbar = plt.colorbar(sc, ax=ax, orientation='vertical', shrink=0.5)
        cbar.set_label('Temperature °C')

        # Set plot extent to the United States
        ax.set_extent([-125, -65, 20, 50], ccrs.PlateCarree())

        # Show the plot
        plt.title(titles[k])
        plt.show()


def TrainDensificationModel(this_df,date_time, param = "T_CALC"):
    x_train = []
    y_train = []
    df_one_date_time = this_df[(this_df["DATETIME"] == date_time) & (this_df[param]!=-9999)]
    for index, row in df_one_date_time.iterrows():
        x_train.append([(row['LONGITUDE']+130.0)/35, (row['LATTITUDE']-37.5)/12.5])
        y_train.append(row['T_CALC']/tempScale)
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
    torch.save(model, "models/"+date_time+"_densification_model.pt")
    return model

def PrepDataSet(this_df):
    this_df = this_df[((this_df['T_CALC'] != -9999) & (this_df['LONGITUDE'] > -130) & (this_df['LONGITUDE'] < -60) & (this_df['LATTITUDE'] > 25) & (this_df['LATTITUDE'] < 50))]
    this_df["DATETIME"] = this_df["UTC_DATE"].astype(str)+this_df["UTC_TIME"].astype(str)
    entries_to_clean = len(this_df["DATETIME"].unique())
    i = 0;
    for date_time in this_df["DATETIME"].unique():
        if(i%100 == 0):
            print("Cleaning Data "+str(float(i)/float(entries_to_clean)*100.0)+"%")
        df_one_date_time = this_df[this_df["DATETIME"] == date_time]
        if(len(df_one_date_time['LONGITUDE']) != num_weather_centers):
            this_df = this_df[this_df['DATETIME'] != date_time]
        i += 1
    this_df.to_csv("db/minidb_clean.csv")
    return this_df

#####################################################################################################################################################
if clean_data:
    print("Starting DB Cleaning")
    df = PrepDataSet(pd.read_csv("db/minidb.csv"))
else:
    df = pd.read_csv("db/minidb.csv")
    df = df[((df['LONGITUDE'] > -130) & (df['LONGITUDE'] < -60) & (df['LATTITUDE'] > 25) & (df['LATTITUDE'] < 50))]
    df["DATETIME"] = df["UTC_DATE"].astype(str)+df["UTC_TIME"].astype(str)

if train_densification:
    print("Starting Training")
    
else:
    densification_model = torch.load("model/DensificationModel.pt")

DisplayMapsAndComputeError(df,"20230303"+"100")