import matplotlib.pyplot as plt
import numpy as np
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import pandas as pd
import geopandas as gpd
from scipy.interpolate import griddata
from shapely.geometry import Point

checkBorder = False

df = pd.read_csv("db/minidb.csv")
filtered_df = df[(df['T_CALC'] != -9999) & (df['LONGITUDE'] > -130) & (df['LONGITUDE'] < -60) & (df['LATTITUDE'] > 25) & (df['LATTITUDE'] < 50)]
# Sample longitude and latitude data
lon = filtered_df['LONGITUDE'] # Sample longitudes for 100 points
lat = filtered_df['LATTITUDE']      # Sample latitudes for 100 points
data = filtered_df['T_CALC']                 # Sample data to plot as colormap
print(np.min(data))

# Load the shapefile of the US border
us_border = gpd.read_file("geo/cb_2022_us_nation_20m.shx")

# Create a 2D grid for longitude and latitude
lon_grid, lat_grid = np.meshgrid(np.linspace(-130, -60, 280), np.linspace(25, 50, 100))

# Interpolate data onto the 2D grid
data_grid = griddata((lon, lat), data, (lon_grid, lat_grid), method='linear')
data_mask = np.copy(data_grid)
# Iterate over the grid and set values to NaN if they fall outside the US border
if checkBorder:
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
ax.add_feature(cfeature.STATES.with_scale('10m'), edgecolor='black', linewidth=0.5)

# Add ocean background outside the US map
ax.add_feature(cfeature.OCEAN, edgecolor='none', facecolor='dimgray')

# Plot the data as a colormap
sc = ax.contourf(lon_grid, lat_grid, data_grid, cmap='coolwarm', levels = 30, transform=ccrs.PlateCarree())

# Add map features
ax.coastlines()
ax.gridlines()

# Add colorbar
cbar = plt.colorbar(sc, ax=ax, orientation='vertical', shrink=0.5)
cbar.set_label('Temperature °C')

# Set plot extent to the United States
ax.set_extent([-125, -65, 20, 50], ccrs.PlateCarree())

# Show the plot
plt.title('United States Temperature')
plt.show()
