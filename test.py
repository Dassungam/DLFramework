import rasterio
import numpy as np
import matplotlib.pyplot as plt

# 1. Open the raw TIF file
with rasterio.open('your_raw_image.tif') as src:
    # Read the first band (assuming it's a single-band image)
    img = src.read(1)
    
    # Optional: Read the mask if it's a separate file or band
    # mask = src.read(2) 

# 2. Handle NoData values (convert them to NaN so they don't mess up our math)
# If your image has a specific nodata value (like 0 or -9999), replace it
# img = np.where(img == 0, np.nan, img)

# 3. Calculate the 2nd and 98th percentiles (The QGIS default)
# We use nanpercentile to ignore any NaN values we created above
vmin, vmax = np.nanpercentile(img, (2, 98))

# 4. Clip the data to these percentiles
img_stretched = np.clip(img, vmin, vmax)

# 5. Normalize the data to a 0.0 - 1.0 range for plotting
img_normalized = (img_stretched - vmin) / (vmax - vmin)

# 6. Plot the image
plt.figure(figsize=(10, 5))
plt.imshow(img_normalized, cmap='gray')

# If you want to overlay your mask, you can do it here using an alpha channel
# plt.imshow(mask_array, cmap='viridis', alpha=0.5)

plt.title("Python Contrast Stretch (Similar to QGIS)")
plt.axis('off')
plt.show()

