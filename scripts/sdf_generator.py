import numpy as np
import matplotlib.pyplot as plt

# Image size
img_size = 28
half = img_size / 2

# Create coordinate grid normalized to [-1,1]
y, x = np.meshgrid(np.linspace(0, img_size - 1, img_size), np.linspace(0, img_size - 1, img_size))
x = (x - half) / half
y = (y - half) / half

# SDF functions
def sdf_circle(x, y, radius=0.5):
    return np.sqrt(x**2 + y**2) - radius

def sdf_triangle(px, py):
    k = np.sqrt(2)
    px = np.abs(px)
    px -= 0.5
    py += 0.5
    return np.maximum(k * px + py * 0.5, -py * 0.5)

def sdf_square(x, y, size=0.5):
    d = np.maximum(np.abs(x) - size, np.abs(y) - size)
    return np.maximum(d, 0) + np.minimum(d, 0)

def sdf_hexagon(x, y, radius=0.5):
    x = np.abs(x)
    y = np.abs(y)
    return np.maximum(x * 0.866025 + y * 0.5, y) - radius

def sdf_to_image(sdf, sharpness=10.0):
    # Return raw SDF values for auto-decoder training
    img = sdf
    return img





def normalize_sdf_dataset(sdf_data, method='sigmoid'):
    """Normalize SDF values to [0,1] range while preserving structure"""
    
    if method == 'sigmoid':
        # Sigmoid normalization - good for preserving SDF boundaries
        normalized = 1.0 / (1.0 + np.exp(sdf_data * 5.0))
    
    elif method == 'tanh':
        # Tanh normalization - symmetric around 0.5
        normalized = (np.tanh(-sdf_data * 2.0) + 1.0) / 2.0
    
    elif method == 'minmax':
        # Min-max normalization - can be problematic with outliers
        normalized = (sdf_data - sdf_data.min()) / (sdf_data.max() - sdf_data.min())
      elif method == 'clamp':
        # Clamp and normalize - loses some SDF information
        clamped = np.clip(sdf_data, -2.0, 2.0)  # Reasonable SDF range
        normalized = (clamped + 2.0) / 4.0  # Map [-2,2] to [0,1]
    
    return normalized.astype(np.float32)


def generate_sdf_dataset_sample(n_samples_per_class=1):
    """Generate minimal SDF dataset with just one sample per shape class"""
    img_h, img_w = img_size, img_size
    half_h, half_w = img_h / 2, img_w / 2
    
    # Create coordinate grid
    y_grid, x_grid = np.meshgrid(np.linspace(0, img_h - 1, img_h), 
                                 np.linspace(0, img_w - 1, img_w))
    x_norm = (x_grid - half_w) / half_w
    y_norm = (y_grid - half_h) / half_h
    
    x_data = []
    y_data = []
    
    # Generate classes with fixed parameters for consistency
    shape_types = ['circle', 'triangle', 'square', 'hexagon']
    
    for shape_idx, shape_type in enumerate(shape_types):
        for i in range(n_samples_per_class):
            # Fixed parameters for consistent shapes
            radius = 0.5
            size = 0.5
            offset_x = 0
            offset_y = 0
            sharpness = 15
            
            # Generate SDF based on shape type
            if shape_type == 'circle':
                sdf = sdf_circle(x_norm + offset_x, y_norm + offset_y, radius)
            elif shape_type == 'triangle':
                sdf = sdf_triangle(x_norm + offset_x, y_norm + offset_y)
            elif shape_type == 'square':
                sdf = sdf_square(x_norm + offset_x, y_norm + offset_y, size)
            elif shape_type == 'hexagon':
                sdf = sdf_hexagon(x_norm + offset_x, y_norm + offset_y, radius)
                
            img = sdf_to_image(sdf, sharpness)
            x_data.append(img)
            y_data.append(shape_idx)
    
    # Convert to numpy arrays
    x_data = np.array(x_data, dtype=np.float32)
    y_data = np.array(y_data)
    
    return x_data, y_data
