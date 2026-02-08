# Nano-ARPES Browser

Professional nano-ARPES data browser and analysis tool.

## Features

- Interactive spatial map visualization
- ARPES spectrum viewer with ROI selection
- Angle to k-space conversion
- Support for NXS/HDF5 file formats

## Installation (Development)

```bash
# Clone the repository
git clone https://github.com/yourusername/nano-arpes-browser
cd nano-arpes-browser

# Install all dependencies (requires uv)
make install

# Run the application
make run


## Export

### Full dataset (Igor .itx)

Exports:
- `arpes_4d` (optional, can be skipped if too large)
- `spatial_map`
- `x_spatial`, `y_spatial`, `angle_axis`, `energy_axis`

In code:

```python
from src.core.io import DataExporter
info = DataExporter.save_full_dataset_itx(dataset, "full_dataset.itx")
print(info)
