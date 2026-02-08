# Nano-ARPES Browser

A lightweight GUI application for browsing and exporting **nano-/micro-ARPES 4D datasets**
(spatial map + spectra) acquired at synchrotron nano-ARPES beamlines (currently: **ANTARES / SOLEIL** `.nxs`).

The app is designed for fast exploration:
- choose a point (or small region) on the spatial map
- inspect the local ARPES spectrum
- optionally view the spectrum in **k∥**
- export exactly what you need (map / spectrum / selected 4D region / full dataset)

---

## What the app shows

### Data model
Internal intensity array shape:

(y, x, angle, energy)


Display conventions:
- The spatial image is computed as: `rot90(sum over angle & energy)` for convenient orientation.
- The raw beamline X index is reversed internally; the app converts between **display X** and **data X** consistently.

### Main views
- **Spatial Map (left)**  
  Integrated intensity over the current spectral window (full range by default, or ROI-selected).
- **ARPES Spectrum (right)**  
  Spectrum at the selected spatial position (optionally integrated over a small spatial box).

---

## Features

### Interactive browsing
- Click/drag the crosshair in the spatial map to select a position.
- View the corresponding ARPES spectrum (angle × energy).
- Optional spatial integration: sum spectra in a rectangle around the current position.

### ROI-based spectral filtering
- Draw a rectangular ROI in the spectrum view:
  - In **Angle mode**: ROI is (angle, energy)
  - In **k-space mode**: ROI is (k, energy)
- The spatial map updates to show intensity integrated in the selected ROI.

### Angle → k∥ conversion (basic)
- Convert the spectrum x-axis from emission angle to **k∥** using:

\[
k_\parallel\,[\mathrm{\AA^{-1}}] = 0.5123167\sqrt{E_\mathrm{kin}\,[\mathrm{eV}]}\,\sin(\theta)
\]

- The conversion uses a user-adjustable **zero-angle** offset (θ₀) to align k = 0.

> Note: current implementation assumes a simple geometry (no analyzer/sample tilt terms).
> See **TODO** for planned improvements.

---

## Export

Exports are available from the **Export** menu/button and are aimed at keeping files small when needed.

### 1) Spatial Map export
Exports the **currently displayed** spatial image (after ROI / k-space ROI integration).

- **CSV (`.csv`)**: raw 2D array
- **Igor Pro (`.itx`)**: 2D wave with axis scaling

Typical use:
- quick sharing
- external plotting
- feeding into other analysis scripts

### 2) Spectrum export
Exports the **currently displayed** spectrum at the selected position (or integrated region).

- **CSV (`.csv`)**: raw 2D array (x × energy)
- **Igor Pro (`.itx`)**: 2D wave + separate axis waves

Notes:
- In k-space mode the x-axis is `k∥ (Å⁻¹)`
- Otherwise the x-axis is emission angle (deg)

### 3) Selected Region export (Igor `.itx`) — recommended for saving space
This export is intended to **avoid huge full-dataset files**.

It writes:
- `region_4d`: the selected 4D cube (X × Y × angle × energy) for the chosen spatial region
- `region_spatial`: integrated spatial map of that region
- `region_spectrum`: integrated spectrum of that region
- `region_x`, `region_y`, `region_angle`, `region_energy`: axis waves
- `region_center`: (center_x, center_y)

How the region is chosen:
- enable **Spatial Integration**
- set X/Y pixels (half-widths around the current position)
- choose **Export → Selected Region (Igor .itx)**

This is the best workflow when:
- you want to analyze a subset in Igor Pro
- you want to archive/share only the interesting part of the measurement

### 4) Full Dataset export (Igor `.itx`)
Exports the full dataset and axes.

Creates:
- `arpes_4d` (optional, can be large)
- `spatial_map`
- `x_spatial`, `y_spatial`, `angle_axis`, `energy_axis`

Use this only when you truly need the entire cube in Igor.

---

## Supported input formats

### ANTARES / SOLEIL NeXus (`.nxs`)
Currently supported loader: `DataLoader.load_nxs()`.

If you want support for additional beamlines/formats, open an issue and attach:
- the file structure (HDF5 tree)
- axis metadata location
- any rotation / instrument geometry fields you rely on

---

## Installation (development)

```bash
git clone https://github.com/senkovskiy/nano-arpes-browser
cd nano-arpes-browser

# Install dependencies (requires uv)
make install

# Run
make run
```
Alternative (without Makefile):

```bash
uv sync --all-extras
uv run nano-arpes-browser
```
---

## Notes on k-space conversion

- The app converts angle → k∥ using kinetic energy.

- If your dataset is in binding energy, kinetic energy can be computed if photon energy is known:

\[
E_{\mathrm{kin}} = h\nu - \phi - E_B
\]

(currently the GUI focuses on kinetic-energy datasets; metadata plumbing can be extended as needed)

## TODO

- **Tilt / geometry correction for k-space conversion**
    Include sample/analyzer tilt (τ) and mapping geometry so that “k along the slit” remains correct when τ ≠ 0.

- Add photon energy entry in GUI (enables binding↔kinetic conversion workflows)

- Improve fast k-space conversion path (and/or remove unused convert_spectrum_fast)

- More importers (other beamlines / HDF5 conventions)

- Performance: optional memory-mapped IO / chunked processing for very large datasets

## License

MIT (see ``pyproject.toml``).