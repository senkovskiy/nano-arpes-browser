# Architecture Diagrams

This document shows how Nano-ARPES Browser is organized at a practical level:
what loads data, what displays it, what owns the scientific model, and what writes
exports.

## Big Picture

```mermaid
flowchart LR
    user[User] --> gui[GUI<br/>PyQt6 + pyqtgraph]
    gui --> core[Core Data Model<br/>ARPESDataset]
    core --> processing[Processing<br/>k-space conversion<br/>ROI integration]
    gui --> export[Export Workflows]

    files_in[ANTARES / SOLEIL<br/>.nxs .h5 .hdf5] --> loader[DataLoader]
    loader --> core

    export --> writer[DataExporter]
    core --> writer
    writer --> files_out[CSV / Igor Pro .itx]

    settings[Qt QSettings<br/>window state + last folder] <--> gui
```

The key idea: the GUI controls interaction, but the scientific data conventions
live in the core model.

## Runtime Data Flow

```mermaid
sequenceDiagram
    actor User
    participant GUI as MainWindow / Widgets
    participant Loader as DataLoader
    participant Dataset as ARPESDataset
    participant KSpace as KSpaceConverter
    participant Exporter as DataExporter
    participant Disk as Filesystem

    User->>GUI: Open data file
    GUI->>Loader: load(path)
    Loader->>Disk: read NeXus/HDF5
    Loader-->>GUI: ARPESDataset

    GUI->>Dataset: integrated_image
    Dataset-->>GUI: spatial map

    User->>GUI: Move spatial crosshair
    GUI->>Dataset: get_spectrum_at(position, integration)
    Dataset-->>GUI: angle-energy spectrum

    opt k-space enabled
        GUI->>KSpace: convert_spectrum(...)
        KSpace-->>GUI: k-energy spectrum
    end

    User->>GUI: Move spectrum ROI
    GUI->>Dataset: get_spatial_image(...)
    Dataset-->>GUI: ROI-integrated map

    User->>GUI: Export
    GUI->>Exporter: save current data
    Exporter->>Disk: write .csv or .itx
```

## Package Boundaries

```mermaid
flowchart TB
    subgraph gui["src/nano_arpes_browser/gui"]
        app[app.py<br/>application startup]
        main[main_window.py<br/>main coordinator]
        exports[export_controller.py<br/>export workflows]
        widgets[widgets/<br/>viewer and control widgets]
    end

    subgraph core["src/nano_arpes_browser/core"]
        models[models.py<br/>ARPESDataset, axes, ROI, parameters]
        loaders[io/loaders.py<br/>ANTARES/SOLEIL HDF5 loading]
        exporters[io/exporters.py<br/>CSV and Igor writers]
        export_models[io/export_models.py<br/>export payload models]
        kspace[processing/kspace.py<br/>angle ↔ k conversion]
    end

    app --> main
    main --> widgets
    main --> exports
    main --> models
    exports --> exporters
    exports --> export_models
    loaders --> models
    models --> kspace
```

## Main Responsibilities

- `gui/app.py`: starts the Qt application.
- `gui/main_window.py`: connects widgets, dataset state, and user actions.
- `gui/export_controller.py`: handles export dialogs and prepares export payloads.
- `core/models.py`: owns array shape conventions, coordinate mapping, ROI slicing,
  spatial images, and spectrum extraction.
- `core/io/loaders.py`: loads supported ANTARES / SOLEIL NeXus-HDF5 files.
- `core/io/exporters.py`: writes CSV and Igor Pro `.itx` files.
- `core/processing/kspace.py`: performs the current angle-to-k conversion.

## Important Boundary

`core/` should remain independent of PyQt6 and pyqtgraph. This keeps data
loading, slicing, conversion, and export logic testable without starting the GUI.
