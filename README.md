# Instant3D

**Instant3D** is a lightweight open-source graphical user interface (GUI) frontend for **TotalSegmentator**. It helps users run TotalSegmentator-based medical image segmentation from CT or MRI data and directly generate reusable 3D reconstruction-oriented outputs, including STL meshes, NIfTI segmentation images, optional SVG masks, and optional CSV volume tables.

Instant3D is designed for researchers, clinicians, educators, and students who want a streamlined workflow for segmentation and 3D output generation without repeatedly using command-line operations during routine use. TotalSegmentator is used as the segmentation backend and must be installed separately during the initial setup.

[日本語はこちら](https://github.com/SatoruMuro/Instant3D/blob/main/READMEJP.md)

![Instant3D Top Image](https://github.com/SatoruMuro/Instant3D/blob/main/files/Instant3D_image01.jpg)

---

## Scope and Limitations

Instant3D is intended as a **workflow tool** for running TotalSegmentator and exporting 3D reconstruction-oriented outputs.

### What Instant3D does

* Provides a simple GUI for running TotalSegmentator
* Supports DICOM folders, NIfTI files, and NRRD files
* Allows ROI selection with autocomplete suggestions
* Supports multiple ROI selection
* Supports batch processing of multiple datasets
* Exports STL meshes and NIfTI segmentation images by default
* Optionally exports per-slice SVG masks and CSV volume tables
* Supports downstream refinement of SVG masks using SegRef3D

### What Instant3D does not do

* Instant3D is not a standalone segmentation model
* Segmentation accuracy depends on the installed TotalSegmentator backend
* Instant3D does not include an integrated 3D viewer
* Detailed visualization and manual editing should be performed using external tools such as 3D Slicer or SegRef3D
* The current packaged executable is primarily intended for Windows users
* Initial TotalSegmentator setup still requires some command-line operations


---


## Download and Placement

The latest version of Instant3D is distributed as a zip file from GitHub Releases:
[Instant3D Releases Page](https://github.com/SatoruMuro/Instant3D/releases/tag/Instant3Dv20260617)

1. Download the zip file from the link above.
2. Unzip it and place the entire folder directly under the `C:\\` drive (e.g., `C:\\Instant3D`).
   → This placement simplifies the path and helps avoid errors.

---

## Installation (first-time only)

### Option A: CPU version (easy setup, recommended for most users)

1. **Start menu → Right-click “Windows Terminal” or “Command Prompt” → Run as administrator**
   (Either is fine. On Windows 11, “PowerShell” opens by default.)

2. Copy and paste the following script:

```bat
:: === Instant3D Setup (CPU mode) ===
:: 1) Install Python via winget (skips if already installed)
winget install -e --id Python.Python.3.12 -h || echo (Python 3.12 already installed or set manually)

:: 2) Upgrade pip & install TotalSegmentator (CPU version)
python -m pip install --upgrade pip
pip install TotalSegmentator

:: 3) Download the model (only once, several GB)
totalsegmentator --download_model

:: 4) Test run
totalsegmentator -h
echo.
echo === CPU setup complete! Close this window. ===
pause
```

⚠️ **If winget is not available**, install **Python 3.12 (64-bit)** from the [official Python website](https://www.python.org/downloads/windows/).
During installation, check **“Add Python to PATH”**. Then run only steps **2)–4)** above.

---

### Option B: GPU version (faster processing, requires NVIDIA CUDA GPU)

1. **Start menu → Right-click “Windows Terminal” or “Command Prompt” → Run as administrator**

2. Copy and paste the following script:

```bat
:: === Instant3D Setup (GPU mode) ===
:: 1) Install Python via winget (skips if already installed)
winget install -e --id Python.Python.3.12 -h || echo (Python 3.12 already installed or set manually)

:: 2) Upgrade pip
python -m pip install --upgrade pip

:: 3) Install PyTorch with CUDA (example: CUDA 12.1 build)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

:: 4) Install TotalSegmentator
pip install TotalSegmentator

:: 5) Download the model (only once, several GB)
totalsegmentator --download_model

:: 6) Test run
totalsegmentator -h
echo.
echo === GPU setup complete! Close this window. ===
pause
```

⚠️ **Notes for GPU users**

* Make sure your GPU drivers and CUDA toolkit are up to date.
* Adjust the PyTorch install command to match your CUDA version (see the [PyTorch installation guide](https://pytorch.org/get-started/locally/)).
* If no compatible GPU is found, TotalSegmentator will fall back to CPU automatically.

---

## Basic Usage

1. **Preparation**

   * Place your DICOM folder or NIfTI file in the same folder as `Instant3D.exe`.
   * If your DICOM files do not have the `.dcm` extension, add it first using the following batch file:  
     👉 [add_dcm_extension.bat](https://github.com/SatoruMuro/Instant3D/blob/main/files/add_dcm_extension.bat)


     **How to use:**

     1. Download the batch file from the above link.
     2. Place it in the same folder as your DICOM files.
     3. Double-click the batch file to run — `.dcm` extensions will be automatically added to all files in that folder.

2. **Launch Instant3D**
   Double-click `Instant3D.exe` to start the program.

3. **Input Selection**

   * Click **Browse** in the **Input** section.
   * If your dataset is a single file (e.g., `.nii`), select the file directly.
   * If your dataset consists of multiple DICOM files, cancel the file selection dialog once; a folder selection dialog will appear — select your DICOM folder there.

4. **ROI Settings**

   * Enter the desired ROI and click **Add ROI** (multiple ROIs can be added).
   * As you start typing, **Instant3D automatically suggests ROI names** based on the available segmentation options — simply select the correct one from the suggestions.
   * For structures not included in the **basic segmentation task**, switch to the appropriate **specialized task** before setting the ROI.

     * Example: To reconstruct the **mandible**, select `craniofacial_structures` under **segmentation task**, then specify ROIs such as `mandible` and `skull`.

5. **Run Segmentation**
   Click **Run** to start processing.

6. **Output**

   The output will be automatically saved in a folder named `<input_name>_Instant3D` in the same directory.

   Default outputs include:

   * **STL files**: 3D surface meshes for visualization, 3D printing, surgical simulation, or downstream modeling
   * **NIfTI segmentation images**: voxel-based segmentation images generated by TotalSegmentator

   Optional outputs include:

   * **SVG masks**: per-slice editable masks that can be opened and refined in SegRef3D
   * **CSV volume tables**: quantitative volume measurements for the selected ROIs

   When multiple ROIs are selected, each ROI is exported as a separate output item with consistent file naming. CSV files summarize volume measurements by ROI.



<img src="https://github.com/SatoruMuro/Instant3D/blob/main/files/Instant3D_image02.png" width="50%">

---

## ROI Catalogs (per task)

Lists of **available ROI names** for each task are provided as `.txt` files here:
**[https://github.com/SatoruMuro/Instant3D/tree/main/files/resources](https://github.com/SatoruMuro/Instant3D/tree/main/files/resources)**

* Example files: `roi_catalog_body.txt`, `roi_catalog_body_mr.txt`, `roi_catalog_abdominal_muscles.txt`, `roi_catalog_brain_structures.txt` etc.
* Each file corresponds to a **TotalSegmentator task** (e.g., `body`, `body_mr`, `abdominal_muscles`).
* In the **ROI input field**, enter the names exactly as listed in the txt files (e.g., `liver`, `kidney_right`, `femur_left`). Autocomplete suggestions will appear as you type.
* Some special subtasks (e.g., `appendicular_bones`, `tissue_types`, `heartchambers_highres`, `face`) require an **Academic License**. See the section *TotalSegmentator: Academic License for Special Subtasks* below.

For more detailed information on available structures and tasks, please also refer to the **official TotalSegmentator repository**:
[https://github.com/wasserth/TotalSegmentator](https://github.com/wasserth/TotalSegmentator)

---

## What if my target structure is not in the ROI catalog?

Instant3D relies on the ROI definitions provided by **TotalSegmentator**.
If the structure you want to reconstruct is **not included in the available ROI lists**, you can use **[SegRef3D](https://github.com/SatoruMuro/SAM2GUIfor3Drecon)**.

SegRef3D is a **general-purpose 3D reconstruction tool** that provides:

* A **highly versatile automatic segmentation function** that works regardless of imaging modality or target structure
* Extraction and 3D reconstruction of arbitrary structures not covered by ROI catalogs
* STL and quantitative data export similar to Instant3D

In short:

* **Instant3D** → Fast & automatic for predefined ROIs
* **SegRef3D** → Flexible & customizable for any structure, beyond the predefined ROI catalog

---

## FAQ

**Q1. It says `totalsegmentator` not found**
A. Run the following command in the terminal to confirm it is in PATH:

```sh
totalsegmentator -h
```

If not found, add the following folder to PATH (example):

```
C:\Users\<your-username>\AppData\Local\Programs\Python\Python312\Scripts\
```

Then restart the terminal and try again.

---

**Q2. Model download is very slow / stuck**
A. Depending on network conditions, downloading can take 30 minutes or more. Several GB of disk space is required.

---

**Q3. Is a GPU required?**
A. Not required (CPU mode is available). With a GPU, processing can be faster.

---

**Q4. Where are the outputs saved?**
A. In the parent folder of the input, a new folder `<input_name>_Instant3D` will be created.

---

## Maintenance and Future Development

Instant3D is maintained through this GitHub repository. Updates, bug reports, and feature requests will be managed using GitHub Releases and GitHub Issues.

Planned or potential future developments include:

* Simplified installation
* More complete backend bundling
* Docker-based distribution
* Broader cross-platform packaging beyond Windows
* Improved progress reporting and error handling
* More detailed resource monitoring
* Expanded documentation and example workflows

Users are encouraged to report issues or request features through the GitHub Issues page.


---

## Update / Uninstall

### Update TotalSegmentator

```sh
pip install -U TotalSegmentator
```

### Re-download / update model

```sh
totalsegmentator --download_model
```

### Uninstall

```sh
pip uninstall TotalSegmentator
```

(To uninstall Python itself, use Windows “Apps & Features”.)

---

## Troubleshooting Checklist

* Can you run `Instant3D.exe`?
* Does `totalsegmentator -h` show help? (If not, it’s a PATH issue.)
* Is the model fully downloaded? (First time only, large size.)
* Do you have write permissions to the output folder (`<input_name>_Instant3D`)? (Be careful with network drives.)

---

## TotalSegmentator: Academic License for Special Subtasks

Some TotalSegmentator subtasks (e.g., **appendicular\_bones**, **tissue\_types**, **heartchambers\_highres**, **face**) are provided under a **restricted Academic / Non-commercial license**. To use these, you must obtain and register a license key.

### How to obtain a license key

1. Apply here for an Academic License:
   [Academic License for TotalSegmentator Special Models](https://backend.totalsegmentator.com/license-academic/)

2. Fill in affiliation and purpose of use.

3. Upon approval, you will receive a **license key** (example: `aca_XXXXXXXXXX`).

### How to register the license key

In the terminal or command prompt, run:

```sh
totalseg_set_license -l aca_XXXXXXXXXX
```

Once registered, the special subtasks (e.g., `--task appendicular_bones`) become available.

> Note: The Academic License is **for non-commercial use only**. For commercial usage, a separate commercial license is required.

---

## Reference

Instructions on how to open 3D data (e.g., STL files) are provided here:
[How to Open 3D Data with 3D Slicer (English)](https://github.com/SatoruMuro/Instant3D/blob/main/files/HowToOpen3D%283Dslicer%29EN.pdf)

---

## Citation

If you use **Instant3D** in your research or publications, please cite the following paper:

**Instant3D**
Satoru Muro, Takuya Ibara, Akimoto Nimura, et al.
*Instant3D: A User-Friendly GUI Integrating TotalSegmentator for Immediate Medical Image Segmentation and 3D Reconstruction.*
PREPRINT (Version 1), Research Square, 19 November 2025.
[https://doi.org/10.21203/rs.3.rs-8150723/v1](https://doi.org/10.21203/rs.3.rs-8150723/v1)

**BibTeX:**

```bibtex
@article{muro2025instant3d,
  title={Instant3D: A User-Friendly GUI Integrating TotalSegmentator for Immediate Medical Image Segmentation and 3D Reconstruction},
  author={Muro, Satoru and Ibara, Takuya and Nimura, Akimoto and others},
  journal={Research Square},
  year={2025},
  month={November},
  note={PREPRINT (Version 1)},
  doi={10.21203/rs.3.rs-8150723/v1}
}
```

---

If you use **TotalSegmentator** in research or publications, please refer to the
**official TotalSegmentator repository** for the recommended citation format:
[https://github.com/wasserth/TotalSegmentator](https://github.com/wasserth/TotalSegmentator)

