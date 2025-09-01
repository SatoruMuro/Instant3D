# Instant3D

**Instant3D** is a tool that automatically segments organs and tissues from CT or MRI DICOM images and NIfTI files, and easily outputs 3D STL models and analysis data. It utilizes **TotalSegmentator** internally, allowing physicians and researchers to perform 3D reconstruction without additional programming.

[日本語はこちら](https://github.com/SatoruMuro/Instant3D/blob/main/READMEJP.md)

![Instant3D Top Image](https://github.com/SatoruMuro/Instant3D/blob/main/files/Instant3D_image01.jpg)

---

## Download and Placement

The latest version of Instant3D is distributed as a zip file from GitHub Releases:
[Instant3D Releases Page](https://github.com/SatoruMuro/Instant3D/releases/tag/Instant3Dv20250829)

1. Download the zip file from the link above.
2. Unzip it and place the entire folder directly under the `C:\\` drive (e.g., `C:\\Instant3D`).
   → This placement simplifies the path and helps avoid errors.

---

## Installation (first-time only)

### Easy one-step setup (recommended, Windows 10/11)

1. **Start menu → Right-click “Windows Terminal” or “Command Prompt” → Run as administrator**
   (Either is fine. On Windows 11, “PowerShell” opens by default.)

2. Copy and paste the following script (installs Python and TotalSegmentator, and downloads the model):

```bat
:: === Instant3D Setup (Run in Administrator PowerShell/CMD) ===
:: 1) Install Python via winget (skips automatically if already installed)
winget install -e --id Python.Python.3.12 -h || echo (Python 3.12: already installed or manually installed)

:: 2) Upgrade pip & install TotalSegmentator (CPU version)
python -m pip install --upgrade pip
pip install TotalSegmentator

:: 3) Download the model (only once, several GB)
totalsegmentator --download_model

:: 4) Test run (if help is shown, setup is OK)
totalsegmentator -h
echo.
echo === Setup complete! Close this window. ===
pause
```

⚠️ **If winget is not available**, install **Python 3.12 (64-bit)** from the [official Python website](https://www.python.org/downloads/windows/).
During installation, check **“Add Python to PATH”**. Then run only steps **2)–4)** above.

---

## Basic Usage

1. Double-click `Instant3D.exe` to launch.
2. **Input**: Select a DICOM folder or NIfTI file.
3. Enter ROI and click **Add ROI** (multiple can be added).
4. Click **Run**.
5. Output will be automatically saved in a folder named `<input_name>_Instant3D`.

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

SegRef3D is a **general-purpose 3D reconstruction tool** that enables you to:

* Import and refine **custom segmentation masks** (manual or AI-generated).
* Handle structures not covered by predefined ROI catalogs.
* Export STL and quantitative data, similar to Instant3D.

In short:

* **Instant3D** → Fast & automatic for predefined ROIs.
* **SegRef3D** → Flexible & customizable for any structure.

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

If you use TotalSegmentator in research or publications, please refer to the **official TotalSegmentator repository** for the recommended citation format:
[https://github.com/wasserth/TotalSegmentator](https://github.com/wasserth/TotalSegmentator)

---
