# AgroVision AI — Notebook Training + Streamlit UI

## Dataset

This project uses the PlantVillage dataset for plant leaf disease classification across 38 classes.

The complete dataset is not stored in this GitHub repository because of its large size.

**Download the dataset:** https://drive.google.com/file/d/1nbYtTyaT-lv0scpnnF6xQrt1QQLeCEnk/view?usp=drive_link

### Setup

1. Download and extract the dataset.
2. Place it in the `dataset/PlantVillage/` directory.
3. Run the development notebook to train the model, or use the provided trained model if available.
4. Launch the Streamlit application.

The dataset directory should contain one subfolder per class, with the corresponding leaf images inside each folder.
## Files
- `AgroVision_Development.ipynb`: dataset inspection, preprocessing, EfficientNetB0 transfer learning, training, evaluation, testing, and export.
- `app.py`: Streamlit UI copied from the uploaded AgroVision interface.
- `predict.py`, `recommendation.py`, `class_names.py`, `disease_database.json`: runtime helpers required by the UI.
- `requirements.txt`: starting dependency list; install into a dedicated environment.

## Expected dataset
Set the notebook's `DATASET_DIR` to the directory whose immediate subfolders are the plant/disease class names, for example `dataset/PlantVillage/Apple___Apple_scab/...`.

## Setup (Windows PowerShell)
```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install jupyter ipykernel scikit-learn matplotlib
```

Open `AgroVision_Development.ipynb` in VS Code, select the `.venv` kernel, set `DATASET_DIR`, and run the cells from top to bottom. Training downloads ImageNet weights on first run and may take time.

The notebook saves the best model to `saved_model/best_model.keras`, class mapping to `class_names.py` and `saved_model/class_names.json`, and evaluation artifacts under `outputs/`.

## Launch UI
After training has created `saved_model/best_model.keras`:
```powershell
streamlit run app.py
```

Keep `app.py`, `predict.py`, `recommendation.py`, `class_names.py`, and `disease_database.json` in the same project directory. The notebook overwrites `class_names.py` based on the discovered dataset class order so the UI labels stay aligned with model outputs.

## Important
The dataset and trained model are not included in this package. No training metrics are claimed until you execute the notebook. Check that the actual dataset classes match the intended PlantVillage classes before training.
