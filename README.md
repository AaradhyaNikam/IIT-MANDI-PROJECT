# Plant Village Classification Project

This repository contains a Jupyter notebook and a trained Keras model for classifying plant diseases using the PlantVillage dataset.

Repository structure

- `Code.ipynb` - Main Jupyter notebook with data processing, training/inference examples.
- `models/`
  - `model.keras` - Trained Keras model (HDF5/keras saved model format).
  - `class_mapping.json` - Mapping from class indices to human-readable labels.
- `requirements.txt` - Python package dependencies for running the notebook and inference.

Setup (Windows PowerShell)

1. Create and activate a virtual environment (recommended):

```powershell
python -m venv .venv; .\.venv\Scripts\Activate.ps1
```

2. Upgrade pip and install dependencies:

```powershell
python -m pip install --upgrade pip; pip install -r requirements.txt
```

Running the notebook

1. Start Jupyter Lab/Notebook:

```powershell
jupyter notebook
```

2. Open `Code.ipynb` in your browser.

Quick inference example (Python snippet)

```python
import json
from tensorflow import keras
import numpy as np
from PIL import Image

# Load model and class mapping
model = keras.models.load_model('models/model.keras')
with open('models/class_mapping.json','r') as f:
    class_map = json.load(f)

# Preprocess function (adapt to how the model was trained)
def preprocess_image(path, target_size=(224,224)):
    img = Image.open(path).convert('RGB').resize(target_size)
    arr = np.array(img)/255.0
    return np.expand_dims(arr, axis=0)

# Example usage
img = preprocess_image('path/to/test_image.jpg')
preds = model.predict(img)
label_idx = int(np.argmax(preds, axis=1)[0])
print('Predicted:', class_map.get(str(label_idx), label_idx))
```

Notes

- The provided `preprocess_image` uses simple resizing and scaling to [0,1]; adjust normalization if the model expects different preprocessing (mean-subtraction, different input size, etc.).
- If your model was saved in the newer TensorFlow SavedModel format, the path may differ.
- For GPU acceleration, install the appropriate TensorFlow GPU package and drivers. See TensorFlow docs for guidance.

Contact

For questions or improvements, open an issue or contact the repo owner.
