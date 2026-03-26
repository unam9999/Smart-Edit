---
description: how to run the VisionSort ML model
---

1. Open a terminal and navigate to the project directory:
```powershell
cd "c:\Users\Manu T J\Documents\CODING\VisionSort"
```

2. Create a virtual environment (if you haven't already):
```powershell
python -m venv .venv
```

3. Activate the virtual environment:
```powershell
.venv\Scripts\Activate.ps1
```

4. Install the required libraries:
```powershell
pip install -r requirements.txt
```

5. Prepare your data:
    * Place your training images in the `data/` folder.
    * Each folder (e.g., `data/animal/`, `data/portrait/`) should contain images for that category.

6. Start training the model:
```powershell
python backend/train.py
```

7. To test the model on a single image (after training):
```powershell
python backend/predict.py --image path/to/your/image.jpg
```

8. To see training progress in your browser:
```powershell
tensorboard --logdir backend/logs
```
