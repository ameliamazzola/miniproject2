# miniproject2

# MiniProject2 — Traffic Vehicle Recognition

## Objective
Use the UA-DETRAC dataset to:

- Detect vehicles in traffic videos
- Compute:
  - Maximum road load (vehicles in a single frame)
  - Traffic flow over time (vehicles passing through)

---
```
## Project Structure
miniproject2/
│
├── notebooks/
│ ├── train.ipynb
│ ├── test.ipynb
│ ├── inference.ipynb
│
├── src/
│ ├── __init__.py
│ ├── dataset_utils.py
│ ├── train_utils.py
│ ├── eval_utils.py
│ └── inference_utils.py
│
├── model/
├── outputs/
│ ├── plots/
│ └── videos/
```
## Setup (Google Colab)

```python
!git clone https://github.com/ameliamazzola/miniproject2.git
%cd miniproject2
!pip install -r requirements.txt
