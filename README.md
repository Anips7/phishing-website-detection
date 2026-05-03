# Phishing Website Detection Using Machine Learning
### Class Project

---

## Objective
Build a machine learning system that automatically detects phishing websites based on URL and webpage features.

---

## Project Structure
```
phishing_detection/
├── phishing_detection.py   ← Main script (run this)
├── predict.py              ← Predict on new URLs
├── requirements.txt        ← Python dependencies
├── data/
│   └── phishing.csv        ← (Download UCI dataset here — optional)
└── outputs/
    ├── results.csv             ← Model performance table
    ├── model_comparison.png    ← Bar chart of all models
    ├── confusion_matrices.png  ← Confusion matrices
    └── feature_importance.png  ← Top 15 features (Random Forest)
```

---

## Setup & Run

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. (Optional) Use the real UCI dataset
- Download from: https://archive.ics.uci.edu/ml/datasets/Phishing+Websites
- Save file as `data/phishing.csv`
- The script auto-detects it. Without it, a realistic synthetic dataset is used.

### 3. Run the project
```bash
python phishing_detection.py
```

### 4. View results
All output files are saved to the `outputs/` folder.

---

## Models Used

| Model | Type | Library |
|---|---|---|
| Logistic Regression | Linear classifier | scikit-learn |
| Random Forest | Ensemble (bagging) | scikit-learn |
| Gradient Boosting | Ensemble (boosting) | scikit-learn |
| SVM | Kernel-based | scikit-learn |
| Deep Neural Network | Neural network | TensorFlow/Keras |

---

## Dataset Features (30 total)

The UCI Phishing dataset uses these feature categories:
- **URL-based**: IP in URL, URL length, shortening service, @ symbol, double slash, prefix/suffix, subdomains
- **Domain-based**: SSL state, domain registration length, DNS record, domain age
- **HTML/JS-based**: Favicon, iFrame, mouseover, right-click disabled, popups
- **External**: Google index, page rank, web traffic, links pointing to page

All features are encoded as: `-1` (phishing indicator), `0` (suspicious), `1` (legitimate)

---

## Expected Results

| Model | Accuracy |
|---|---|
| Random Forest | ~97% |
| Gradient Boosting | ~96% |
| Deep Neural Network | ~95% |
| SVM | ~92% |
| Logistic Regression | ~88% |

---

## Report Sections (for submission)

1. **Introduction** — What is phishing? Why detect it?
2. **Dataset** — UCI Phishing dataset, 11,055 samples, 30 features
3. **Methodology** — Pipeline: collection → extraction → preprocessing → training → evaluation
4. **Results** — Model comparison table + plots
5. **Conclusion** — Random Forest performs best; ensemble methods outperform linear models

---

*Built for academic/class purposes using publicly available data.*
