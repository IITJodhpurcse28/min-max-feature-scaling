# Min-Max Normalization (Feature Scaling) with Scikit-Learn

A beginner-friendly notebook demonstrating **Min-Max Scaling** — one of the most important preprocessing steps in machine learning — using the classic Wine dataset.

---

## What is Min-Max Normalization?

Min-Max Normalization (also called **Min-Max Scaling**) rescales every feature so that its values fall between **0 and 1**, using the formula:

```
X_scaled = (X - X_min) / (X_max - X_min)
```

> **Why does this matter?**  
> Many ML algorithms (like KNN, SVM, and neural networks) are sensitive to the *scale* of features. If one feature ranges from 0–1 and another from 0–1000, the larger one will dominate the model's learning. Scaling puts all features on equal footing.

---

## Dataset

**UCI Wine Dataset** — a classic multiclass classification dataset with 178 samples across 3 wine classes.

This notebook uses only 3 columns:
| Column | Description |
|--------|-------------|
| `class_label` | Wine class (1, 2, or 3) |
| `alcohol` | Alcohol content |
| `malic_acid` | Malic acid content |

> The dataset is loaded from `wine_data.csv`. You can download the full UCI Wine dataset from [here](https://archive.ics.uci.edu/dataset/109/wine).

---

## What This Notebook Covers

1. **Loading & Exploring the Data** — Reading the CSV, naming columns, and inspecting the first few rows
2. **Visualizing the Raw Data** — Scatter plot of `alcohol` vs `malic_acid` colored by class label using Seaborn
3. **Train-Test Split** — 75% training / 25% test split using `train_test_split` with `random_state=0` for reproducibility
4. **Applying MinMaxScaler** — Fitting the scaler **only on training data**, then transforming both train and test sets
5. **Before vs After Comparison** — Side-by-side scatter plots showing the data distribution before and after scaling

> **Important:** The scaler is fit **only on `x_train`**, not on the full dataset. Fitting on test data would cause **data leakage** — the model would be "peeking" at test information during training, giving falsely optimistic results.

---

## Key Concept: Why Fit on Train Only?

```python
scaler = MinMaxScaler()
scaler.fit(x_train)          # Learn min/max from training data only
x_train_scaled = scaler.transform(x_train)
x_test_scaled  = scaler.transform(x_test)   # Apply same scale to test
```

Never do `scaler.fit(x_test)` or `scaler.fit_transform` on the full dataset before splitting — that leaks information from the test set into your preprocessing.

---

## Libraries Used

- `numpy` — numerical operations
- `pandas` — data loading and manipulation
- `matplotlib` — plotting
- `seaborn` — styled scatter plots
- `scikit-learn` — `train_test_split`, `MinMaxScaler`

---

## How to Run

```bash
git clone https://github.com/your-username/your-repo-name.git
cd your-repo-name
pip install numpy pandas matplotlib seaborn scikit-learn
jupyter notebook min_max_normalization.ipynb
```

Make sure `wine_data.csv` is in the same directory as the notebook.

---

## Files

```
├── min_max_normalization.ipynb   # Main notebook
├── wine_data.csv                 # Dataset (alcohol, malic_acid, class_label)
└── README.md
```

---

## Related Techniques

| Technique | When to Use |
|-----------|-------------|
| **Min-Max Scaling** (this notebook) | When you need values bounded to [0, 1]; sensitive to outliers |
| **Standard Scaling (Z-score)** | When data is roughly Gaussian; less sensitive to outliers |
| **Robust Scaling** | When data has significant outliers |
