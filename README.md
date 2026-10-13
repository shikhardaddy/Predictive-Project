# Predictive-Project
Handwritten Digit Recognition using Linear and Quadratic Discriminant Analysis
# Handwritten Digit Recognition using Linear and Quadratic Discriminant Analysis (LDA vs QDA)

**Course:** Predictive Analytics

## 1. Problem Statement
Automatically reading handwritten digits is used in postal sorting, cheque processing and form digitisation. This project classifies 8x8 pixel images of handwritten digits (0 to 9) with **Linear Discriminant Analysis (LDA)** and compares it with **Quadratic Discriminant Analysis (QDA)**.

## 2. Dataset
- **Optical Recognition of Handwritten Digits**, UCI Machine Learning Repository
- Link: https://archive.ics.uci.edu/dataset/80/optical+recognition+of+handwritten+digits
- 1,797 images, 64 features (pixel intensities 0 to 16), 10 balanced classes (about 180 images each); no missing values
- The same data ships with scikit-learn (`sklearn.datasets.load_digits`), so the code runs without downloading anything.

## 3. Method
1. EDA: sample images, class balance, average image per class
2. Stratified 75/25 train/test split (1,347 train, 450 test)
3. **LDA** (`solver="lsqr"`) assumes every class is Gaussian with a *shared* covariance matrix, giving linear decision boundaries. The covariance shrinkage was tuned by 5-fold cross-validation (best = 0.3).
4. **QDA** lets each class have its *own* covariance matrix, giving quadratic boundaries. Its `reg_param` was tuned the same way (best = 0.7); some pixels are constant within a class, so regularisation is required (`reg_param = 0` fails with a singular covariance error, which is expected).
5. Evaluation on the held-out test set
6. LDA is also used as a dimensionality reduction tool: the 64 pixels are projected onto 2 discriminants for visualisation

## 4. Results
| Model | CV accuracy | Test accuracy | Macro F1 |
|---|---|---|---|
| LDA (shrinkage 0.3) | 0.957 +/- 0.008 | 0.951 | 0.951 |
| QDA (reg_param 0.7) | 0.992 | **0.984** | **0.984** |

LDA's weakest digits are 1 (F1 0.87), 9 (0.89) and 8 (0.91). The first two discriminants capture 29% and 19% of the between-class variance, and the 2-D projection already separates several digits clearly.

## 5. Discussion
- QDA beats LDA by about 3.3 percentage points. Handwriting styles differ a lot between digits, so the assumption of one shared covariance matrix (LDA) is too restrictive, and class-specific covariances (QDA) fit better.
- QDA needed heavy regularisation because there are only about 135 training images per class for 64 features.
- LDA is faster, simpler and offers a useful low-dimensional view of the data, which QDA does not.
- Limitations: tiny 8x8 images, a single split, and no comparison with non-Gaussian methods such as SVM or neural networks.

## 6. Project Structure
```
train.py            # full pipeline
requirements.txt
outputs/
  metrics.json
  lda_model.joblib, qda_model.joblib
  01_sample_digits.png ... 06_lda_projection.png
```

## 7. How to Run
```bash
pip install -r requirements.txt
python train.py
```
