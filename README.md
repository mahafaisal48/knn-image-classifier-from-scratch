# KNN Image Classifier (From Scratch)

A K-Nearest Neighbors image classifier implemented from scratch in Python — no machine learning libraries (no scikit-learn, no numpy for the core logic). It classifies grayscale images by comparing two distance metrics, **Euclidean** and **Cosine similarity**, across multiple values of K.

## How It Works
1. **Data loading** — reads images from a dataset folder where each subfolder name is a class label (e.g. `dataset/cat/`, `dataset/dog/`).
2. **Preprocessing** — each image is loaded in grayscale with OpenCV, resized to 50x50, and flattened into a 1D feature vector.
3. **Train/test split** — for each class, images are shuffled and split 70% train / 30% test.
4. **Classification** — for every test image, distances to all training images are computed manually:
   - **Euclidean distance** — smaller is more similar.
   - **Cosine similarity** — larger is more similar.
5. **Voting** — for each K in `[3, 5, 7, 9]`, the K nearest neighbors vote on the predicted class (majority vote), computed separately for each distance metric.
6. **Evaluation** — prints classification accuracy (%) for both metrics, at each value of K.

## Sample Output
```
K= 3 : Euclidean  82.5 %, Cosine  85.0 %
K= 5 : Euclidean  84.2 %, Cosine  86.7 %
K= 7 : Euclidean  83.8 %, Cosine  87.1 %
K= 9 : Euclidean  81.9 %, Cosine  85.9 %
```
*(exact numbers depend on your dataset)*

## Requirements
- Python 3.x
- OpenCV (`pip install opencv-python`)
