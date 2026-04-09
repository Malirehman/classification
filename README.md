# 🧠 Breast Cancer Classification using KNN

This project implements a **machine learning classification model** using the **K-Nearest Neighbors (KNN)** algorithm to predict whether a tumor is **malignant** or **benign**.

---

## 📊 Dataset

The dataset is loaded using:

```python
from sklearn.datasets import load_breast_cancer
```

It is a built-in dataset from scikit-learn.

### Features:

* 30 numerical features related to tumor characteristics
* Examples: radius, texture, perimeter, area, smoothness

### Target:

* `0` → Malignant (Cancerous)
* `1` → Benign (Non-cancerous)

---

## ⚙️ Steps Performed

1. Loaded dataset using `load_breast_cancer`
2. Split data into training and testing sets
3. Applied feature scaling using `StandardScaler`
4. Trained a **K-Nearest Neighbors (KNN)** model
5. Evaluated model performance using accuracy score

---

## 🛠️ Technologies Used

* Python
* NumPy
* Scikit-learn

---

## 🚀 Implementation

```python
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier

# Load dataset
X, y = load_breast_cancer(return_X_y=True)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Feature scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Model training
knn = KNeighborsClassifier()
knn.fit(X_train_scaled, y_train)

# Model evaluation
accuracy = knn.score(X_test_scaled, y_test)
print("Accuracy:", accuracy)
```

---

## 📈 Model Performance

* The model is evaluated using **accuracy score**
* Accuracy may vary depending on data split and parameters

---

## 🔍 Key Concepts

* **KNN Algorithm**: Classifies data based on nearest neighbors
* **Feature Scaling**: Important for distance-based algorithms
* **Train-Test Split**: Prevents overfitting and ensures fair evaluation

---

## 🚀 Future Improvements

* Tune `n_neighbors` for better accuracy
* Add:

  * Confusion Matrix
  * Precision & Recall
* Compare with other models:

  * Logistic Regression
  * Decision Tree

---

## 📌 Conclusion

This project demonstrates a complete machine learning workflow using KNN, including preprocessing, training, and evaluation.


and the next file is logistic regression and loan approval problems classifying weather the  person has his loan approved or not

---

## 👨‍💻 Author

Muhammad Ali Rehman
