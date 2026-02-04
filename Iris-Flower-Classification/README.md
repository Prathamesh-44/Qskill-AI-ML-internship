# Iris Flower Classification

## 📌 Objective
To classify iris flowers into three species — **Setosa**, **Versicolor**, and **Virginica** — based on sepal and petal measurements using a supervised machine learning approach.

---

## 📂 Dataset
This project uses the **Iris dataset**, a standard benchmark dataset available in the **scikit-learn library**, originally sourced from the **UCI Machine Learning Repository**.

### Features
- Sepal length (cm)
- Sepal width (cm)
- Petal length (cm)
- Petal width (cm)

### Target Classes
- Setosa
- Versicolor
- Virginica

---

## 🛠️ Steps Performed
1. Loaded the Iris dataset using scikit-learn.
2. Converted the dataset into a Pandas DataFrame for easier analysis.
3. Visualized the dataset using scatter plots to observe class separation.
4. Split the dataset into training and testing sets (80% training, 20% testing).
5. Trained a **K-Nearest Neighbors (KNN)** classifier.
6. Evaluated the model using accuracy score and confusion matrix.

---

## ⚙️ Algorithm Used
- **K-Nearest Neighbors (KNN)**
- Number of neighbors: **3**

---

## 📊 Results
The model achieved an accuracy of approximately **97%–100%** on the test dataset.  
Model performance was evaluated using an accuracy score and a confusion matrix.

---

## 🖼️ Output Screenshots

### Scatter Plot Visualization
![Iris Scatter Plot](screenshots/iris_scatter_plot.png)

### Model Accuracy & Confusion Matrix
![Model Output](screenshots/iris_accuracy_output_and_confusion_matrix.png)

---

## 🚀 How to Run
1. Install the required libraries:
   ```bash
   pip install -r requirements.txt
