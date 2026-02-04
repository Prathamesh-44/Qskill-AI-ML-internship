# Spam Mail Detector

## 📌 Objective
To build a machine learning model that classifies messages as **Spam** or **Ham (Not Spam)** using Natural Language Processing techniques.

---

## 📂 Dataset
The project uses the **SMS Spam Collection Dataset**, a public dataset from the **UCI Machine Learning Repository**.

Each message is labeled as:
- Spam
- Ham

---

## 🛠️ Steps Performed
1. Loaded and explored the dataset.
2. Preprocessed text by:
   - Converting to lowercase
   - Removing punctuation and stopwords
3. Converted text into numerical features using **TF-IDF**.
4. Split the data into training and testing sets.
5. Trained a **Naive Bayes classifier**.
6. Evaluated the model using accuracy, precision, recall, and F1 score.

---

## ⚙️ Algorithms Used
- TF-IDF Vectorization
- Multinomial Naive Bayes

---

## 📊 Results
The model achieved an accuracy of approximately **96% to 97%** on the test dataset.

---

## 🚀 How to Run
1. Install required libraries:
   ```bash
   pip install -r requirements.txt
