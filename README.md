# 📧 Spam Mail Detection App

A simple web application built using **Streamlit** and **Scikit-learn** that allows users to upload a dataset of emails and classify new email messages as **spam** or **ham** using a Logistic Regression model.

## 🚀 Features

- Upload a CSV dataset of email messages
- Preprocess and vectorize text using TF-IDF
- Train and evaluate a Logistic Regression classifier
- View training and test accuracy
- Classify user-input email messages

## 🛠️ Tech Stack

- Python
- Streamlit
- Pandas
- Scikit-learn
- NumPy

## 📂 Dataset Format

Your CSV file should contain at least the following two columns:

- `Category`: Label of the mail (`spam` or `ham`)
- `Message`: The text content of the email

Example:

| Category | Message                            |
|----------|------------------------------------|
| ham      | Hey, are we still meeting today?   |
| spam     | Win a free iPhone by clicking here!|

## 🏁 Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/yourusername/spam-mail-detector.git
cd spam-mail-detector
