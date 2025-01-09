# Toxic Sentiment Detector

This repository contains the code and resources for a **Toxic Sentiment Detection** project. The goal of this project is to classify text as **toxic or non-toxic** using **Natural Language Processing (NLP) and Machine Learning (ML)** techniques.

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Model Training](#model-training)
- [Evaluation](#evaluation)
- [Contributors](#contributors)
- [License](#license)

## Overview

The rise of social media has led to an increase in online toxicity. This project aims to detect toxic sentiments in text-based content, helping to identify and filter harmful messages. The system is built using **Python, Streamlit, and Machine Learning models**.

## Features

- **Text Classification**: Detects toxic sentiment in user-provided text.
- **Machine Learning Model**: Trained using NLP techniques.
- **Web Interface**: Built using **Streamlit** for easy user interaction.

## Installation

To run this project locally, ensure you have Python installed. Then, follow these steps:

1. **Clone the repository:**

   ```bash
   git clone https://github.com/SahilPitale06/Toxic-Sentiment-Detector.git
   cd Toxic-Sentiment-Detector
   ```

2. **Create a virtual environment (optional but recommended):**

   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install the required dependencies:**

   ```bash
   pip install -r requirements.txt
   ```

   *Note: If `requirements.txt` is not available, manually install the necessary packages:*

   ```bash
   pip install numpy pandas scikit-learn flask streamlit nltk
   ```

4. **Download NLP resources (if using NLTK):**

   ```bash
   import nltk
   nltk.download('stopwords')
   nltk.download('punkt')
   ```

## Usage

1. **Run the Streamlit application:**

   ```bash
   streamlit run app.py
   ```

2. **Access the web interface:**

   Open a web browser and navigate to the **local Streamlit URL** displayed in the terminal. You can enter text to check whether it contains toxic sentiment.

## Model Training

The model training process is documented in the Jupyter Notebook `toxic_sentiment.ipynb`. It includes:

- **Data Preprocessing**: Cleaning and tokenizing text data.
- **Feature Engineering**: Using NLP techniques like **TF-IDF Vectorization**.
- **Model Selection**: Training ML models such as **Logistic Regression, Naive Bayes, and Random Forest**.
- **Model Serialization**: Saving the trained model (`model.pkl`) for future use.

## Evaluation

The trained model's performance is evaluated using metrics like **accuracy, precision, recall, and F1-score**. Detailed evaluation results and visualizations are available in the Jupyter Notebook.

## Contributors

- [Sahil Pitale](https://github.com/SahilPitale06)

## License

This project is licensed under the **MIT License**. See the [LICENSE](LICENSE) file for more details.

---

*Note: This README provides a general overview. For detailed explanations and code insights, refer to the Jupyter Notebook and Python scripts in the repository.*
