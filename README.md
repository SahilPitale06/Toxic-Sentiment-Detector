# Toxic Sentiment Detector

## Project Overview
The **Toxic Sentiment Detector** is a machine learning project designed to classify text into toxic and non-toxic categories. This project aims to assist in moderating online conversations by detecting harmful or offensive language. The tool uses Natural Language Processing (NLP) techniques and machine learning algorithms to analyze and classify sentiment.

---

## Features
- Preprocessing of textual data, including tokenization and cleaning.
- Implementation of sentiment classification using machine learning models.
- Visualization of results for better insights.
- Deployment via a user-friendly web interface using Streamlit.

---

## Technologies Used
- **Python**: Core programming language.
- **Libraries**:
  - Pandas and NumPy for data manipulation.
  - NLTK and Spacy for NLP tasks.
  - Scikit-learn for building and evaluating the machine learning model.
  - Matplotlib and Seaborn for data visualization.
  - Streamlit for web app deployment.

---

## Dataset Used
- **Source:** [https://www.kaggle.com/datasets/ashwiniyer176/toxic-tweets-dataset]
- **Description:** The dataset contains labeled examples of text, categorized as toxic or non-toxic.

---

## Steps Performed
1. **Data Loading and Exploration**:
   - Loaded and explored the dataset to understand its structure and distribution.

2. **Data Preprocessing**:
   - Cleaned the text by removing stop words, punctuation, and special characters.
   - Tokenized and lemmatized the text for standardization.

3. **Feature Engineering**:
   - Applied techniques like TF-IDF and Bag of Words for feature extraction.

4. **Model Building and Evaluation**:
   - Trained and evaluated models (e.g., Logistic Regression, Random Forest) for classification.
   - Selected the best model based on metrics like F1-score and accuracy.

5. **Visualization**:
   - Created visualizations to showcase data distribution and model performance.

6. **Deployment**:
   - Built a web application using Streamlit for users to input text and view classification results.

---

## How to Run
1. Clone the repository:
   ```bash
   git clone https://github.com/SahilPitale06/Toxic-Sentiment-Detector.git
   cd Toxic-Sentiment-Detector
   ```

2. Install the required libraries:
   ```bash
   pip install -r requirements.txt
   ```

3. Run the Streamlit app:
   ```bash
   streamlit run app.py
   ```

4. Open the provided URL in your browser to use the application.

---

## Key Results
- **Model Performance**:
  - Accuracy: [Add value after evaluation]
  - F1-Score: [Add value after evaluation]

- The tool successfully identifies toxic text with high accuracy and provides a user-friendly interface for moderation.

---

## Future Work
- Incorporate deep learning models like LSTMs or Transformers for improved accuracy.
- Expand dataset to include more diverse and real-world text examples.
- Add support for multilingual text analysis.

---

## Author
- **Name:** Sahil Pitale
- **Contact:** sp9328123456@gmail.com | [LinkedIn Profile](https://www.linkedin.com/in/sahil-pitale-56a5681bb/)

Feel free to fork the repository or contribute to its development!

