# 💊 Medical Drug Review Sentiment Classification

A machine learning project that classifies medical drug reviews into sentiment categories (Positive, Neutral, Negative) using Natural Language Processing (NLP) and Naive Bayes classification.

## 📋 Overview

This project analyzes patient reviews of medications to automatically categorize them based on sentiment. Using the drugsComTest dataset, the model processes text reviews and predicts whether a patient's experience with a medication was positive, neutral, or negative based on their rating and review text.

## 🎯 Objectives

- Preprocess medical text data with tokenization, lemmatization, and stopword removal
- Categorize drug reviews into three sentiment classes based on ratings
- Build a Multinomial Naive Bayes classifier for sentiment prediction
- Evaluate model performance using accuracy, precision, recall, and F1-score
- Visualize results with confusion matrix analysis

## 📊 Dataset

- **Source**: drugsComTest_raw.csv
- **Size**: 53,766 patient reviews
- **Features**:
  - Drug name
  - Medical condition
  - Review text
  - Patient rating (1-10)
  - Date and useful count

**Sentiment Categorization**:
- **Positive**: Rating ≥ 9
- **Neutral**: Rating 5-8
- **Negative**: Rating 1-4

## 🔧 Technologies Used

- **Python**: Core programming language
- **Pandas & NumPy**: Data manipulation and analysis
- **NLTK**: Natural Language Processing (tokenization, lemmatization, stopwords)
- **Scikit-learn**: Machine learning models and evaluation metrics
- **Matplotlib**: Data visualization
- **CountVectorizer**: Text feature extraction (Bag of Words)
- **MultinomialNB**: Naive Bayes classification algorithm

## 🚀 Installation

```bash
# Install required packages
pip install pandas numpy nltk scikit-learn matplotlib

# Download NLTK data
python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords'); nltk.download('wordnet')"
```

## 💻 Usage

1. **Load and clean data**:
   - Remove missing values
   - Extract review and rating columns

2. **Preprocess text**:
   - Convert to lowercase
   - Remove punctuation
   - Tokenize text
   - Remove stopwords
   - Apply lemmatization

3. **Feature extraction**:
   - Convert text to numerical vectors using CountVectorizer (Bag of Words)

4. **Train model**:
   - Split data (80% train, 20% test)
   - Train Multinomial Naive Bayes classifier

5. **Evaluate performance**:
   - Generate classification report
   - Create confusion matrix visualization

## 📈 Results

```
Model Performance:
- Overall Accuracy: 63.8%

Class-wise Performance:
├── Negative
│   ├── Precision: 65%
│   ├── Recall: 61%
│   └── F1-Score: 63%
├── Neutral
│   ├── Precision: 48%
│   ├── Recall: 39%
│   └── F1-Score: 43%
└── Positive
    ├── Precision: 69%
    ├── Recall: 79%
    └── F1-Score: 74%
```

**Key Insights**:
- Best performance on Positive sentiment (highest recall at 79%)
- Moderate performance on Negative sentiment
- Neutral category is most challenging to classify
- Model shows bias toward positive predictions

## 📊 Confusion Matrix

The confusion matrix visualization shows the model's prediction patterns across all three sentiment categories, highlighting strengths (positive classification) and weaknesses (neutral classification).

## 🔮 Future Improvements

- Experiment with TF-IDF vectorization for better feature representation
- Try advanced models (SVM, Random Forest, LSTM)
- Handle class imbalance using SMOTE or class weights
- Incorporate bigrams/trigrams for context preservation
- Use domain-specific medical word embeddings
- Perform hyperparameter tuning for optimization
- Add cross-validation for robust evaluation

## 💡 Applications

- **Patient Feedback Analysis**: Automatically categorize patient reviews for pharmaceutical companies
- **Drug Safety Monitoring**: Identify negative experiences requiring investigation
- **Healthcare Decision Support**: Help patients make informed medication choices
- **Pharmaceutical Research**: Analyze medication effectiveness trends
- **Regulatory Compliance**: Monitor adverse drug reactions from patient reports

## 📁 Project Structure

```
├── Medical Drug review classification.ipynb
├── drugsComTest_raw.csv
└── README.md
```

## 🤝 Contributing

Contributions are welcome! Areas for improvement:
- Better text preprocessing techniques
- Alternative classification algorithms
- Enhanced visualization methods
- Feature engineering strategies

## 📄 License

This project is available for educational and research purposes.

---

**Note**: This model is for educational purposes and should not replace professional medical advice. Always consult healthcare professionals for medication decisions.
