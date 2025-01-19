# This will be our preprocessing file

"""
Our preprocessing step:
1. Clean text - remove html, keep letters and spaces, remove digit and extra space
2. Text Normalization and removed stop word

"""
import pandas as pd
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import re
class TextPreprocessor:
    def __init__(self, data, text_column):
        self.data = data
        self.text_column = text_column
        self.stop_words = set(stopwords.words('english'))
        self.lemmatizer = WordNetLemmatizer()

        # Automatically run the preprocessing steps upon initialization
        self.run_preprocessing()

    def clean_text(self, text):
        """Clean the input text by removing URLs and unwanted characters."""
        text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', text)
        text = re.sub(r'www\.[a-zA-Z0-9\-]+\.[a-zA-Z]{2,}', '', text)
        text = re.sub(r'[^\w\s]', '', text)  # Keeps only letters and spaces
        text = re.sub(r'\d+', '', text)  # Remove digits
        text = re.sub(r'\s+', ' ', text).strip()  # Strip extra spaces
        return text

    def preprocess_text(self):
        """Clean and normalize text."""
        self.data['cleaned_text'] = self.data[self.text_column].apply(self.clean_text)
        self.data['cleaned_text'] = self.data['cleaned_text'].str.lower()

    def tokenize(self):
        """Tokenize the cleaned text."""
        self.data['tokenized_text'] = self.data['cleaned_text'].apply(word_tokenize)

    def remove_stopwords(self):
        """Remove stop words from tokenized text."""
        self.data['stopword_removed'] = self.data['tokenized_text'].apply(
            lambda x: [word for word in x if word not in self.stop_words]
        )

    def lemmatize(self):
        """Lemmatize the stopword-removed text."""
        self.data['lemmatization'] = self.data['stopword_removed'].apply(
            lambda x: [self.lemmatizer.lemmatize(word) for word in x]
        )
        self.data['lemmatized_text'] = self.data['lemmatization'].apply(lambda x: ' '.join(x))
    
    def final(self):
            """Drop unnecessary columns and return the cleaned DataFrame."""
            columns_to_drop = ['stopword_removed', 'tokenized_text', 'lemmatization']
            self.data.drop(columns=columns_to_drop, inplace=True, errors='ignore')
            return self.data

    def run_preprocessing(self):
        """Run all preprocessing steps sequentially."""
        self.preprocess_text()
        self.tokenize()
        self.remove_stopwords()
        self.lemmatize()
        self.final()

# Example usage:
# df = pd.DataFrame({'text': ["This is a sample sentence.", "Another example sentence."]})
# processor = TextPreprocessor(df, 'text')
# print(processor.data)  # Preprocessed DataFrame