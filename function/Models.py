import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from textblob import TextBlob
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

# Download required NLTK data
nltk.download('vader_lexicon')
nltk.download('wordnet')

class SentimentAnalyzer:
    def __init__(self):
        """
        Initialize the SentimentAnalyzer with multiple sentiment models:
        - VADER SentimentIntensityAnalyzer
        - TextBlob
        - Pre-trained XLM-Roberta model for Twitter sentiment
        - ClimateBERT model for climate-related sentiment analysis
        """
        self.sia = SentimentIntensityAnalyzer()

        # Initialize pre-trained XLM-Roberta model and tokenizer for sentiment analysis
        self.xlm_model_name = "cardiffnlp/twitter-xlm-roberta-base-sentiment"
        self.xlm_model = AutoModelForSequenceClassification.from_pretrained(self.xlm_model_name)
        self.xlm_tokenizer = AutoTokenizer.from_pretrained(self.xlm_model_name)

        # Initialize ClimateBERT model and tokenizer
        self.climatebert_model_name = "climatebert/distilroberta-base-climate-sentiment"
        self.climatebert_tokenizer = AutoTokenizer.from_pretrained(self.climatebert_model_name)
        self.climatebert_model = AutoModelForSequenceClassification.from_pretrained(self.climatebert_model_name)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.climatebert_model.to(self.device)

    def vader_sentiment(self, text):
        """
        Analyze sentiment using VADER.
        
        Parameters:
        - text (str): The input text to analyze.
        
        Returns:
        - str: The sentiment label ("Positive", "Negative", or "Neutral").
        """
        scores = self.sia.polarity_scores(text)
        if scores['compound'] >= 0.05:
            return "Positive"
        elif scores['compound'] <= -0.05:
            return "Negative"
        else:
            return "Neutral"

    def textblob_sentiment(self, text):
        """
        Analyze sentiment using TextBlob.
        
        Parameters:
        - text (str): The input text to analyze.
        
        Returns:
        - str: The sentiment label ("Positive", "Negative", or "Neutral").
        """
        blob = TextBlob(text)
        if blob.sentiment.polarity > 0:
            return "Positive"
        elif blob.sentiment.polarity < 0:
            return "Negative"
        else:
            return "Neutral"

    def xlm_sentiment(self, text):
        """
        Analyze sentiment using pre-trained XLM-Roberta sentiment model for Twitter.
        
        Parameters:
        - text (str): The input text to analyze.
        
        Returns:
        - str: The sentiment label ("negative", "neutral", or "positive").
        """
        inputs = self.xlm_tokenizer(text, return_tensors='pt', max_length=512, truncation=True, padding='max_length')
        with torch.no_grad():
            outputs = self.xlm_model(**inputs)
        scores = torch.nn.functional.softmax(outputs.logits, dim=1)
        sentiment = torch.argmax(scores).item()
        labels = {0: "negative", 1: "neutral", 2: "positive"}
        return labels[sentiment]

    def climatebert_sentiment(self, text):
        """
        Analyze sentiment using ClimateBERT model for climate-related sentiment analysis.
        
        Parameters:
        - text (str): The input text to analyze.
        
        Returns:
        - str: The sentiment label ("positive", "neutral", or "negative").
        """
        inputs = self.climatebert_tokenizer(text, padding=True, truncation=True, return_tensors='pt', max_length=512)
        inputs = {key: value.to(self.device) for key, value in inputs.items()}
        
        with torch.no_grad():
            outputs = self.climatebert_model(**inputs)
        
        logits = outputs.logits
        probabilities = torch.nn.functional.softmax(logits, dim=1)
        prediction = torch.argmax(probabilities, dim=1).item()
        
        # Define label mapping
        label_map = {0: 'positive', 1: 'neutral', 2: 'negative'}
        return label_map[prediction]

    def analyze_all(self, df, text_column):
        """
        Apply all sentiment analysis models on a DataFrame and create new columns with the results.
        
        Parameters:
        - df (pd.DataFrame): The input DataFrame containing text data.
        - text_column (str): The name of the column containing the text to analyze.
        
        Returns:
        - pd.DataFrame: The DataFrame with new sentiment columns added.
        """
        df['Vader sentiment'] = df[text_column].apply(self.vader_sentiment)
        df['Textblob sentiment'] = df[text_column].apply(self.textblob_sentiment)
        df['XLM sentiment'] = df[text_column].apply(self.xlm_sentiment)
        
        # Process ClimateBERT in batches due to GPU/memory constraints
        texts = df[text_column].astype(str).tolist()
        predicted_labels = []
        
        for text in texts:
            predicted_label = self.climatebert_sentiment(text)
            predicted_labels.append(predicted_label)
        
        df['ClimateBERT Sentiment'] = predicted_labels
        return df


'''# Initialize the SentimentAnalyzer
analyzer = SentimentAnalyzer()

# Assuming processed_data is a DataFrame with a 'lemmatized_text_2' column
processed_data = analyzer.analyze_all(processed_data, 'lemmatized_text_2')

# Display results
print(processed_data[['lemmatized_text_2', 'Vader sentiment', 'Textblob sentiment', 'XLM sentiment', 'ClimateBERT Sentiment']])'''