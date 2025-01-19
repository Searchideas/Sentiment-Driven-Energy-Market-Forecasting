import sys

def safe_display(obj):
    if 'ipykernel' in sys.modules:
        from IPython.display import display
        display(obj)
    else:
        print(obj)
        
import requests
import json
import spacy
import nltk
from nltk.corpus import wordnet
import ipywidgets as widgets
#from IPython.display import display  # Comment out or remove this line

# Check and download necessary models and datasets
def ensure_models_and_datasets():
    # Check and download spaCy model
    try:
        nlp = spacy.load("en_core_web_sm")
    except OSError:
        print("Downloading spaCy model...")
        spacy.cli.download("en_core_web_sm")
        nlp = spacy.load("en_core_web_sm")
    
    # Check and download NLTK wordnet
    try:
        wordnet.all_synsets()
    except LookupError:
        print("Downloading NLTK wordnet...")
        nltk.download('wordnet', quiet=True)

# Call this function at the beginning of your script
ensure_models_and_datasets()

class GlossaryProcessor:
    def __init__(self, api_endpoint, headers):
        """
        Initialize the GlossaryProcessor with the API endpoint and headers.
        
        Parameters:
        - api_endpoint (str): The API URL to fetch the glossary data.
        - headers (dict): The headers required to make a request to the API.
        """
        self.api_endpoint = api_endpoint
        self.headers = headers
        self.glossary_dict = {}

    def fetch_glossary(self):
        """
        Fetch the glossary data from the API and store it as a dictionary for quick lookup.
        """
        print("Fetching glossary from API...")
        response = requests.get(self.api_endpoint, headers=self.headers)
        if response.status_code == 200:
            glossary_terms = json.loads(response.content)
            self.glossary_dict = {term["term"].lower(): term["definition"] for term in glossary_terms["glossary"] if term["term"] is not None}
            print(f"Glossary fetched successfully with {len(self.glossary_dict)} terms.")
        else:
            print(f"Failed to fetch glossary. Status code: {response.status_code}")
            self.glossary_dict = {}

    def word_has_wordnet_definition(self, word):
        """
        Check if a word has a WordNet definition.
        
        Parameters:
        - word (str): The word to check in WordNet.
        
        Returns:
        - bool: True if the word has a WordNet definition, False otherwise.
        """
        synsets = wordnet.synsets(word)
        return bool(synsets)

    def process_row(self, row):
        """
        Process a single row to map glossary terms and common words.
        
        Parameters:
        - row (pd.Series): A row of the DataFrame with a 'lemmatized_text' column.
        
        Returns:
        - str: A comma-separated string of glossary definitions or common words.
        """
        lemmatized_text = row["lemmatized_text"].lower().split()
        glossary_terms_list = []

        for word in lemmatized_text:
            if self.word_has_wordnet_definition(word):
                glossary_terms_list.append(word)  # Common word
            elif word in self.glossary_dict:
                glossary_terms_list.append(self.glossary_dict[word])  # Glossary term
        
        return ', '.join(glossary_terms_list)

    def process_glossary_mapping(self, data):
        """
        Process the entire DataFrame and map glossary terms and common words to a new column.
        
        Parameters:
        - data (pd.DataFrame): The DataFrame with 'lemmatized_text' column.
        
        Modifies the DataFrame in place, adding a 'Glossary_Map' column.
        """
        num_rows = len(data)
        progress = widgets.IntProgress(value=0, max=num_rows, description='Processing:')
        safe_display(progress)

        for index, row in data.iterrows():
            data.at[index, 'Glossary_Map'] = self.process_row(row)
            if index % 1000 == 0:
                print(f"Processed {index} rows")
            progress.value = index + 1
        
        progress.value = num_rows
        print("Glossary mapping completed.")
'''
# Usage Example:
# Set API endpoint and headers
api_endpoint = "https://api.weather.gov/glossary"
headers = {
    "Accept": "application/ld+json",
    "User-Agent": "MyWeatherApp/1.0"
}

# Initialize the GlossaryProcessor
glossary_processor = GlossaryProcessor(api_endpoint, headers)

# Fetch the glossary data
glossary_processor.fetch_glossary()

# Process the DataFrame to map glossary terms
# Assuming `data` is a DataFrame with a 'lemmatized_text' column
glossary_processor.process_glossary_mapping(data)

# Verify the results
print(data[['lemmatized_text', 'Glossary_Map']]) '''