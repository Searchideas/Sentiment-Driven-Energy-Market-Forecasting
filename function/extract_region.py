import spacy
from spacy.pipeline import EntityRuler
import os
import pandas as pd

class CountryNERMapper:
    def __init__(self, filepath):
        self.city_country_map, self.iso2_country_map, self.iso3_country_map, self.country_set = self.load_countries_map(filepath)
        self.nlp = self.initialize_spacy_model()

    def load_countries_map(self, filepath):
        countries_dictionary = pd.read_csv(filepath)
        columns_to_lower = ['city', 'city_ascii', 'country', 'admin_name', 'capital', 'iso2', 'iso3']
        countries_dictionary[columns_to_lower] = countries_dictionary[columns_to_lower].apply(lambda col: col.str.lower())

        city_country_map = dict(zip(countries_dictionary['city'].str.lower(), countries_dictionary['country']))
        iso2_country_map = dict(zip(countries_dictionary['iso2'].str.lower(), countries_dictionary['country']))
        iso3_country_map = dict(zip(countries_dictionary['iso3'].str.lower(), countries_dictionary['country']))
        country_set = set(countries_dictionary['country'].str.lower())

        return city_country_map, iso2_country_map, iso3_country_map, country_set

    def map_country(self, lemmatized_text):
        words = lemmatized_text.lower().split()

        for word in words:
            if word in self.city_country_map:
                return self.city_country_map[word]
            elif word in self.iso2_country_map:
                return self.iso2_country_map[word]
            elif word in self.iso3_country_map:
                return self.iso3_country_map[word]
            elif word in self.country_set:
                return word

        return 'Unknown'

    @staticmethod
    def download_spacy_model():
        model_name = 'en_core_web_sm'
        try:
            spacy.load(model_name)
        except OSError:
            print(f"Downloading the SpaCy model: {model_name}")
            os.system(f'python -m spacy download {model_name}')

    @staticmethod
    def setup_entity_ruler(nlp):
        ruler = nlp.add_pipe("entity_ruler", before="ner")
        patterns = [{"label": "GPE", "pattern": "iceland"}]  # Add more patterns as needed
        ruler.add_patterns(patterns)

    def initialize_spacy_model(self):
        self.download_spacy_model()
        nlp = spacy.load('en_core_web_sm')
        self.setup_entity_ruler(nlp)
        return nlp

    def extract_regions(self, tweet):
        doc = self.nlp(tweet)
        regions = [ent.text for ent in doc.ents if ent.label_ == "GPE"]
        return ", ".join(regions) if regions else None

    def get_final_location(self,Glossary_Map):
        """
        Combine mapping and region extraction to determine the final location.

        Args:
            lemmatized_text: A string of lemmatized text to analyze.

        Returns:
            The final location based on country mapping and extracted regions.
        """
        mapped_country = self.map_country(Glossary_Map)
        location = self.extract_regions(Glossary_Map)  # Use the same text for region extraction

        if mapped_country != 'Unknown':
            return mapped_country
        elif pd.notna(location):
            return location
        else:
            return 'Unknown'

# Example usage
# filepath = 'path_to_country_data.csv'
# country_ner_mapper = CountryNERMapper(filepath)
# Now you can directly create final_location in one line:

