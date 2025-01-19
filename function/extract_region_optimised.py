import spacy
from spacy.pipeline import EntityRuler
import os
import pandas as pd
from functools import lru_cache

class CountryNERMapper:
    def __init__(self, filepath):
        self.city_country_map, self.iso2_country_map, self.iso3_country_map, self.country_set = self.load_countries_map(filepath)
        self.nlp = self.initialize_spacy_model()
        self.location_cache = {}

    def load_countries_map(self, filepath):
        countries_dictionary = pd.read_csv(filepath)
        columns_to_lower = ['city', 'city_ascii', 'country', 'admin_name', 'capital', 'iso2', 'iso3']
        countries_dictionary[columns_to_lower] = countries_dictionary[columns_to_lower].apply(lambda col: col.str.lower())

        city_country_map = dict(zip(countries_dictionary['city'].str.lower(), countries_dictionary['country']))
        iso2_country_map = dict(zip(countries_dictionary['iso2'].str.lower(), countries_dictionary['country']))
        iso3_country_map = dict(zip(countries_dictionary['iso3'].str.lower(), countries_dictionary['country']))
        country_set = set(countries_dictionary['country'].str.lower())

        return city_country_map, iso2_country_map, iso3_country_map, country_set

    @lru_cache(maxsize=1024)
    def map_country(self, lemmatized_text):
        words = lemmatized_text.lower().split()
        for word in words:
            if word in self.country_set:
                return word
            elif word in self.iso2_country_map:
                return self.iso2_country_map[word]
            elif word in self.iso3_country_map:
                return self.iso3_country_map[word]
            elif word in self.city_country_map:
                return self.city_country_map[word]
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
        patterns = [
            {"label": "GPE", "pattern": "iceland"}
        ]
        ruler.add_patterns(patterns)

    def initialize_spacy_model(self):
        self.download_spacy_model()
        nlp = spacy.load('en_core_web_sm')
        self.setup_entity_ruler(nlp)
        return nlp

    @lru_cache(maxsize=1024)
    def extract_regions(self, tweet):
        if tweet in self.location_cache:
            return self.location_cache[tweet]
        
        doc = self.nlp(tweet)
        regions = [ent.text for ent in doc.ents if ent.label_ == "GPE"]
        result = ", ".join(regions) if regions else None
        self.location_cache[tweet] = result
        return result

    def final_location(self, lemmatized_text):
        mapped_country = self.map_country(lemmatized_text)
        if mapped_country != 'Unknown':
            return mapped_country
        
        location = self.extract_regions(lemmatized_text)
        return location if location else 'Unknown'

    @staticmethod
    @lru_cache(maxsize=1024)
    def categorize_country(country):
        eu_countries = {
            "austria", "belgium", "bulgaria", "croatia", "cyprus", "czech republic",
            "denmark", "estonia", "finland", "france", "germany", "greece",
            "hungary", "ireland", "italy", "latvia", "lithuania", "luxembourg",
            "malta", "netherlands", "poland", "portugal", "romania", "slovakia",
            "slovenia", "spain", "sweden",
            "albania", "andorra", "belarus", "bosnia and herzegovina", "iceland",
            "liechtenstein", "monaco", "montenegro", "north macedonia", "norway",
            "russia", "san marino", "serbia", "switzerland", "ukraine",
            "united kingdom", "vatican city"
        }

        country_lower = country.lower()
        if country_lower in eu_countries:
            return "EU"
        elif country_lower == "united states":
            return "US"
        elif country_lower == "unknown":
            return "Unknown"
        else:
            return "Other"