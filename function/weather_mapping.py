import pandas as pd
import re

class WeatherConditionMapper:
    def __init__(self, file_path):
        """
        Initialize the WeatherConditionMapper with the mapping data from a CSV file.

        Args:
        - file_path (str): The path to the CSV file containing the word mappings.
        """
        self.mapping_dict = self.load_word_mapping(file_path)

    def load_word_mapping(self, file_path):
        """
        Loads the word mapping data from a CSV file, reshapes it,
        and creates a mapping dictionary.

        Args:
        - file_path (str): The path to the CSV file containing the word mappings.

        Returns:
        - dict: A dictionary where keys are words and values are their mapped conditions.
        """
        # Load the CSV into a DataFrame
        mapping_df = pd.read_csv(file_path)
        
        # Reshape the DataFrame using pd.melt
        mapping_df = pd.melt(mapping_df, id_vars=['word', 'need_location'], 
                             value_vars=['EU', 'USA'], var_name='region', value_name='mapping')
        
        # Create and return a dictionary from the word mapping DataFrame
        mapping_dict = dict(zip(mapping_df['word'].str.lower(), mapping_df['mapping']))
        
        return mapping_dict

    def map_weather_conditions(self, tweet):
        """
        Maps weather-related words in a tweet to their corresponding conditions
        based on a word mapping dictionary.

        Args:
        - tweet (str): The input tweet text.

        Returns:
        - dict: A dictionary with matched words and conditions or None if no match is found.
        """
        # Extract words from the tweet
        words = re.findall(r'\w+', tweet.lower())
        
        matched_words = set()      # To store the matched words from the tweet
        matched_conditions = set()  # To store the corresponding conditions
        
        # Loop over the mapping dictionary and check if any key is found in the tweet
        for key in self.mapping_dict.keys():
            if any(key in word for word in words):
                matched_words.add(key)
                matched_conditions.add(self.mapping_dict[key])
        
        # If matches are found, return the results in a dictionary
        if matched_words:
            return {
                'words': ', '.join(matched_words),
                'conditions': ', '.join(matched_conditions)
            }
        else:
            return None

    def apply_mapping_to_df(self, df, text_column):
        """
        Applies the weather condition mapping function to a specific text column in the DataFrame.

        Args:
        - df: The pandas DataFrame containing the tweets.
        - text_column: The name of the column containing the tweets.

        Returns:
        - df: The DataFrame with two new columns: 'weather_words' and 'weather_conditions'.
        """
        # Apply the mapping function to extract weather words and conditions
        mapped_data = df[text_column].apply(lambda x: self.map_weather_conditions(x))
        
        # Create new columns in the DataFrame for the matched words and conditions
        df['weather_topics'] = mapped_data.apply(lambda x: x['words'] if x else None)
        df['weather_temp'] = mapped_data.apply(lambda x: x['conditions'] if x else None)
        
        return df

"""
# Example usage:
# Step 1: Load the mapping dictionary from the CSV
# mapping_dict = load_mapping_file('../data/word_mapping.csv')
# Step 2: Assume you have a DataFrame `df` with a column 'clean_text' containing tweets
# df = pd.DataFrame({
#'clean_text': ["It's going to rain today", "The weather is sunny in the USA", "Windy conditions expected tomorrow"]
# })
# Step 3: Apply the mapping function to extract weather-related words and conditions
# df = apply_mapping_to_df(df, 'clean_text', mapping_dict)
# Step 4: Print the updated DataFrame to see the results
# print(df)"""