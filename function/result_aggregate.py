import pandas as pd
import numpy as np

class WeatherDataAggregator:
    def __init__(self, dataframe):
        self.df = dataframe.copy()
        self.result = None

    def preprocess_data(self):
        # 1. Exclude rows with 3 words or less in lemmatized_text
        self.df['word_count'] = self.df['lemmatized_text'].str.split().str.len()
        self.df = self.df[self.df['word_count'] > 3]

        # 2. Create a new column for grouping (weather_topic or LDA top 3 words)
        self.df['group_topic'] = self.df['weather_topics'].fillna(self.df['top_3_words'])

        # 3. Map climate_bert_temp_sentiment to categories
        self.df['temp_category'] = self.df['climate_bert_temp_sentiment'].apply(self.map_sentiment)

    @staticmethod
    def map_sentiment(x):
        if pd.isna(x):
            return 'unknown'
        elif x == 0:
            return 'very cold'
        elif x == 1:
            return 'cold'
        elif x == 2:
            return 'mild/not relevant'
        elif x == 3:
            return 'hot'
        elif x == 4:
            return 'very hot'
        else:
            return 'unknown'

    @staticmethod
    def calculate_percentages(counts):
        total = sum(counts.values())
        return {k: f"{v} ({v/total*100:.1f}%)" for k, v in counts.items()}

    def aggregate_data(self):
        # 4. Group and aggregate
        grouped = self.df.groupby(['group_topic', 'final_location'])

        # 5. Calculate counts and percentages
        self.result = grouped.agg({
            'temp_category': lambda x: x.value_counts().to_dict(),
            'account': ['count', 'nunique']
        }).reset_index()

        # Flatten the multi-level columns
        self.result.columns = ['group_topic', 'final_location', 'temp_counts', 'tweet_count', 'account_count']

        # 6. Calculate percentages
        self.result['temp_percentages'] = self.result['temp_counts'].apply(self.calculate_percentages)

        # 7. Sort by total count in descending order
        self.result['total_count'] = self.result['temp_counts'].apply(lambda x: sum(x.values()))
        self.result = self.result.sort_values('total_count', ascending=False).drop('total_count', axis=1)

        # 8. Reorder columns
        self.result = self.result[['group_topic', 'final_location', 'tweet_count', 'account_count', 'temp_counts', 'temp_percentages']]

    def process(self):
        self.preprocess_data()
        self.aggregate_data()
        return self.result

def aggregate_weather_data(dataframe):
    aggregator = WeatherDataAggregator(dataframe)
    return aggregator.process()

if __name__ == "__main__":
    # Example usage
    # Assuming you have a CSV file named 'weather_data.csv'
    df = pd.read_csv('weather_data.csv')
    result = aggregate_weather_data(df)
    print(result)