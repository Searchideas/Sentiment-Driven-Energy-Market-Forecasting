import pandas as pd
from function.Preprocess import TextPreprocessor
from function.weather_mapping import WeatherConditionMapper
from function.extract_region_optimised import CountryNERMapper
from function.Glossary import GlossaryProcessor
from function.LDA import LDAProcessor
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch
from function.result_aggregate import WeatherDataAggregator

def preprocess_and_analyze(tweet_dataset_path = 'data/ws_data.csv', output_path_result_df = 'output/result_df.csv', output_path_aggregated = 'output/aggregated_result.csv'):
    """
    Main function to preprocess tweets, perform LDA, run the model, and aggregate results.

    Args:
    tweet_dataset_path (str): Path to the CSV file containing the tweet dataset.
    output_path_result_df (str): Path to save the processed dataframe with LDA and model results.
    output_path_aggregated (str): Path to save the aggregated results.

    Returns:
    tuple: (result_df, aggregated_result)
    """
    # Load data
    data = pd.read_csv(tweet_dataset_path)
    print("Data loaded. Shape:", data.shape)

    # Text preprocessing
    text_processor = TextPreprocessor(data, 'text')
    processed_data = text_processor.data
    print("Text preprocessing completed.")

    # Apply weather glossary mapping
    api_endpoint = "https://api.weather.gov/glossary"
    headers = {
        "Accept": "application/ld+json",
        "User-Agent": "MyWeatherApp/1.0"
    }
    glossary_processor = GlossaryProcessor(api_endpoint, headers)
    glossary_processor.fetch_glossary()
    glossary_processor.process_glossary_mapping(processed_data)
    print("Weather Glossary mapping completed.")

    # Weather mapping
    file_path = 'data/word_mapping.csv'
    mapper = WeatherConditionMapper(file_path)
    processed_data = mapper.apply_mapping_to_df(processed_data, 'cleaned_text')
    print("Weather mapping completed.")

    # Region extraction and categorization
    filepath = 'data/filtered_worldcities_us_europe.csv'
    country_ner_mapper = CountryNERMapper(filepath)
    processed_data['final_location'] = processed_data.apply(
        lambda row: country_ner_mapper.final_location(row['lemmatized_text']), axis=1
    )
    processed_data['region'] = processed_data.apply(
        lambda row: country_ner_mapper.categorize_country(row['final_location']), axis=1
    )
    print("Region extraction and categorization completed.")

    # LDA steps
    print("LDA processing start.")
    lda_processor = LDAProcessor(processed_data)
    result_df = lda_processor.process()
    print("LDA processing completed.")

    # Load fine-tuned model for temperature sentiment prediction
    model_path = "./fine_tuned_climatebert"
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    # Function to make predictions
    def predict_temperature(text):
        inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
        outputs = model(**inputs)
        probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
        predicted_class = torch.argmax(probs, dim=1).item()
        return predicted_class

    # Apply temperature sentiment prediction
    print("Temperature sentiment prediction start.")
    result_df['combined_text'] = processed_data['lemmatized_text'] + ' ' + processed_data['weather_temp'].astype(str)
    result_df['climate_bert_temp_sentiment'] = result_df['combined_text'].apply(predict_temperature)
    print("Temperature sentiment prediction completed.")

    # Save the result dataframe
    result_df.to_csv(output_path_result_df, index=False)
    print(f"Result dataframe saved to '{output_path_result_df}'")

    # Aggregate results
    aggregator = WeatherDataAggregator(result_df)
    aggregated_result = aggregator.process()

    # Save the aggregated results
    aggregated_result.to_csv(output_path_aggregated, index=False)
    print(f"Aggregated results saved to '{output_path_aggregated}'")
    print("Aggregated view of tweets:")
    print(aggregated_result)
    

    return result_df, aggregated_result

if __name__ == "__main__":
    tweet_dataset_path = 'data/ws_data.csv'
    output_path_result_df = 'output/result_df.csv'
    output_path_aggregated = 'output/aggregated_result.csv'
    result_df, aggregated_result = preprocess_and_analyze(tweet_dataset_path, output_path_result_df, output_path_aggregated)