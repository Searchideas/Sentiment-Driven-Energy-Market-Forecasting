# Weather Sentiment Analysis from Tweets

## Table of Contents
- [Installation](#installation)
- [Usage](#usage)
- [File Structure](#file-structure)
- [Output](#output)

## Installation
1. Create a virtual environment (optional but recommended):

python -m venv venv
source venv/bin/activate  # On Windows use `venv\Scripts\activate`

Note: the code works with python version: 3.11.5

2. Install required libraries:

pip install -r requirements.txt


## Usage
1. Ensure you have a tweet file in the same format (ws_data.csv) file in the data folder. This file should contain the tweets you want to analyze.

2. Run the analysis by executing the following Python code:

from function.weather_analysis_output import preprocess_and_analyze
result_df, aggregated_result = preprocess_and_analyze()


note1: It may take a couple of minutes to run a 16000 rows of tweets
note2: The inputs have been predefined in the function: preprocess_and_analyze( tweet_dataset_path='data/ws_data.csv',
    output_path_result_df='output/result_df.csv',
    output_path_aggregated='output/aggregated_result.csv'
)

This function takes the following parameters:
- tweet_dataset_path: Path to the input dataset (default: 'data/ws_data.csv')
- output_path_result_df: Path for the output result DataFrame (default: 'output/result_df.csv')
- output_path_aggregated: Path for the output aggregated results (default: 'output/aggregated_result.csv')

The function returns:
- result_df: DataFrame with tweets and predicted temperatures sentiment labels
- aggregated_result: Aggregated analysis results

3. Temperature sentiment labels are integers in the range [0,4]:
- 0: Very cold
- 1: Cold
- 2: Mild/No mention of temperature
- 3: Hot
- 4: Very hot

4. The script requires an output folder in the directory

5. If you will like to fine-tune the climatebert model, you can update the synthetic dataset (synthetic_weather_tweets.csv) in the data folder (follows the same column name) and run the following function:

from function.model_finetune import ClimateBERTFinetuner

finetuner = ClimateBERTFinetuner('data\synthetic_weather_tweets.csv')
finetuner.run_finetuning()
 

It will save the fine-tune model in the "fine_tuned_climatebert" folder

## File Structure
weather-sentiment-analysis/
│
├── data/
│   ├── ws_data.csv (to be added/replaced by user)
│   └── [other mapping files]
│
├── function/
│   └── weather_analysis_output.py
│
├── fine_tuned_climatebert/
│
├── output/
│   ├── result_df.csv (generated after running)
│   └── aggregated_result.csv (generated after running)
│
├── requirements.txt
└── README.md