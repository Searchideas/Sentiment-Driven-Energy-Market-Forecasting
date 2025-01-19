import pandas as pd
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
import torch
from datasets import Dataset
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

class ClimateBERTFinetuner:
    def __init__(self, data_path, model_name="climatebert/distilroberta-base-climate-f", num_labels=5):
        self.data_path = data_path
        self.model_name = model_name
        self.num_labels = num_labels
        self.df = None
        self.train_df = None
        self.val_df = None
        self.tokenizer = None
        self.model = None
        self.trainer = None

    def load_data(self):
        self.df = pd.read_csv(self.data_path)
        #print(self.df.head())
        self.train_df, self.val_df = train_test_split(self.df, test_size=0.2, random_state=42)

    def load_model_and_tokenizer(self):
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(self.model_name, num_labels=self.num_labels)

    def tokenize_function(self, examples):
        return self.tokenizer(examples["text"], padding="max_length", truncation=True)

    def prepare_datasets(self):
        train_dataset = Dataset.from_pandas(self.train_df)
        val_dataset = Dataset.from_pandas(self.val_df)
        self.tokenized_train = train_dataset.map(self.tokenize_function, batched=True)
        self.tokenized_val = val_dataset.map(self.tokenize_function, batched=True)

    @staticmethod
    def compute_metrics(pred):
        labels = pred.label_ids
        preds = pred.predictions.argmax(-1)
        precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='weighted')
        acc = accuracy_score(labels, preds)
        return {
            'accuracy': acc,
            'f1': f1,
            'precision': precision,
            'recall': recall
        }

    def setup_trainer(self, output_dir="./results", num_train_epochs=3, per_device_train_batch_size=16,
                      per_device_eval_batch_size=64, warmup_steps=500, weight_decay=0.01):
        training_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=num_train_epochs,
            per_device_train_batch_size=per_device_train_batch_size,
            per_device_eval_batch_size=per_device_eval_batch_size,
            warmup_steps=warmup_steps,
            weight_decay=weight_decay,
            logging_dir="./logs",
            logging_steps=10,
            evaluation_strategy="epoch",
            save_strategy="epoch",
            load_best_model_at_end=True,
        )

        self.trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=self.tokenized_train,
            eval_dataset=self.tokenized_val,
            compute_metrics=self.compute_metrics
        )

    def train(self):
        self.trainer.train()

    def save_model(self, path="./fine_tuned_climatebert"):
        self.trainer.save_model(path)
        self.tokenizer.save_pretrained(path)

    def evaluate(self):
        eval_results = self.trainer.evaluate()
        print(eval_results)

    def run_finetuning(self):
        self.load_data()
        self.load_model_and_tokenizer()
        self.prepare_datasets()
        self.setup_trainer()
        self.train()
        self.save_model()
        self.evaluate()


#from climate_bert_finetuner import ClimateBERTFinetuner

#finetuner = ClimateBERTFinetuner('data\synthetic_weather_tweets.csv')
#finetuner.run_finetuning()