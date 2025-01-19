import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from gensim.corpora import Dictionary
from gensim.models import LdaModel

class LDAProcessor:
    # def __init__(self, data, glossary_list):
        # self.data = data
        # self.glossary_list = glossary_list
        # self.processed_data = None
        # self.split_glossary = self.vocab(glossary_list)  # Initialize split_glossary here
    
    # def vocab(self, glossary_list):
        # """ Splits and collects unique words from glossary phrases. """
        # split_glossary = set()
        # for phrase in glossary_list:
            # split_glossary.update(phrase.split())
        # return split_glossary
    
    # def update_vocab_column(self):
        # """ Updates the Vocab column to only include words in split_glossary. """
        # self.processed_data['Vocab'] = self.processed_data['lemmatized_text'].apply(
            # lambda x: [word for word in x.split() if word in self.split_glossary]
        # )
    def __init__(self, data):
        self.data = data
        self.processed_data = None
        self.lda_model = None
        self.dictionary = None
        self.corpus = None

    
    def preprocess_data(self):
        self.processed_data = self.data[
            (self.data['lemmatized_text'].str.split().str.len() > 3) &
            (self.data.loc[:, 'weather_topics'].isnull())
        ]



    # def create_dictionary(self, tokens):
        # """ Create a dictionary from tokenized texts. """
        # return Dictionary(tokens)

    # def create_document_matrix(self, tokens, id2word):
        # """ Create a document-term matrix (corpus) from tokenized texts. """
        # return [id2word.doc2bow(text) for text in tokens]

    # def run_lda(self, num_topics=4, alpha=0.5, beta=0.1, iterations=100, top_n_words=2):
        # """ Run LDA to extract topics from the processed data. """
        # self.preprocess_data()
        # valid_lda_input = self.processed_data['lemmatized_text'].dropna().apply(lambda x: x.split())
        # id2word = self.create_dictionary(valid_lda_input)
        # corpus = self.create_document_matrix(valid_lda_input, id2word)
        
        # lda_model = LdaModel(corpus=corpus, num_topics=num_topics, id2word=id2word, 
                             # alpha=alpha, eta=beta, iterations=iterations)
        # lda_topics_df = self.get_lda_topics(lda_model, num_topics, top_n_words)
        # return lda_topics_df, lda_model, corpus

    def run_lda(self, num_topics=4, alpha=0.5, beta=0.1, iterations=100, top_n_words=3):
        self.preprocess_data()
        texts = self.processed_data['lemmatized_text'].str.split().tolist()
        self.dictionary = Dictionary(texts)
        self.corpus = [self.dictionary.doc2bow(text) for text in texts]
        
        self.lda_model = LdaModel(
            corpus=self.corpus,
            id2word=self.dictionary,
            num_topics=num_topics,
            alpha=alpha,
            eta=beta,
            iterations=iterations
        )
        
        
    def get_top_words(self, num_words=3):
        return [
            [word for word, _ in self.lda_model.show_topic(topic_id, topn=num_words)]
            for topic_id in range(self.lda_model.num_topics)
        ]

    def assign_topics_and_words(self):
        topic_words = self.get_top_words()
        
        def get_dominant_topic(bow):
            topic_probs = self.lda_model.get_document_topics(bow)
            return max(topic_probs, key=lambda x: x[1])[0]
        
        self.processed_data['dominant_topic'] = [get_dominant_topic(bow) for bow in self.corpus]
        self.processed_data['top_3_words'] = self.processed_data['dominant_topic'].apply(lambda x: ', '.join(topic_words[x]))
        
        # Merge results back to the original dataframe
        topic_results = self.processed_data[['dominant_topic', 'top_3_words']]
        self.data = self.data.merge(topic_results, left_index=True, right_index=True, how='left')

    def process(self):
        self.run_lda()
        self.assign_topics_and_words()
        return self.data


'''import pandas as pd
import gensim
from gensim.corpora import Dictionary
from gensim.models import LdaModel

class LDAProcessor:
    def __init__(self, data, glossary_list):
        self.data = data
        self.glossary_list = glossary_list
        self.processed_data = None
    
    def preprocess_data(self):
        # Preprocess data to extract the relevant text for LDA
        self.processed_data = self.data[self.data['lemmatized_text'].notnull() & (self.data['lemmatized_text'] != '')]

    def create_dictionary(self, tokens):
        """ Create a dictionary from tokenized texts. """
        return Dictionary(tokens)

    def create_document_matrix(self, tokens, id2word):
        """ Create a document-term matrix (corpus) from tokenized texts. """
        return [id2word.doc2bow(text) for text in tokens]

    def run_lda(self, num_topics=10, alpha='auto', beta='auto', iterations=10000, top_n_words=3):
        """ Run LDA to extract topics from the processed data. """
        self.preprocess_data()

        # Prepare the data for LDA
        valid_lda_input = self.processed_data['lemmatized_text'].dropna().apply(lambda x: x.split())
        id2word = self.create_dictionary(valid_lda_input)
        corpus = self.create_document_matrix(valid_lda_input, id2word)

        # Build the LDA model
        lda_model = LdaModel(corpus=corpus, num_topics=num_topics, id2word=id2word, 
                             alpha=alpha, eta=beta, iterations=iterations)

        # Get LDA topics
        lda_topics_df = self.get_lda_topics(lda_model, num_topics, top_n_words)

        return lda_topics_df, lda_model, corpus

    def get_lda_topics(self, lda_model, num_topics, top_n_words):
        """ Extract topics from the LDA model. """
        topics = lda_model.show_topics(num_topics=num_topics, num_words=top_n_words, formatted=False)
        topics_words = {f'Topic {i}': [word for word, _ in words] for i, words in topics}
        return pd.DataFrame(dict([(k, pd.Series(v)) for k, v in topics_words.items()]))

    def get_relevance_words_for_row(self, lda_model, corpus_row, id2word, num_words=3):
        """ Get the most relevant words for a given document based on the LDA model. """
        word_relevance = {}
        for word_id, freq in corpus_row:
            word = id2word[word_id]
            word_topics = lda_model.get_term_topics(word_id, minimum_probability=0)
            if word_topics:  # Check if there are topics associated with the word
                avg_relevance = sum(prob for _, prob in word_topics) / len(word_topics)
                word_relevance[word] = avg_relevance * freq

        # Sort words by relevance and get the top N
        sorted_words = sorted(word_relevance.items(), key=lambda x: x[1], reverse=True)
        top_words = [word for word, _ in sorted_words[:num_words]]
        return ' '.join(top_words)

    def add_lda_words_column(self, lda_model, corpus, id2word, num_words=3):
        # Get valid rows where 'lemmatized_text' is not null
        valid_rows = self.processed_data.index[self.processed_data['lemmatized_text'].notna()]

        # Create a new column for storing top words and initialize with pd.NA
        self.processed_data['top_3_relevance_words'] = pd.NA

        # Iterate through valid rows and assign the top words based on relevance
        for i in valid_rows:
            if i < len(corpus):  # Check if index is within range of corpus
                self.processed_data.at[i, 'top_3_relevance_words'] = self.get_relevance_words_for_row(lda_model, corpus[i], id2word, num_words)

        return self.processed_data



    def label_tweets_based_on_topics(self, lda_model, num_words=3):
        """
        Label tweets based on prominent words from LDA topics.
        """
        # Create a dictionary to store labels
        labels = {}
        
        # Get the prominent words for each topic
        topics = lda_model.show_topics(num_topics=lda_model.num_topics, num_words=num_words, formatted=False)
        for i, words in topics:
            labels[f'Topic {i}'] = [word for word, _ in words]

        # Initialize a new column for labels using pd.Series to avoid SettingWithCopyWarning
        self.processed_data['labels'] = pd.Series([pd.NA] * len(self.processed_data), index=self.processed_data.index)

        # Check for each tweet if it contains any prominent word and assign the corresponding label
        for index, row in self.processed_data.iterrows():
            for topic, words in labels.items():
                if any(word in row['cleaned_text'] for word in words):
                    self.processed_data.at[index, 'labels'] = topic
                    break  # Stop at the first matching topic

        return self.processed_data
    
    def finalize_weather_words(labeled_data, lda_topics_df):
        # Create a new column for final weather words
        labeled_data['final_weather_words'] = None

        # Map topic labels to their respective top 3 words
        topic_to_words = {f'Topic {i}': lda_topics_df.iloc[:, i].tolist()[:3] for i in range(lda_topics_df.shape[1])}

        for index, row in labeled_data.iterrows():
            if row['weather_words'] is not None:
                labeled_data.at[index, 'final_weather_words'] = row['weather_conditions']
            elif row['labels'] in topic_to_words:
                # Use the top 3 words for the corresponding topic
                labeled_data.at[index, 'final_weather_words'] = ', '.join(topic_to_words[row['labels']])
            else:
                labeled_data.at[index, 'final_weather_words'] = None  # Default case

        return labeled_data'''

    # def compute_tfidf_by_label(self):
        ## Ensure 'Vocab' exists
        # if 'Vocab' not in self.processed_data.columns:
            # self.processed_data['Vocab'] = self.processed_data['lemmatized_text'].apply(lambda x: x.split())

        ##Now group by 'labels' and aggregate
        # grouped = self.processed_data.groupby('labels', as_index=False).agg({
            # 'Vocab': lambda x: ' '.join([' '.join(v) for v in x])
        # })
        
        # vectorizer = TfidfVectorizer()
        # tfidf_matrix = vectorizer.fit_transform(grouped['Vocab'])
        
        #Return a DataFrame of TF-IDF values
        # return pd.DataFrame(tfidf_matrix.toarray(), columns=vectorizer.get_feature_names_out(), index=grouped['labels'])


    # def update_weather_words(self, tfidf_df):
        # filtered_data = self.processed_data[(self.processed_data['weather_words'].isna()) & 
                                            # (self.processed_data['lemmatized_text'].str.len() > 3)]
        # for index, row in filtered_data.iterrows():
            # label = row['labels']
            # vocab = row['Vocab']
            # weather_words = row['weather_words']
            # if pd.isna(label) or not pd.isna(weather_words):
                # continue
            # tfidf_values = {item: tfidf_df.at[label, item] for item in vocab if item in tfidf_df.columns}
            # top_tfidf = sorted(tfidf_values.items(), key=lambda x: x[1], reverse=True)[:2]
            # if top_tfidf:
                # top_words = [word for word, _ in top_tfidf]
                # filtered_data.at[index, 'weather_words'] = ', '.join(top_words)
        # self.processed_data.update(filtered_data[['weather_words']])
        # return self.processed_data

