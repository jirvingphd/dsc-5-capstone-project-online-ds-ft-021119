
####
class TEXT(object):
    """Class intended for preprocessing text for use with Word2Vec vectors, Kera's Tokenizer, 
    creating FreqDists and an embedding_layer for Kera's models.
    TEXT = TEXT(text_data, text_labels, word2vecmodel)"""
    
    def __init__(self,df, text_data_col,text_labels_col=None,word2vec_model=None,fit_model=False,verbose=2):
        """
         Initializes the class with the text_data (series), text_labels (y data), and a word2vecmodel.
        Performs all processing steps on entire body of text to generate corupus-wide analyses. 
        i.e. FreqDist, word2vec models, fitting tokenzier
        
        - if no word2vec model is provided, one is fit on the text_data using TEXT.fit_word2vec()
        - calls on fit_tokenizer() to fit keras tokenizer to text_sequences.
        text_data and text_labels are saved as TEXT._text_data_ and TEXT._text_labels_

        Args:
            df ([DataFrame]): DataFrame containing text to store and analyze.
            text_data_col (str): Column containing text to process. 
            text_labels_col (str, optional): Column with text classification labels. Defaults to None.
            word2vec_model (Word2Vec Model, optional): Previously fit word2vec model. Defaults to None.
            fit_model (bool, optional): [description]. Fit word2vec model.Defauts to False.
            verbose (int, optional): [description]. Defaults to 0.
        """
        
       
        import numpy as np
        import pandas as pd
        ## Save text_data
        text_data = df[text_data_col].copy()
        self._text_data_ = text_data
        self._verbose_=verbose        
        
        ## Save or create empty text_labels
        if text_labels_col is not None:
            text_labels = df[text_labels_col].copy()
        else:
            text_labels = np.empty(len(text_data))
            text_labels[:] = pd.Series(np.nan)
            
            text_labels_col='text_labels'
            df[text_labels_col] = text_labels
            
        self._text_labels_ = text_labels
        
        
        ## SAVE TO INTERNAL DATAFRAME
        self.df = pd.concat([df[text_data_col],df[text_labels_col]],axis=1)
        self.df.columns=['input_text_data','raw_text_labels']

        ## CALL ON PREPARE_DATA FUNCTIONS

        ## Prepare text_body for corpus-wide operations
        self.prepare_corpus()

        ## Fit word2vec model if not provided
        if word2vec_model is not None:
            self.wv= word2vec_model.wv 
            self._word2vec_model_ = word2vec_model      

        else:
            if self._verbose_ >0:
                print('no word2vec model provided.')

            # if fit_model:
            #     self.fit_word2vec(text_data=self._text_data_)#self.tokenized_text_body)
            else:
                self.wv = None
                self._word2vec_model_ = None
        
        if fit_model:
            self.fit_models()


    def fit_models(self):
        
        ## Fit Word2Vec
        if self.wv is None:
            self.fit_word2vec(text_data=self._text_data_)#self.tokenized_text_body)

        ## Fit keras tokenizer
        self.fit_tokenizer()
        
        ## Get FreqDist
        self.fit_FreqDist()

        ## Create Embedding Layer
        # self.get_embedding_layer(return_layer=False)
    

        
    def prepare_corpus(self,text_data_to_prep=None,delim=','):
        """Joins, regexp_tokenizes text_data"""
        #         text_data_to_prep=[]

        import numpy as np
        import pandas as pd

        if text_data_to_prep is None:
            text_data_to_prep = self._text_data_
            delim.join(text_data_to_prep)


        if isinstance(text_data_to_prep,list) | isinstance(text_data_to_prep, np.ndarray) \
        | isinstance(text_data_to_prep,pd.Series):
            # print('prepare_text: type=',type(text_data_to_prep))
            # print(text_data_to_prep)
            text_joined = delim.join(text_data_to_prep)
            # text_data_to_prep =  delim.join([str(x) for x in text_data_to_prep])
        else:
            text_joined = text_data_to_prep
            # print('prepare_text: text not list, array, or series')

        self._text_body_for_processing_ = text_joined#text_data_to_prep

        tokenized_text =  self.regexp_tokenize(text_joined)
        self.tokenized_text_body = tokenized_text

        if self._verbose_>0:
            print('Text processed and saved as TEXT.tokenized_text')       


    def fit_word2vec(self,text_data=None,vector_size=300, window=5, min_count=2, workers=3, epochs=10):
        """Fits a word2vec model on text_data and saves the model.wv object as TEXT.wv and the full model
        as ._word2vec_model_"""
        from gensim.models import Word2Vec
        import numpy as np
        import pandas as pd

        if text_data is None:
            text_data = self.tokenized_text_body
        elif isinstance(text_data, np.ndarray):
            text_data = pd.Series(text_data)
        
        elif isinstance(text_data,pd.Series):
            text_data = text_data.apply(lambda x: self.regexp_tokenize(x))
        else:
            if self._verbose_ >0:
                print('Using raw text_data to fit_word2vec')

        # text_data = ' '.join(text_data)
        wv_keras = Word2Vec(text_data, size=vector_size, window=window, min_count=min_count, workers=workers)
        wv_keras.train(text_data,total_examples=wv_keras.corpus_count, epochs=epochs)
                       
        self._word2vec_model_ =  wv_keras                       
        self.wv =  wv_keras.wv
#         vocab_size = len(wv_keras.wv.vocab)
        print(f'There are {len(self.wv.vocab)} words in the word2vec vocabulary (TEXT.wv), with a vector size {vector_size}.')


    def make_stopwords_list(self, incl_punc=True, incl_nums=True, 
                            add_custom= ['http','https','...','…','``','co','“','’','‘','”',
                                         "n't","''",'u','s',"'s",'|','\\|','amp',"i'm"]):
        from nltk.corpus import stopwords
        import string

        stopwords_list = stopwords.words('english')
        if incl_punc==True:
            stopwords_list += list(string.punctuation)
        stopwords_list += add_custom #['http','https','...','…','``','co','“','’','‘','”',"n't","''",'u','s',"'s",'|','\\|','amp',"i'm"]
        if incl_nums==True:
            stopwords_list += [0,1,2,3,4,5,6,7,8,9]
            
        self._stopwords_list_ = stopwords_list

        return  stopwords_list


    def remove_stopwords(self, text_data, stopwords_list=None, tokenize=True,
                        pattern = "([a-zA-Z]+(?:'[a-z]+)?)",return_tokens=False):
        """EX: df['text_stopped'] = df['content'].apply(lambda x: remove_stopwords(stopwords_list,x))"""
        print('return to .remove_stopwords and verify saved vs unused variables')
        from nltk import regexp_tokenize

        if stopwords_list is None:
            stopwords_list = self.make_stopwords_list()
            
        if text_data is None:
            text_data = self._text_data_
            
        if tokenize==True:
            tokenized_text = self.regexp_tokenize(text_data)
            text_data = tokenized_text


        stopped_text = [x.lower() for x in text_data if x.lower() not in stopwords_list]

        if return_tokens==True:
            return regexp_tokenize(' '.join(stopped_text),pattern)
        else:
            return ' '.join(stopped_text)
        
        
        

    
    def fit_tokenizer(self,text_data=None,labels=None):
        """Fits Keras Tokenizer using tokenizer.fit_on_texts to create TEXT.X_sequences and 
        also saves the labels as TEXT.y
        Tokenizer is saved as TEXT._tokenizer_. 
        Word Index (tokenizer.index_word)  is saved as TEXT.word_index
        Reverse dictionary is saved as TEXT.reverse_index"""
        if text_data is None:
            text_data = self.tokenized_text_body #_text_data_
        if labels is None:
            text_labels = self._text_labels_
            
        from keras.preprocessing.text import Tokenizer                   
        tokenizer = Tokenizer(num_words=len(self.wv.vocab))
        tokenizer.fit_on_texts(list(text_data) )#tokenizer.fit_on_texts(text_data)

        self._tokenizer_ = tokenizer
        self.word_index = tokenizer.index_word
        self.reverse_index =  {v:k for k,v in self.word_index.items()}
        
        ## GET SEQUENCES FROM TOKENIZER
        X_sequences = self.text_to_sequences(text_data = text_data)
        self.X_sequences  = X_sequences
        
        y = text_labels
        self.y = y
        if self._verbose_ >0:
            print('tokenizer fit and TEXT.X_sequences, TEXT.y created')

    # def text_to_sequences(self, text_data, save_to_model=False, regexp_tokenize=False):
    #     X_seq = self._text_to_sequences_(self, text_data=text_data,save_to_model=save_to_model,regexp_tokenize=regexp_tokenize)
    #     return X_seq 
    # def text_to_sequences(self,text_data, save_to_model=False, regexp_tokenize=False):
    #     """Calls on internval _text_to_sequences_ to return use the fit self._tokenizer 
    #     to make sequences via self._tokenizer_.texts_to_sequences()"""
    #     _text_to_sequences_ = self._text_to_sequences_
    #     X_seq = self._text_to_sequences_(self, text_data= text_data, save_to_model=False, regexp_tokenize=False)
    #     return X_seq

    def text_to_sequences(self, text_data = None, save_to_model=True, regexp_tokenize=False):        
        """Uses fit _tokenzier_ to create X_sequences
        from tokenzier.texts_to_sequences"""
        import numpy as np
        import pandas as pd

        if text_data is None:
            text_data = self.tokenized_text_body #_text_data_
            
        elif regexp_tokenize:
                if isinstance(text_data,pd.Series) | isinstance(text_data, pd.DataFrame):
                    text_data = text_data.apply(lambda x: self.regexp_tokenize(x))
                else:
                    text_data = self.regexp_tokenize(text_data)
            
        tokenizer = self._tokenizer_
        
        from keras.preprocessing import text, sequence
        X = tokenizer.texts_to_sequences(text_data)
        X_sequences = sequence.pad_sequences(X)

        if save_to_model:
            if self._verbose_ >0:
                print("saving to self.X_sequences()")
            self.X_sequences = X_sequences

        else:
            if self._verbose_ >0:
                print("X_sequences returned, not save_to_model.")

        return X_sequences

    def sequences_to_text(self,sequences=None):
        """Return generated sequences back to original text"""
        if sequences is None:
            sequences = self.X_sequences
        
        tokenizer = self._tokenizer_
        text_from_seq = tokenizer.sequences_to_texts(sequences)
        return text_from_seq
    
    def regexp_tokenize(self,text_data,pattern="([a-zA-Z]+(?:'[a-z]+)?)"):
        """Apply nltk's regex_tokenizer using pattern"""
        # if text_data is None:
        #     text_data = self._text_body_for_processing_ # self._text_data_

        if pattern is None:
            pattern = "([a-zA-Z]+(?:'[a-z]+)?)"
        self._pattern_ = pattern

        # CREATING TEXT DICT FOR FREQUENCY DISTRIBUTIONS
        from nltk import regexp_tokenize
        try:
            tokenized_data = regexp_tokenize(text_data, pattern)
        except TypeError:
            print('TypeError: text_data is already tokenized.')
            tokenized_data = text_data
            # print('Error using regexp_tokenize')
            # print(f"Data Type:\t{type(text_data)}")
            # print(text_data[:100])
#         self.tokens = tokenized_data
            # return None

        return tokenized_data

    def fit_FreqDist(self, clean_text=True, plot=True):
        """ Fits nltk's FreqDist on tokenized text and saved as .freq_dist"""
        
        # if clean_text:
            
        from nltk import FreqDist
        freq_dist = FreqDist(self.tokenized_text_body)
        self.FreqDist = freq_dist
        
        if plot==True:
            self.plot_FreqDist()
        
    def plot_FreqDist(self, top_n_words=25):
        """Create FreqDist plot of top_n_words"""
        import matplotlib.pyplot as plt
        try:
            self.FreqDist
        except:
            self.fit_FreqDist()

        with plt.style.context('seaborn-notebook'):
            self.FreqDist.plot(top_n_words)
        
    
    # def summary_report(self):
    #     """Print summary info about word2vec vocab, vectors, tokenized_text and embedding matrix"""
    #     print(f"Word2Vec Vocabulary Size = {len(self.wv.vocab)}")
    #     print(f"Word2Vec vector size = {self.wv.vector_size}")
    #     print(f"\nLength of tokenized_text = {len(self.tokenized_text_body)}")
    #     print(f"_embedding_matrix_ size = {self._embedding_matrix_.shape}")
        
        
    def prepare_text_sequences(self, process_as_tweets=False, tweet_final_col='cleaned_tweets'):
        """
        Individually process each entry in text_data for stopword removal and tokenization:
        stopwords_list = TEXT.make_stopwords_list()
        
        Args:
            process_as_tweets (bool, optional): use . Defaults to True.
            tweet_final_col (str, optional): [description]. Defaults to 'cleaned_tweets'.
        """
        
        

        self.make_stopwords_list()

        # if process_as_tweets:
        #     self.tweet_specific_processing(tweet_final_col='cleaned_tweets')
        #     text_to_process = self.df[tweet_final_col]
        #     colname_base = tweet_final_col
        # else:
        text_to_process = self.df['input_text_data']
        colname_base = 'text'
        
        # Get stopped, non-tokenzied text
        proc_text_series = text_to_process.apply(lambda x: self.remove_stopwords(x,
                                                                                stopwords_list=None,
                                                                                tokenize=True,
                                                                                return_tokens=False))
        self.df[colname_base+'_stopped'] = proc_text_series
        
        # Get stopped-tokenized text
        proc_text_tokens = text_to_process.apply(lambda x: self.remove_stopwords(x,
                                                                        stopwords_list=None,
                                                                        tokenize=True,
                                                                        return_tokens=True))
        self.df[colname_base+'_stopped_tokens'] = proc_text_tokens

        
    # def tweet_specific_processing(self,tweet_final_col='cleaned_tweets'):
    #     import re
    #     # Get initiial df
    #     df = self.df
        
    #     raw_tweet_col = 'input_text_data'
    #     fill_content_col = tweet_final_col

    #     ## PROCESS RETWEETS
    #     df['has_RT']=df[raw_tweet_col].str.contains('RT')
    #     df['starts_RT']=df[raw_tweet_col].str.contains('^RT')
        
    #     re_RT = re.compile('RT [@]?\w*:')

    #     df['content_starts_RT'] = df[raw_tweet_col].apply(lambda x: re_RT.findall(x))
    #     df[fill_content_col] =  df[raw_tweet_col].apply(lambda x: re_RT.sub(' ',x))
        
        
    #     ## PROCESS URLS
    #     urls = re.compile(r"(http[s]?://\w*\.\w*/+\w+)")
        
    #     check_content_col = fill_content_col
    #     df['content_urls'] = df[check_content_col].apply(lambda x: urls.findall(x))
    #     df[fill_content_col] =  df[check_content_col].apply(lambda x: urls.sub(' ',x))

    #     ## PROCESS HASHTAGS
    #     hashtags = re.compile(r'\#\w*')

    #     df['content_hashtags'] =  df[check_content_col].apply(lambda x: hashtags.findall(x))
    #     df[fill_content_col] =  df[check_content_col].apply(lambda x: hashtags.sub(' ',x))
        
    #     ## PROCESS MENTIONS
    #     # Remove and save mentions (@)'s
    #     mentions = re.compile(r'\@\w*')
    #     df['content_mentions'] =  df[check_content_col].apply(lambda x: mentions.findall(x))
    #     df[fill_content_col] =  df[check_content_col].apply(lambda x: mentions.sub(' ',x))

        
    #     self.df = df 
    #     def empty_lists_to_strings(x):
    #         """Takes a series and replaces any empty lists with an empty string instead."""
    #         if len(x)==0:
    #             return ' '
    #         else:
    #             return ' '.join(x) #' '.join(tokens)
        


########################################################################################    

def replace_embedding_layer(nlp_keras_model, word2vec_model,input_text_series, verbose=2):
    """Takes the original Keras model with embedding_layer, a series of new text, a TEXT object,
    and replaces the embedding_layer (layer[0]) with a new embedding layer with the correct size for new text"""
    ## CONVERT MODEL TO JSON, REPLACE OLD INPUT LAYER WITH NEW ONE FROM TEXT object
    json_model = nlp_keras_model.to_json()

    import functions_combined_BEST as ji
    # pprint(json_model)
    import json
    json_model = json.loads(json_model)
    # ji.display_dict_dropdown(json_model)

    if verbose>0:## Find the exact parameters for shape size that need to change
        print('---'*10,'\n',"json_model['config']['layers'][0]:")
        print('\t','batch_input_shape: ',json_model['config']['layers'][0]['config']['batch_input_shape'])
        print('\t','input_dim: ',json_model['config']['layers'][0]['config']['input_dim'])
        print('\t','input_length:',json_model['config']['layers'][0]['config']['input_length'])


    # Save layer 0 as separate variable to edit, and then replace in the dict
    layer_0 = json_model['config']['layers'][0]
    if verbose>0:
        ji.display_dict_dropdown(layer_0)
        
        
    ## FOR SEQUENCES FROM EXTERNAL NEW TEXT (tweets):
    tokenizer, X_seq = get_tokenizer_and_text_sequences(word2vec_model, input_text_series) #self.text_to_sequences(text_data = input_text_series,regexp_tokenize=True)

    if verbose>0:
        ## To get Text back from X_seq:
        # text_from_seq = TEXT.sequences_to_text(X_seq)
        print('(num_rows_in_df, num_words_in_vocab)')
        print(X_seq.shape)
        
    ## Get new embedding layer's config  (that is fit to new text)
    output = make_keras_embedding_layer(word2vec_model, X_sequences=X_seq)#input_size=X_seq.shape[1])
    new_emb_config = output.get_config()
    
    ## Copy original model
    new_json_model = json_model
    
    ## Replace old layer 0  config with new_emb_config
    new_json_model['config']['layers'][0]['config'] = new_emb_config
    
    # convert model to string (json.dumps) so can use model_from_json
    string_model = json.dumps(json_model)    
    
    ## Make model from json to return 
    from keras.models import model_from_json
    new_model = model_from_json(string_model)
    
    return new_model

## NEW FUNCTIONS FOR WORD2VEC AND KERAS TOKENIZATION/SEQUENCE GENERATION
def make_word2vec_model(df, text_column='content_min_clean', regex_pattern ="([a-zA-Z]+(?:'[a-z]+)?)",verbose=0,
                       vector_size=300,window=3,min_count=1, workers=3,epochs=10,summary=True, return_full=False,**kwargs):
    """    w2v_params = {'sg':1, #skip-gram=1
                'hs':0, #1=heirarchyical softmarx, if 0, and 'negative' is non-zero, use negative sampling
                'negative': 5, # number of 'noisy" words to remove by negative sampling
                'ns_exponent': 0.75, # value between -1 to 1. 0.0 samples all words equaly, 
                1.0 samples proportional to frequency, negative value=lowfrequency words sampled more
    }"""
    

    ## Regexp_tokenize text_column
    from nltk import regexp_tokenize
    text_data = df[text_column].apply(lambda x: regexp_tokenize(x, regex_pattern))

    ## Instantiate Word2Vec Model
    from gensim.models import Word2Vec
    vector_size = 300

    w2v_params = {'sg':1, #skip-gram=1
    'hs':0, #1=heirarchyical softmarx, if 0, and 'negative' is non-zero, use negative sampling
    'negative': 5, # number of 'noisy" words to remove by negative sampling
    'ns_exponent': 0.75, # value between -1 to 1. 0.0 samples all words equaly, 1.0 samples proportional to frequency, negative value=lowfrequency words sampled more
    }

    for k,v in kwargs.items():
        if k in w2v_params:
            w2v_params[k] = v

    if verbose>0:
        # print(f'[i] Using these params for Word2Vec model trained:')
        params_to_print={}
        params_to_print['min_count'] = min_count
        params_to_print['window'] = window
        params_to_print['epochs'] = epochs

        for k,v in w2v_params.items():
            params_to_print[k] = v

        print(f'[i] Training Word2Vec model using:\n\t{params_to_print}')
        # print(w2v_params)

    wv_keras = Word2Vec(text_data, size=vector_size, window=window, min_count=min_count, sg=w2v_params['sg'],
     hs=w2v_params['hs'], negative=w2v_params['negative'], ns_exponent=w2v_params['ns_exponent'], workers=workers)
    
    # Train Word2Vec Model
    wv_keras.train(text_data,total_examples=wv_keras.corpus_count, epochs=epochs)

    
    ## Display summary
    if summary:
        wv = wv_keras.wv
        vocab_size = len(wv_keras.wv.vocab)
        print(f'\t[i] Training complete. model vocab has {vocab_size} words, with vector size {vector_size}.')
        if return_full:
            ans = 'full model'
        else:
            ans = 'model.wv'
        
        print(f'\t[o] returned model is {ans}.')
        
    if return_full:
        return wv_keras
    else:
        return wv_keras.wv
    
def get_wv_from_word2vec(word2vec_model):
    """Checks if model is full word2vec or .wv attribute. 
    Returns wv."""
    import gensim 
    if isinstance(word2vec_model,gensim.models.word2vec.Word2Vec):
        wv = word2vec_model.wv
    elif isinstance(word2vec_model,gensim.models.keyedvectors.Word2VecKeyedVectors):
        wv = word2vec_model
    return wv



def make_embedding_matrix(word2vec_model,verbose=1):#,X_sequences = None,input_size=None):#, return_matrix=False, return_layer=True):
        """Uses the word2vec model to construct an embedding_layer for Keras models.
        To override the default size of the input for the embedding layer, provide an input_size value
        (which will likely be the size of hte vocabulary being fed in for predictions)."""
        import numpy as np
        import pandas as pd     
        
        wv = get_wv_from_word2vec(word2vec_model)
        vocab_size = len(wv.vocab)
        vector_size = wv.vector_size
            
        ## Create the embedding matrix from the vectors in wv model 
        embedding_matrix = np.zeros((vocab_size + 1, vector_size))
        for i, vec in enumerate(wv.vectors):
            embedding_matrix[i] = vec
            embedding_matrix.shape
            
        if verbose:
            print(f'embedding_matrix.shape = {embedding_matrix.shape}')
        
        return embedding_matrix

def make_keras_embedding_layer(word2vec_model,X_sequences,embedding_matrix= None,verbose=1):
        """Creates an embedding layer for Kera's neural networks using the 
        embedding matrix and text X_sequences
        embedding_layer =layers.Embedding(vocab_size+1,
                                    vector_size,
                                    input_length=X_sequences.shape[1],
                                    weights=[embedding_matrix],
                                    trainable=False)"""
        if embedding_matrix is None:
            # embedding_matrix = make_embedding_matrix(word2vec_model,verbose=0)
            import numpy as np
            import pandas as pd     
            
            wv = get_wv_from_word2vec(word2vec_model)
            vocab_size = len(wv.vocab)
            vector_size = wv.vector_size
                
            ## Create the embedding matrix from the vectors in wv model 
            embedding_matrix = np.zeros((vocab_size + 1, vector_size))
            for i, vec in enumerate(wv.vectors):
                embedding_matrix[i] = vec
                embedding_matrix.shape
                
            if verbose:
                print(f'embedding_matrix.shape = {embedding_matrix.shape}')
        
        wv = get_wv_from_word2vec(word2vec_model)
        vocab_size = len(wv.vocab)
        vector_size = wv.vector_size
                
        from keras import layers         
        embedding_layer =layers.Embedding(vocab_size+1,
                                          vector_size,
                                          input_length=X_sequences.shape[1],
                                          weights=[embedding_matrix],
                                          trainable=False)
        return embedding_layer

    
def get_tokenizer_and_text_sequences(word2vec_model,text_data):    
    # sentences_train =text_data # df_tokenize['tokens'].values
    from keras.preprocessing.text import Tokenizer
    wv = get_wv_from_word2vec(word2vec_model)

    tokenizer = Tokenizer(num_words=len(wv.vocab))
    
    ## FIGURE OUT WHICH VERSION TO USE WITH SERIES:
    tokenizer.fit_on_texts(text_data)
#     tokenizer.fit_on_texts(list(text_data)) 

    word_index = tokenizer.index_word
    reverse_index = {v:k for k,v in word_index.items()}
    
    # return integer-encoded sentences
    from keras.preprocessing import text, sequence
    X = tokenizer.texts_to_sequences(text_data)
    X = sequence.pad_sequences(X)
    return tokenizer, X


########################################################################################    
    

def search_for_tweets_with_word(twitter_df,word, from_column='content_min_clean',display_n=50, ascending=False,
                                return_index=False, display_df=True,for_interactive=False, display_cols = [
                                    'retweet_count','favorite_count','source',
                                    'compound_score','sentiment_class']):
    """Searches the df's `from_column` for the specified `word`.
    - if display_df: Displays first `n` rows of styled dataframe with, columns=`display_cols`.
        - display the most-recent or oldest tweets using `ascending` parameter.
    - if return_index: return the datetimeindex of the tweets containing the word."""
    import pandas as pd
    import functions_combined_BEST as ji
    from IPython.display import display
    import numpy as np
    n=display_n

    ## Make list of cols starting with from_column and adding display_cols
    select_cols = [from_column]
    [select_cols.append(x) for x in display_cols]
    
    # Create new df copy with select_cols
    df = twitter_df[select_cols].copy()
    
    ## Check from_column for word.lower() in text.lower()
    check_word = df[from_column].apply(lambda x: True if word.lower() in x.lower() else False)
    # Tally number of tweets containing word
    found_words = np.sum([1 for x in check_word if x ==True])
    
    ## Get rows with the searched word
    res_df_ = df.loc[check_word]
    
    # Save datetime index to output before changing
    output_index = res_df_.index.to_series()
    
    ## Sort res_df_ by datetime index, before resetting index
    res_df_.sort_index(inplace=True, ascending=ascending)
    res_df_.reset_index(inplace=True)
    
    
    # Take n # of rows from the top of the dataframe

    
    ## Set table_style for display df.styler
    table_style =[{'selector':'caption',
                'props':[('font-size','1.3em'),
                         ('color','darkblue'),
                         ('font-weight','semibold'),
                        ('text-align','left')]},
                 {'selector':'th',
                  'props':[('font-size','1.1em'),('text-align','center')]}]
#                  {'selector':'td','props':[('pad','0.1em')]}]
    


    if display_n is None:
        n=res_df_.shape[0]-1
        
    res_df = res_df_.iloc[:n,:]
    # full_index = res_df_.index
    
    ## Create styled df with caption, table_style, hidden index, and text columns formatted
    if for_interactive==False:
        df_to_show = res_df
    else:
        df_to_show = res_df_

    ## Caption for df
    capt_text = f'Tweets Containing "{word}" ({display_n} of {found_words})'
    
    dfs = df_to_show.style.hide_index().\
    set_caption(capt_text).set_properties(subset=[from_column],
                                          **{'width':'400px',
                                            'text-align':'center',
                                            'padding':'2em',
                                            'font-size':'1.2em'}).set_table_styles(table_style)
    ## Display dataframe if display_df
    if for_interactive:
        return dfs
    if display_df:
        display(dfs)
        remaining_tweets = found_words - n
        next_index = res_df_['date'].iloc[n+1]

        print(f'\t * there are {remaining_tweets} tweets not shown. Next index = {next_index}')
    
    ## Return datetimeindex of all found tweets with the word
    if return_index==True:
        return output_index

    



def compare_freq_dists(text1,label1,text2,label2,top_n=20,figsize=(12,6),style='seaborn-poster', display_df=False):
    from nltk import FreqDist
    import pandas as pd
    from IPython.display import display
    freq_1 = FreqDist(text1)
    freq_2 = FreqDist(text2)

    df_compare=pd.DataFrame()
    df_compare[label1] = freq_1.most_common(top_n)
    df_compare[label2] = freq_2.most_common(top_n)
    if display_df:
        display(df_compare)

    ## Plot dists
    import matplotlib.pyplot as plt
    import matplotlib as mpl 

    with plt.style.context(style):
        mpl.rcParams['figure.figsize']=(12,6)
        plt.title(f'{top_n} Most Frequent Words - {label1}')
        freq_1.plot(25)
        plt.title(f'{top_n} Most Frequent Words - {label2}')
        freq_2.plot(25)

def compare_freq_dists_unique_words(text1,label1,text2, label2,top_n=20, display_dfs=True, return_as_dicts=False):
    from nltk import FreqDist
    import bs_ds_local as bs
    import pandas as pd
    text1_dist = FreqDist(text1)#twitter_df_groups['text1']['text_tokens'])
    text1_words = list(text1_dist.keys())
    
    text2_dist = FreqDist(text2)#twitter_df_groups['text2']['text_tokens'])
    text2_words = list(text2_dist.keys())

    text2_not_text1 = {k:v for k,v in text2_dist.items() if k not in text1_dist.keys()}
    text1_not_text2 = {k:v for k,v in text1_dist.items() if k not in text2_dist.keys()}

    df_text2_words = pd.DataFrame.from_dict(text2_not_text1,orient='index',columns=['Frequency'])
    df_text1_words = pd.DataFrame.from_dict(text1_not_text2,orient='index',columns=['Frequency'])

    dfs = ['df_text2_words','df_text1_words']
    for df in dfs:
        df_ = eval(df)
        df_.sort_values('Frequency',ascending=False,inplace=True)
        name= df.split('_')[1].title()

        df_.index.name=f'Unique {name} Words'
        df_.reset_index(inplace=True)

    if display_dfs:
        dfs_1 = df_text1_words.head(top_n).style.hide_index().set_caption(label1)
        dfs_2 = df_text2_words.head(top_n).style.hide_index().set_caption(label2)
        bs.display_side_by_side(dfs_1,dfs_2)

    if return_as_dicts == False:
        return  df_text1_words, df_text2_words
    else:
        return  text1_not_text2, text2_not_text1




def compare_word_clouds(text1,label1,text2,label2,suptitle_text='', wordcloud_cfg_dict=None, twitter_shaped=True,from_freq_dicts=False,
fig_size = (18,18), subplot_titles_y_loc = 0, suptitle_y_loc=0.8, save_file=False,base_folder='figures/',png_name_from_title=False, png_filename=None,verbose=1,**kwargs):
    """Compares the wordclouds from 2 sets of texts. 
    text1,text2:
        If `from_freq_dicts`=False, texts must be non-tokenized form bodies of text.
        If `from_freq_dicts`=True, texts must be a frequency dictionary with keys=words, value=count
    
    label1,label2:
        string names/labels for the resulting wordcloud subplot titles
    
    cfg_dict:
        a dictionary with altnerative parameters for creation of WordClouds
        defaults are :
                {'max_font_size':100, 'width':400, 'height':400,
                'max_words':150, 'background_color':'white', 
                'cloud_stopwords':make_stopwords_list(),
                'collocations':False, 'contour_color':'cornflowerblue', 'contour_width':2}

    twitter_shaped:
        if true, local images of twitter logo will be used as a mask for the shape of the generated wordclouds. 
        if false, wordclouds will be rectangular (shape=specified 'width' and 'height' config keys)

    save_file:
        if True, saves .png to png_filename using suptitle as name (if given), else filename='wordcloud_figure.png'

    **kwargs:

        valid keywords: 
        - 'subplot_titles_fontdict':{any_matplotlib_text_kwds:values} # for ax.set_title() # passed as fontdict=fontdict
        - 'suptitle_fontdict':{any_matplotlib_text_kwds:values} # fontdict for plt.suptitle() # passed as **fontdict 
        - 'imshow':{'interpolation': #options are ['none', 'nearest', 'bilinear', 'bicubic', 'spline16',
                    'spline36', 'hanning', 'hamming', 'hermite', 'kaiser', 'quadric', 'catrom',
                     'gaussian', 'bessel', 'mitchell', 'sinc', 'lanczos'] }
        - 'group colors':('green','red)

        """
    import functions_combined_BEST as ji
    from wordcloud import WordCloud
    import matplotlib.pyplot as plt
    from PIL import Image
    import numpy as np
    from pprint import pprint as pp
    ## Fill in params dictionary with defaults
    # params= {}
    # params['subplot_titles_fontdict'] = {'fontsize':30}
    # params['suptitle_fontdict'] = {'fontsize':40}
    # params['imshow']= {'interpolation':'gaussian'}

    # ## Check for kwargs replacements to defaults
    # to_do = ['subplot_titles_fontdict','suptitle_fontdict','imshow']
    # for td in to_do:
    #     if td in kwargs and kwargs[td] is not None:
    #         params[td] = kwargs[td]
    
    ## Fill in params dictionary with defaults
    params= {}
    params['subplot_titles_fontdict'] = {'fontsize':30}
    params['suptitle_fontdict'] = {'fontsize':40}
    params['imshow']= {'interpolation':'gaussian'}
    params['group_colors']={'group1':'cornflowerblue','group2':'cornflowerblue'}

    ## Check for kwargs replacements to defaults
    to_do = ['subplot_titles_fontdict','suptitle_fontdict','imshow','group_colors']
    for td in to_do:
        if td in kwargs.keys() and kwargs[td] is not None:
            params[td] = kwargs[td]

    
    ## Handle word2vec_doict
    default_cfg={
                'max_font_size':100, 'width':400, 'height':400,
                'max_words':150, 'background_color':'white', 
                'cloud_stopwords':make_stopwords_list(),
                'collocations':False, 
                'contour_color':'cornflowerblue',
                'contour_width':2
                }
    cfg=default_cfg

    ## Use default config if none provided
    if wordcloud_cfg_dict is None:
        cfg = default_cfg
        if verbose==1: 
            print('Using default cfg. (use verbose=2 for details).')
        elif verbose>1:
            print('cfg kwargs:')
            pp(cfg)
        

    else:
        to_do = list(default_cfg.keys()) #['subplot_titles_fontdict','suptitle_fontdict','imshow']
        for td in to_do:
            if td in wordcloud_cfg_dict.keys() and wordcloud_cfg_dict[td] is not None:
                cfg[td] = wordcloud_cfg_dict[td]

        if verbose>1:
            print('cfg kwargs:')
            pp(cfg)

    ## Set different colors if group_colors specified
    if 'group_colors' in kwargs.keys():
        contour_color_1 = kwargs['group_colors']['group1']
        contour_color_2 = kwargs['group_colors']['group2']

    else: #otherwise use  word2vec cfg 
        contour_color_1 = cfg['contour_color']
        contour_color_2 = cfg['contour_color']

    # instantiate the two word clouds using cfg dictionary parametrs
    wordcloud1 = WordCloud(max_font_size = cfg['max_font_size'], width=cfg['width'], height=cfg['height'], max_words=cfg['max_words'],
    background_color=cfg['background_color'], stopwords=cfg['cloud_stopwords'],collocations=cfg['collocations'],
    contour_color=contour_color_1, contour_width=cfg['contour_width'])


    wordcloud2 = WordCloud(max_font_size = cfg['max_font_size'], width=cfg['width'], height=cfg['height'], max_words=cfg['max_words'],
    background_color=cfg['background_color'], stopwords=cfg['cloud_stopwords'],collocations=cfg['collocations'], 
    contour_color=contour_color_2, contour_width=cfg['contour_width'])

    ## Add .mask attribute to wordclouds if twitter_shaped==True
    if twitter_shaped ==True:
        ## Twitter Bird masks
        mask_f_right = np.array(Image.open('figures/masks/twitter1.png'))
        mask_f_left = np.array(Image.open('figures/masks/twitter1flip.png'))
        
        ## Assign the images to text1 and text2
        mask1=mask_f_right
        mask2=mask_f_left

        # Hashtag and mentions mask 
        # mask_at = np.array(Image.open('figures/masks/Hashtags and Ats Masks-04.jpg'))
        # mask_hashtag = np.array(Image.open('figures/masks/Hashtags and Ats Masks-03.jpg'))

        wordcloud1.mask=mask1
        wordcloud2.mask=mask2


    ## Fit wordclouds to text
    if from_freq_dicts==False:
        wordcloud1.generate(text1)                                          
        wordcloud2.generate(text2)
    
    elif from_freq_dicts==True:
        wordcloud1.generate_from_frequencies(text1)
        wordcloud2.generate_from_frequencies(text2)


    ## PLOTTING THE WORDCLOUDS

    # ## Fill in params dictionary with defaults
    # params= {}
    # params['subplot_titles_fontdict'] = {'fontsize':30}
    # params['suptitle_fontdict'] = {'fontsize':40}
    # params['imshow']= {'interpolation':'gaussian'}
    # params['group_colors':('green','red')]

    # ## Check for kwargs replacements to defaults
    # to_do = ['subplot_titles_fontdict','suptitle_fontdict','imshow','group_colors']
    # for td in to_do:
    #     if td in kwargs and kwargs[td] is not None:
    #         params[td] = kwargs[td]

    
    ## CREATE SUBPLOTS
    import matplotlib.pyplot as plt
    fig,ax = plt.subplots(nrows=1,ncols=2,figsize=fig_size)
    fig.suptitle(suptitle_text,y=suptitle_y_loc,**params['suptitle_fontdict'])

    ## Left Subplot  
    ax[0].imshow(wordcloud1, interpolation=params['imshow']['interpolation'])
    ax[0].axis("off")
    ax[0].set_title(label1, y=subplot_titles_y_loc,fontdict=params['subplot_titles_fontdict'] )

    ## Right Subplot
    ax[1].imshow(wordcloud2, interpolation=params['imshow']['interpolation'])
    ax[1].axis("off")
    ax[1].set_title(label2, y=subplot_titles_y_loc,fontdict=params['subplot_titles_fontdict'] )

    plt.tight_layout()
    plt.show()

    if save_file:

        # if name provided
        if png_filename is not None:

            #if base_folder is provided, check of overlaps
            if base_folder is not None:
                # if the folder is already in png_filename, use png_filename
                if base_folder in png_filename:
                    filename = png_filename

                else: # add base_folder to 
                    filename = base_folder+png_filename

        elif png_filename is None:

            if png_name_from_title:
                if len(suptitle_text)>0:
                    
                    title_for_filename = ji.replace_bad_filename_chars(suptitle_text,replace_spaces=False, replace_with='_')

                    if base_folder is not None:
                        filename = base_folder + title_for_filename+'.png'
                    else:
                        filename =title_for_filename+'.png'
                
            else:
                if base_folder is None:
                    base_folder=''
                prefix= base_folder+"wordcloud_figure"
                filename = ji.auto_filename_time(prefix=prefix)
                filename+='.png'

        fig.savefig(filename,facecolor=cfg['background_color'], format='png', frameon=True)
        print(f'figured saved as {filename}')

    return fig, ax



################################################### ADDITIONAL NLP #####################################################
## Adding in stopword removal to the actual dataframe
def make_stopwords_list(incl_punc=True, incl_nums=True, add_custom= ['http','https','...','…','``','co','“','’','‘','”',"n't","''",'u','s',"'s",'|','\\|','amp',"i'm"]):
    from nltk.corpus import stopwords
    import string

    stopwords_list = stopwords.words('english')
    if incl_punc==True:
        stopwords_list += list(string.punctuation)
    stopwords_list += add_custom #['http','https','...','…','``','co','“','’','‘','”',"n't","''",'u','s',"'s",'|','\\|','amp',"i'm"]
    if incl_nums==True:
        stopwords_list += ['0','1','2','3','4','5','6','7','8','9']#[0,1,2,3,4,5,6,7,8,9]
    
    return  stopwords_list


def remove_stopwords(stopwords_list,  text, tokenize=True,return_tokens=False, pattern = "([a-zA-Z]+(?:'[a-z]+)?)"):
    """EX: df['text_stopped'] = df['content'].apply(lambda x: remove_stopwords(stopwords_list,x))"""
    from nltk import regexp_tokenize
    pattern = "([a-zA-Z]+(?:'[a-z]+)?)"
    if tokenize==True:
        from nltk import regexp_tokenize
        
        text = regexp_tokenize(text,pattern)
        
    stopped = [x.lower() for x in text if x.lower() not in stopwords_list]

    if return_tokens==True:
        return regexp_tokenize(' '.join(stopped),pattern)
    else:
        return ' '.join(stopped)

def empty_lists_to_strings(x):
    """Takes a series and replaces any empty lists with an empty string instead."""
    if len(x)==0:
        return ' '
    else:
        return ' '.join(x) #' '.join(tokens)

def load_raw_twitter_file(filename ='data/trumptwitterarchive_export_iphone_only__08_23_2019.csv', date_as_index=True,rename_map={'text':'content','created_at':'date'}): 
    """import raw copy and pasted to csv export from http://www.trumptwitterarchive.com/archive. 
    Rename columns indicated in rename_map and sets the index to a datetimeindex copy of date column."""    
    # old link'data/trump_tweets_01202017_06202019.csv'
    import pandas as pd
    print(f'[io] Loading raw tweet text file: {filename}')
    df = pd.read_csv(filename, encoding='utf-8')
    mapper=rename_map
    df.rename(axis=1,mapper=mapper,inplace=True)
    df['date']=pd.to_datetime(df['date'])
    if date_as_index==True:
        df.set_index('date',inplace=True,drop=False)
    # df.head()
    return df




## NEW 07/11/19 - function for all sentiment analysis

def full_sentiment_analysis(twitter_df, source_column='content_min_clean',separate_cols=True):#, plot_results=True):
    from nltk.sentiment.vader import SentimentIntensityAnalyzer
    sid = SentimentIntensityAnalyzer()

    source_column='content_min_clean'
    twitter_df['sentiment_scores'] = twitter_df[source_column].apply(lambda x: sid.polarity_scores(x))
    twitter_df['compound_score'] = twitter_df['sentiment_scores'].apply(lambda dict: dict['compound'])
    twitter_df['sentiment_class'] = twitter_df['compound_score'].apply(lambda score: 'pos' if score >=0 else 'neg')
    
    
    # Separate result dictionary into columns  (optional)
    if separate_cols==True:
        # Separate Scores into separate columns in df
        twitter_df_out = get_group_sentiment_scores(twitter_df)
    else:
        twitter_df_out = twitter_df
        
    
#     # plot results (optional)
#     if plot_results==True:
        
#         print("RESULTS OF SENTIMENT ANALYSIS BINARY CLASSIFICATION:\n",'-'*60)
#         # Normalized % of troll sentiment classes
#         plot_sent_class = twitter_df_out['sentiment_class'].value_counts()
#         plot_sent_class_norm = plot_sent_class/(sum(plot_sent_class))
#         print('\tNormalized Troll Classes:\n',plot_sent_class_norm)


#         with plt.style.context('seaborn-notebook'):
#             boxplot = df_sents.boxplot(column=['neg','neu','pos'],notch=True,figsize=(6,4))
#             boxplot.set_xticklabels(['Negative','Neutral','Positive']);
#             boxplot.set_title('Sentiment Scores By Word Type')
#             boxplot.set_ylabel('Sentiment Score')
    
    return twitter_df_out
        
        

# Write a function to extract the group scores from the dataframe
def get_group_sentiment_scores(df, score_col='sentiment_scores'):
    import pandas as pd
    series_df = df[score_col]
    series_neg = series_df.apply(lambda x: x['neg'])
    series_pos = series_df.apply(lambda x: x['pos'])
    series_neu = series_df.apply(lambda x: x['neu'])
    
    series_neg.name='neg'
    series_pos.name='pos'
    series_neu.name='neu'
    
    df = pd.concat([df,series_neg,series_neu,series_pos],axis=1)
    return df



def get_tweet_lemmas(df, text_column = 'cleaned_stopped_tokens',name_for_lemma_col='cleaned_stopped_lemmas'):
        
    text_data = df[text_column]
    
    def lemmatize_tweet(x):
        
        import functions_combined_BEST as ji
        if isinstance(x,str):
            from nltk import regexp_tokenize
            pattern = ji.make_regexp_pattern()
            x = regexp_tokenize(x,pattern)
            
            
        from nltk.stem import WordNetLemmatizer
        lemmatizer=WordNetLemmatizer()
        output = []
        for word in x:
            output.append(lemmatizer.lemmatize(word))
        output = ' '.join(output)
        return output
            
        
    df[name_for_lemma_col] = df[text_column].apply(lambda x:lemmatize_tweet(x))
    return df
    


def full_twitter_df_processing(df,raw_tweet_col='content', name_for_cleaned_tweet_col='content_cleaned', name_for_stopped_col=None, name_for_tokenzied_stopped_col = None,lemmatize=True,name_for_lemma_col='cleaned_stopped_lemmas' ,use_col_for_case_ratio=None,  use_col_for_sentiment='cleaned_stopped_lemmas', RT=True, urls=True,  hashtags=True, mentions=True, str_tags_mentions=True,stopwords_list=[], force=False):
    """
    Accepts df_full, which contains the raw tweets to process, the raw_col name, the column to fill.
    If force=False, returns error if the fill_content_col already exists.
    Processing Workflow:1) Create has_RT, starts_RT columns. 2) Creates [fill_content_col,`content_min_clean`] cols after removing 'RT @mention:' and urls.
    3) Removes hashtags from fill_content_col and saves hashtags in new col. 4) Removes mentions from fill_content_col and saves to new column.
    - if use_cols_for_case_ration is None, the partially completed content_min_clean col is used (only urls and RTs removed)
     Save 'hashtags' column containing all hastags
    
    Args:
        df ([DataFrame]): extract from twitter archive 
        raw_tweet_col (str, optional): text column to process. Defaults to 'content'.
        name_for_cleaned_tweet_col (str, optional): name for new coumn with cleaned tweets   . Defaults to 'content_cleaned'.
        name_for_stopped_col ([type], optional): name for new coumn with stopwords removed. Defaults to None.
        name_for_tokenzied_stopped_col ([type], optional): name for new coumn with tokenized version of stopped_col. Defaults to None.
        lemmatize (bool, optional): [description]. Defaults to True.
        name_for_lemma_col (str, optional): [description]. Defaults to 'cleaned_stopped_lemmas'.
        use_col_for_case_ratio ([type], optional): [description]. Defaults to None.
        use_col_for_sentiment (str, optional): [description]. Defaults to 'cleaned_stopped_lemmas'.
        RT (bool, optional): [description]. Defaults to True.
        urls (bool, optional): [description]. Defaults to True.
        hashtags (bool, optional): [description]. Defaults to True.
        mentions (bool, optional): [description]. Defaults to True.
        str_tags_mentions (bool, optional): [description]. Defaults to True.
        stopwords_list (list, optional): [description]. Defaults to [].
        force (bool, optional): [description]. Defaults to False.
    """
    
            
    #
    import re
    from nltk import regexp_tokenize
    pattern = "([a-zA-Z]+(?:'[a-z]+)?)"
    track_fill_content_col=0
    cleaned_tweet_col = name_for_cleaned_tweet_col
    fill_content_col = cleaned_tweet_col

    if force==False:
        if cleaned_tweet_col in df.columns:
            raise Exception(f'{fill_content_col} already exists. To overwrite, set force=True.')

    if RT ==True:

        # Creating columns for tweets that `has_RT` or `starts_RT`
        df['has_RT']=df[raw_tweet_col].str.contains('RT')
        df['starts_RT']=df[raw_tweet_col].str.contains('^RT')

        ## FIRST REMOVE THE RT HEADERS

        # Remove `RT @Mentions` FIRST:
        re_RT = re.compile('RT [@]?\w*:')

        check_content_col = raw_tweet_col
        fill_content_col = cleaned_tweet_col

        df['content_starts_RT'] = df[check_content_col].apply(lambda x: re_RT.findall(x))
        df[fill_content_col] =  df[check_content_col].apply(lambda x: re_RT.sub(' ',x))
        track_fill_content_col+=1


    if urls==True:
        ## SECOND REMOVE URLS
        # Remove urls with regex
        urls = re.compile(r"(http[s]?://\w*\.\w*/+\w+)")
        
        if track_fill_content_col==0:
            check_content_col = raw_tweet_col
        else:
            check_content_col = fill_content_col

        fill_content_col = fill_content_col

        # df_full['content_urls'] = df_full[check_content_col].apply(lambda x: urls.findall(x))
        df[fill_content_col] =  df[check_content_col].apply(lambda x: urls.sub('',x))

        ## SAVE THIS MINIMALLY CLEANED CONTENT AS 'content_min_clean'
        df['content_min_clean'] =  df[fill_content_col]
        track_fill_content_col+=1


    ## 08/25 ADDING REMOVAL OF PUNCATUATION to min clean @ And # SYMBOLS FOR MIN CLEAN
       
    ## Case Ratio Calculation (optional)
    if use_col_for_case_ratio is None:
        use_col_for_case_ratio='content_min_clean'
        
    df['case_ratio'] = df[use_col_for_case_ratio].apply(lambda x: case_ratio(x))
    print(f'[i] case_ratio calculated from {use_col_for_case_ratio} before text to lowercase')


    def quick_fix(x):
        import string
        punc_list = string.punctuation
        x = x.lower()
        for punc in punc_list:
            x = x.replace(punc,' ')
        return x

    df['content_min_clean'] =   df['content_min_clean'].apply(lambda x: quick_fix(x)) #.replace('#',' '))
    print(f'[i] case->lower and punctuation removed from "content_min_clean" ' )



    if hashtags==True:

        if track_fill_content_col==0:
            check_content_col = raw_tweet_col
        else:
            check_content_col = fill_content_col

        fill_content_col = fill_content_col

        ## REMOVE AND SAVE HASHTAGS, MENTIONS
        # Remove and save Hashtags
        hashtags = re.compile(r'\#\w*')

        df['content_hashtags'] =  df[check_content_col].apply(lambda x: hashtags.findall(x))
        df[fill_content_col] =  df[check_content_col].apply(lambda x: hashtags.sub(' ',x))
        track_fill_content_col+=1
        
        if str_tags_mentions==True: 
            df['hashtag_strings'] = df['content_hashtags'].apply(lambda x: empty_lists_to_strings(x))
        

    if mentions==True:

        if track_fill_content_col==0:
            check_content_col = raw_tweet_col
        else:
            check_content_col = fill_content_col

        fill_content_col = fill_content_col

        # Remove and save mentions (@)'s
        mentions = re.compile(r'\@\w*')


        df['content_mentions'] =  df[check_content_col].apply(lambda x: mentions.findall(x))
        df[fill_content_col] =  df[check_content_col].apply(lambda x: mentions.sub(' ',x))
        track_fill_content_col+=1

        if str_tags_mentions==True: 
            df['mention_strings'] = df['content_mentions'].apply(lambda x: empty_lists_to_strings(x))


    ## Creating content_stopped columns and then tokens_stopped column
    if name_for_stopped_col is None:
        stop_col_name = fill_content_col+'_stop'
    else: 
        stop_col_name = name_for_stopped_col
    print('[i] stopped text column: ',stop_col_name)

    if name_for_tokenzied_stopped_col is None:
        stopped_tok_col_name =  fill_content_col+'_stop_tokens'
    else:
        stopped_tok_col_name = name_for_tokenzied_stopped_col
    print('[i] tokenized stopped text column: ',stopped_tok_col_name)

    if len(stopwords_list)==0:
        stopwords_list=make_stopwords_list()

    df[stop_col_name] = df[fill_content_col].apply(lambda x: remove_stopwords(stopwords_list,x,tokenize=True, return_tokens=False, pattern=pattern))
    df[stopped_tok_col_name] = df[stop_col_name].apply(lambda x: remove_stopwords(stopwords_list,x,tokenize=True, return_tokens=True, pattern=pattern))

    if lemmatize:
        # text_data = df[stopped_tok_col_name]
        df = get_tweet_lemmas(df, text_column=stopped_tok_col_name, name_for_lemma_col=name_for_lemma_col)
        print(f'[i] lemmaztied columns: {name_for_lemma_col}')

    # ## Calculate case ratio is non-default was used
    # if use_col_for_case_ratio is not None and 'content_min_clean' not in use_col_for_case_ratio:
    #     df['case_ratio'] = df[use_col_for_case_ratio].apply(lambda x: case_ratio(x))
    #     print(f'[i] case_ratio calculated from {use_col_for_case_ratio}')

    ## Sentiment Analysis (optional)
    if use_col_for_sentiment is not None:
        df = full_sentiment_analysis(df,source_column=use_col_for_sentiment,separate_cols=True)
    
    df.sort_index(inplace=True)
    return df



def case_ratio(msg):
    """Accepts a twitter message (or used with .apply(lambda x:)).
    Returns the ratio of capitalized characters out of the total number of characters.
    
    EX:
    df['case_ratio'] = df['text'].apply(lambda x: case_ratio(x))"""
    import numpy as np
    if isinstance(msg,str)==False:
        error = f"[!] Input string for case_ratio is not a string, it is a {type(msg)}"
        raise Exception(error)

    msg_length = len(msg)

    if msg_length == 0:
        return np.nan

    test_upper = [1 for x in msg if x.isupper()]
    # test_lower = [1 for x in msg if x.islower()]
    test_ratio = np.round(sum(test_upper)/msg_length,5)
    return test_ratio


def get_group_texts_for_word_cloud(twitter_df, text_column='content_min_clean', groupby_column='delta_price_class'):
    
    groups = twitter_df.groupby(groupby_column).groups
    group_df_dict = {}
    for group in groups.keys():
        group_df = twitter_df.groupby(groupby_column).get_group(group)
        group_df_dict[group]= {'df':group_df}
        
    
    group_text_dict = {}
    for k,v in group_df_dict.items():
        df = v['df']
        
        from nltk import regexp_tokenize
        pattern = "([a-zA-Z]+(?:'[a-z]+)?)"
        text = df[text_column] #.apply(lambda x: regexp_tokenize(x,pattern))
        text = ','.join(text.values)
        
        text_tokens = regexp_tokenize(text, pattern)
        
        group_text_dict[k]={}
        group_df_dict[k]['text_tokens']= text_tokens
        group_text_dict[k]['tokens']=text_tokens
        
        text_joined = ' '.join(text_tokens)
        
        group_df_dict[k]['joined']=text_joined
        group_text_dict[k]['joined']=text_joined
        
    return group_df_dict, group_text_dict


def make_regexp_pattern():
    pattern="([a-zA-Z]+(?:'[a-z]+)?)"
    return pattern




def make_tweet_bigrams_by_group(twitter_df_groups,top_n=20,text_key=None,
                                colname_if_key_is_df='cleaned_stopped_content',
                                group_label_key={'neg':'Stock Market Decreased',
                                                 'pos':'Stock Market Increased'},
                                side_by_side=True, return_group_dfs = False, return_dfs_styled=True):
    """Uses groups from `get_group_texts_for_word_cloud` to display df for each group.
    EX:
    >> twitter_df_groups,twitter_group_text = ji.get_group_texts_for_word_cloud(twitter_df, 
                                                                      text_column='cleaned_stopped_content', 
                                                                      groupby_column='delta_price_class')
    >> make_tweet_bigrams_by_group(twitter_df_groups, )
                                                                      """
    # MAKE BIGRAMS
    from nltk.collocations import BigramAssocMeasures, BigramCollocationFinder, TrigramAssocMeasures, TrigramCollocationFinder
    from nltk import regexp_tokenize
    import functions_combined_BEST as ji
    import bs_ds_local as bs
    from IPython.display import display

    group_names = list(twitter_df_groups.keys())
    group1 = group_names[0]
    group2 = group_names[1]
    
    pattern = ji.make_regexp_pattern()
    
    if text_key is None:
        col = colname_if_key_is_df
        text_data_1 = twitter_df_groups[group1]['df'][col].apply(lambda x: regexp_tokenize(x,pattern))
        text_data_2 = twitter_df_groups[group2]['df'][col].apply(lambda x: regexp_tokenize(x,pattern))
        
    else:
        text_data_1 = twitter_df_groups[group1][text_key]
        text_data_2 = twitter_df_groups[group2][text_key]



    bigram_measures =BigramAssocMeasures()

    tweet_finder1 = BigramCollocationFinder.from_documents(text_data_1)
    tweets_scored1 = tweet_finder1.score_ngrams(bigram_measures.raw_freq)

    tweet_finder2 = BigramCollocationFinder.from_documents(text_data_2)
    tweets_scored2 = tweet_finder2.score_ngrams(bigram_measures.raw_freq)


    df_1 = ji.quick_table(tweets_scored1[:top_n], col_names =['Bigram','Frequency'],
                           caption='Tweet Bigrams', display_df=False)
    
    df_1['Bigram'] = df_1['Bigram'].apply(lambda x: ' '.join(x))
    df_1.set_index('Bigram',inplace=True)
    df_1.columms=['Frequency']
    
    # style df
    cap1 = group_label_key[group1] 
    dfs_1 = df_1.style.set_caption(cap1)

    df_2 = ji.quick_table(tweets_scored2[:top_n], col_names =['Bigram','Frequency'],
                           caption='Tweet Bigrams', display_df=False)
    
    df_2['Bigram'] = df_2['Bigram'].apply(lambda x: ' '.join(x))
    df_2.set_index('Bigram',inplace=True)
    df_2.columms=['Frequency']
    
    # style df
    cap2 = group_label_key[group2] 
    dfs_2 = df_2.style.set_caption(cap2)
    
    if side_by_side==True:
        bs.display_side_by_side(dfs_1,dfs_2)
    else:
        display(dfs_1,dfs_2)
        
    if return_group_dfs:
        if return_dfs_styled:
            return dfs_1, dfs_2
        else:
            return df_1,df_2
        



# def merge_stocks_and_tweets(stocks_grouped,twitter_grouped, on='int_bins',how='left', show_summary=True):
#     """Takes stocks_grouped and twitter_grouped dataframes grouped by hour bins, merges
#     them on the 'int_bins' columns."""
#     import pandas as pd
#     from IPython.display import display
#     def fix_tweet_times(x):
#         import numpy as np
#         import pandas as pd
#         time_format="%Y-%m-%d %T"
#         if isinstance(x,pd.Timestamp) | isinstance(x,pd.DatetimeIndex):
#             return x.strftime(time_format)

#         elif  isinstance(x,list) |  isinstance(x,np.ndarray):
#             ts = [pd.to_datetime(t,format=time_format) for t in x] 
#     #         ts = [t.strftime(time_format) for t in x]#ts]
#             return ts

#         elif pd.isnull(x): 
#     #         print('null')
#             return x

#         else: 
#             ts= pd.to_datetime(str(x)) 
#             timestring = ts.strftime(time_format)        #x.strftime(time_format)
#             return timestring

#     ## MERGE DATAFRAMES
#     df_combined = pd.merge(stocks_grouped,twitter_grouped, on='int_bins',how='left')#,indicator=True)

#     ## ADDING COLUMNS TO WEED THROUGH MISSING DATA FROM MERGER
#     df_combined['has_tweets'] = ~df_combined['group_content'].isna()
#     df_combined['has_stocks'] = ~df_combined['price'].isna()
#     df_combined['has_both'] =(df_combined['has_tweets']==True) & (df_combined['has_stocks']==True)# 


#     ## CLEAN UP DATAFRAME
#     # Drop redundant cols
#     drop_cols = ['date','int_times_y','time_bin_y','time_bin_x','num_per_bin_x']
#     df_combined.drop(drop_cols,axis=1,inplace=True)

#     # Rename distinct stock vs tweet columns
#     rename_map = {'date_time_index_y':'tweet_times',
#                  'date_time_index_x':'stock_times',
#                   'int_times_x':'int_times',
#                   'num_per_bin_y':'num_tweets',
#                   'int_times_x':'int_tweets_for_stocks'
#                  }
#     df_combined  = df_combined.rename(axis=1,mapper=rename_map)

#     ## Add fixed ts for tweet_times
#     df_combined['tweet_times'] = df_combined['tweet_times'].apply(lambda x: fix_tweet_times(x))

#     df_combined.set_index('date_time', inplace=True,drop=False)

#     if show_summary:
#         n=3
#         display(df_combined.groupby('has_both').head(3))
#         display(df_combined.loc[ df_combined['num_tweets']>1].head(n))
#     return df_combined



def merge_stocks_and_tweets(stocks_grouped,twitter_grouped, on='int_bins',how='left', show_summary=True):
    """Takes stocks_grouped and twitter_grouped dataframes grouped by hour bins, merges
    them on the 'int_bins' columns."""
    import pandas as pd
    import numpy as np
    import functions_combined_BEST as ji
    from IPython.display import display

    def fix_tweet_times(x):
        time_format="%Y-%m-%d %T"
        if isinstance(x,pd.Timestamp) | isinstance(x,pd.DatetimeIndex):
            return x.strftime(time_format)

        elif  isinstance(x,list) |  isinstance(x,np.ndarray):
            ts = [pd.to_datetime(t,format=time_format) for t in x] 
    #         ts = [t.strftime(time_format) for t in x]#ts]
            return ts

        elif pd.isnull(x): 
    #         print('null')
            return x

        else: 
            ts= pd.to_datetime(str(x)) 
            timestring = ts.strftime(time_format)        #x.strftime(time_format)
            return timestring

    ## MERGE DATAFRAMES
    df_combined = pd.merge(stocks_grouped,twitter_grouped, on='int_bins',how='left')#,indicator=True)

    ## ADDING COLUMNS TO WEED THROUGH MISSING DATA FROM MERGER
    df_combined['has_tweets'] = ~df_combined['group_content'].isna()
    df_combined['has_stocks'] = ~df_combined['price'].isna()
    df_combined['has_both'] =(df_combined['has_tweets']==True) & (df_combined['has_stocks']==True)# 


    ## CLEAN UP DATAFRAME
    # Drop redundant cols
    drop_cols = ['date','int_times_y','time_bin_y','time_bin_x','num_per_bin_x']
    df_combined.drop(drop_cols,axis=1,inplace=True)

    # Rename distinct stock vs tweet columns
    rename_map = {'date_time_index_y':'tweet_times',
                 'date_time_index_x':'stock_times',
                  'int_times_x':'int_times',
                  'num_per_bin_y':'num_tweets',
                  'int_times_x':'int_tweets_for_stocks'
                 }
    df_combined  = df_combined.rename(axis=1,mapper=rename_map)

    ## Add fixed ts for tweet_times
    df_combined['tweet_times'] = df_combined['tweet_times'].apply(lambda x: fix_tweet_times(x))

    df_combined.set_index('date_time', inplace=True,drop=False)

    if show_summary:
        n=3
        display(df_combined.groupby('has_both').head(3))
        display(df_combined.loc[ df_combined['num_tweets']>1].head(n))
    return df_combined


class Word2vecParams():
    """Class for tracking modelparams used for word2vec"""
    import pandas as pd
    def __init__(self,params=None,verbose=1):
        '''Creates empty dataframe, last_params attribute, displays message'''
        import pandas as pd
        self.last_params = {}
        self._df_ = pd.DataFrame(columns= ['text_column', 'window', 'min_count',
                                           'epochs','sg', 'hs', 'negative', 'ns_exponent'])
        if params is not None:
            
            self._df_.loc[self.get_now()] = params
            self.last_params=params
        else:
            self.last_params ={'text_column': '',
                               'window':'',
                               'min_count':'',
                               'epochs': '',
                               'sg':'',
                               'hs':'',
                               'negative':'',
                               'ns_exponent':''
                              }
        if verbose>0:
            print('[i] call .params_template() for dict to copy/pate.')
#             print('[i] call .show_info() to display param meanings.')

    def get_df(self):
        """returns dataframe of all attempts"""
        return self._df_
    
    def get_now(self):
        """calls external function to get current timestamp"""
        import functions_io as io
        return io.get_now()        
            
    def print_params(self):
        """prints self.last_params"""
        print(self.last_params)
        
    def params_template(self):
        """Prints a template dictionary of possible values. Copy-paste ready."""
        print('#TEMPLATE(call.show_info() for details:')
        print(self._template_)

    
    def append(self, params_dict):
        """Appends internal dataframe using the current time as the index"""
        import pandas as pd
        ts = self.get_now()
        self._df_.loc[ts] = pd.Series(params_dict)
        self.last_params = params_dict
        print('- params saved.')
    
    def show_info(self):
        from IPython.display import display
        import pandas as pd
        info_df = self.info
#         info_df = info_df.style
        capt_text = 'Word2Vec Model Params'
        with pd.option_context('display.max_colwidth',0, 'display.colheader_justify','left'):
            table_style =[{'selector':'caption',
            'props':[('font-size','1.1em'),('color','darkblue'),('font-weight','bold'),
            ('vertical-align','0%')]}]
            dfs = info_df.style.set_caption(capt_text).set_properties(**{'width':'400px',
            'text-align':'left',
#             'padding':'1em',
            'font-size':'1.2em'}).set_table_styles(table_style)
            display(dfs)
                
    _template_="""
        w2vparams = {
        'text_column': 'cleaned_stopped_lemmas',
        'window':3-5,
        'min_count':1-3,
        'epochs':10-20,
        'sg':0 or 1, 
        'hs':'0 or 1,
        'negative': 0 or 5-20 ,
        'ns_exponent':-1.0 to 1.0
        }"""
        
    info = pd.DataFrame.from_dict({
        'text_column':'column to sample',
        'window':'# of words per window',
        'min_count':'# of times word must appear',
        'epochs':'# of training epochs',
        'sg':"embedding method. 1=skip-gram, 0=cbow (default)",
        'hs':"1=hierarchical softmax (default). 0 = use negative sampling (if 'negative' is non-zero)",
        'negative':"5-20, default=0. (# number of 'noisy' words to remove by negative sampling)",
        'ns_exponent':"weight for sampling low/high freq words. 0.0=sample all equally, 1.0=sample by word frequency, (-) values = sample infrequent words more"},
        orient='index',columns=['info'])
    
        
        
def get_w2v_kwargs(params):
    ## get kwargs for make_word2vecmodel
    kwarg_keys =['sg','hs','negative','ns_exponent']
    kwargs = {k:params[k] for k in kwarg_keys}
    return kwargs

