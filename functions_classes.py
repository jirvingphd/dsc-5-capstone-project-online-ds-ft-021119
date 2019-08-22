
class TEXT(object):
    """Class intended for preprocessing text for use with Word2Vec vectors, Kera's Tokenizer, 
    creating FreqDists and an embedding_layer for Kera's models.
    TEXT = TEXT(text_data, text_labels, word2vecmodel)"""
    
    def __init__(self,df, text_data_col,text_labels_col=None,word2vec_model=None,fit_model=False,verbose=0):
        """Initializes the class with the text_data (series), text_labels (y data), and a word2vecmodel.
        Performs all processing steps on entire body of text to generate corupus-wide analyses. 
        i.e. FreqDist, word2vec models, fitting tokenzier
        
        - if no word2vec model is provided, one is fit on the text_data using TEXT.fit_word2vec()
        - calls on fit_tokenizer() to fit keras tokenizer to text_sequences.
        text_data and text_labels are saved as TEXT._text_data_ and TEXT._text_labels_"""
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
        self.prepare_text_body()

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
            # ## Fit keras tokenizer
            # self.fit_tokenizer()
            
            # ## Get FreqDist
            # self.make_freq_dist()
            
            # ## Create Embedding Layer
            # self.get_embedding_layer(return_layer=False)

    def fit_models(self):
        
        ## Fit Word2Vec
        if self.wv is None:
            self.fit_word2vec(text_data=self._text_data_)#self.tokenized_text_body)

        ## Fit keras tokenizer
        self.fit_tokenizer()
        
        ## Get FreqDist
        self.make_freq_dist()

        ## Create Embedding Layer
        self.get_embedding_layer(return_layer=False)
    

        
    def prepare_text_body(self,text_data_to_prep=None,delim=','):
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


    def apply_stopwords(self, text_data, stopwords_list=None, tokenize=True,
                        pattern = "([a-zA-Z]+(?:'[a-z]+)?)",return_tokens=False):
        """EX: df['text_stopped'] = df['content'].apply(lambda x: apply_stopwords(stopwords_list,x))"""
        print('return to .apply_stopwords and verify saved vs unused variables')
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

    def get_embedding_layer(self,X_sequences = None,input_size=None, return_matrix=False, return_layer=True):
        """Uses the word2vec model to construct an embedding_layer for Keras models.
        To override the default size of the input for the embedding layer, provide an input_size value
        (which will likely be the size of hte vocabulary being fed in for predictions)."""
        import numpy as np
        import pandas as pd        

        if X_sequences is None:
            X_sequences = self.X_sequences

        if input_size is not None:
            print('[!] RETURN TO get_embedding_layer to verify when to override vocab_size / input size if input_size is not None')
        vocab_size = len(self.wv.vocab)
        vector_size = self.wv.vector_size
            
        ## Create the embedding matrix from the vectors in wv model 
        embedding_matrix = np.zeros((vocab_size + 1, vector_size))
        for i, vec in enumerate(self.wv.vectors):
            embedding_matrix[i] = vec
            embedding_matrix.shape
        self._embedding_matrix_ = embedding_matrix
        
        from keras import layers 

        
        embedding_layer =layers.Embedding(vocab_size+1,
                                          vector_size,
                                          input_length=X_sequences.shape[1],
                                          weights=[embedding_matrix],
                                          trainable=False)
        self.embedding_layer = embedding_layer
        
        ## Return outputs
        return_list=[]
        if return_matrix:
            return_list.append(embedding_matrix)
        if return_layer:
            return_list.append(embedding_layer)
        return return_list[:]
        
    def make_freq_dist(self, plot=False):
        """ Fits nltk's FreqDist on tokenized text and saved as .freq_dist"""
        from nltk import FreqDist
        freq_dist = FreqDist(self.tokenized_text_body)
        self.FreqDist = freq_dist
        
        if plot==True:
            self.freq_dist_plot()
        
    def freq_dist_plot(self, top_n_words=25):
        """Create FreqDist plot of top_n_words"""
        import matplotlib.pyplot as plt
        try:
            self.FreqDist
        except:
            self.make_freq_dist()

        with plt.style.context('seaborn-notebook'):
            self.FreqDist.plot(top_n_words)
        
    
    def summary_report(self):
        """Print summary info about word2vec vocab, vectors, tokenized_text and embedding matrix"""
        print(f"Word2Vec Vocabulary Size = {len(self.wv.vocab)}")
        print(f"Word2Vec vector size = {self.wv.vector_size}")
        print(f"\nLength of tokenized_text = {len(self.tokenized_text_body)}")
        print(f"_embedding_matrix_ size = {self._embedding_matrix_.shape}")
        
        
    def prepare_text_sequences(self, process_as_tweets=True, tweet_final_col='cleaned_tweets'):
        """Individually process each entry in text_data for stopword removal and tokenization:
        stopwords_list = TEXT.make_stopwords_list()"""
        self.make_stopwords_list()

        if process_as_tweets:
            self.tweet_specific_processing(tweet_final_col='cleaned_tweets')
            text_to_process = self.df[tweet_final_col]
            colname_base = tweet_final_col
        else:
            text_to_process = self.df['input_text_data']
            colname_base = 'text'
        
        # Get stopped, non-tokenzied text
        proc_text_series = text_to_process.apply(lambda x: self.apply_stopwords(x,
                                                                                stopwords_list=None,
                                                                                tokenize=True,
                                                                                return_tokens=False))
        self.df[colname_base+'_stopped'] = proc_text_series
        
        # Get stopped-tokenized text
        proc_text_tokens = text_to_process.apply(lambda x: self.apply_stopwords(x,
                                                                        stopwords_list=None,
                                                                        tokenize=True,
                                                                        return_tokens=True))
        self.df[colname_base+'_stopped_tokens'] = proc_text_tokens

        
    def tweet_specific_processing(self,tweet_final_col='cleaned_tweets'):
        import re
        # Get initiial df
        df = self.df
        
        raw_tweet_col = 'input_text_data'
        fill_content_col = tweet_final_col

        ## PROCESS RETWEETS
        df['has_RT']=df[raw_tweet_col].str.contains('RT')
        df['starts_RT']=df[raw_tweet_col].str.contains('^RT')
        
        re_RT = re.compile('RT [@]?\w*:')

        df['content_starts_RT'] = df[raw_tweet_col].apply(lambda x: re_RT.findall(x))
        df[fill_content_col] =  df[raw_tweet_col].apply(lambda x: re_RT.sub(' ',x))
        
        
        ## PROCESS URLS
        urls = re.compile(r"(http[s]?://\w*\.\w*/+\w+)")
        
        check_content_col = fill_content_col
        df['content_urls'] = df[check_content_col].apply(lambda x: urls.findall(x))
        df[fill_content_col] =  df[check_content_col].apply(lambda x: urls.sub(' ',x))

        ## PROCESS HASHTAGS
        hashtags = re.compile(r'\#\w*')

        df['content_hashtags'] =  df[check_content_col].apply(lambda x: hashtags.findall(x))
        df[fill_content_col] =  df[check_content_col].apply(lambda x: hashtags.sub(' ',x))
        
        ## PROCESS MENTIONS
        # Remove and save mentions (@)'s
        mentions = re.compile(r'\@\w*')
        df['content_mentions'] =  df[check_content_col].apply(lambda x: mentions.findall(x))
        df[fill_content_col] =  df[check_content_col].apply(lambda x: mentions.sub(' ',x))

        
        self.df = df 
        def empty_lists_to_strings(x):
            """Takes a series and replaces any empty lists with an empty string instead."""
            if len(x)==0:
                return ' '
            else:
                return ' '.join(x) #' '.join(tokens)
        
        def help(self):
            """
            Initialize TEXT with a df containing the text_data, text_labels(for classificaiton),
            and a word2vec model.
            >> txt = TEXT(df_combined,'content_stop',None,word_model)
            
            ## FOR GETTING X SEQUENCES FOR NEW INPUT TEXT
            * To get sequences for each row in a series:
            >> text_series = df_combined['content_min_clean']
            >> X_seq = txt.text_to_sequences(text_series,regexp_tokenize=True)
            
            * To revert generated sequneces back to text:
            >> text_from_seq = txt.sequences_to_text(X_seq)
            
            ## TO GET CORPUS-WIDE NLP PROCESSING:
            * for word frequencies:
            >> txt.freq_dist.#[anything you can get from nltk's FreqDist]
            """


    def replace_embedding_layer(self, twitter_model, input_text_series, verbose=2):
        """Takes the original Keras model with embedding_layer, a series of new text, a TEXT object,
        and replaces the embedding_layer (layer[0]) with a new embedding layer with the correct size for new text"""
        ## CONVERT MODEL TO JSON, REPLACE OLD INPUT LAYER WITH NEW ONE FROM TEXT object
        json_model = twitter_model.to_json()
        from functions_classes import TEXT
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
        X_seq = self.text_to_sequences(text_data = input_text_series,regexp_tokenize=True)

        if verbose>0:
            ## To get Text back from X_seq:
            # text_from_seq = TEXT.sequences_to_text(X_seq)
            print('(num_rows_in_df, num_words_in_vocab)')
            print(X_seq.shape)
            
        ## Get new embedding layer's config  (that is fit to new text)
        output = self.get_embedding_layer(X_sequences=X_seq,input_size=X_seq.shape[1])
        new_emb_config = output[0].get_config()
        
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


########################################################################################    


## NEW FUNCTIONS FOR WORD2VEC AND KERAS TOKENIZATION/SEQUENCE GENERATION
def make_word2vec_model(df, text_column='content_min_clean', regex_pattern ="([a-zA-Z]+(?:'[a-z]+)?)",
                       vector_size=300,window=3,min_count=1,workers=3,epochs=10,summary=True, return_full=False):
    
    ## Regexp_tokenize text_column
    from nltk import regexp_tokenize
    text_data = df[text_column].apply(lambda x: regexp_tokenize(x, regex_pattern))

    ## Instantiate Word2Vec Model
    from gensim.models import Word2Vec
    vector_size = 300
    wv_keras = Word2Vec(text_data, size=vector_size, window=window, min_count=min_count, workers=workers)
    
    # Train Word2Vec Model
    wv_keras.train(text_data,total_examples=wv_keras.corpus_count, epochs=epochs)

    
    ## Display summary
    if summary:
        wv = wv_keras.wv
        vocab_size = len(wv_keras.wv.vocab)
        print(f'Word2Vec model trained using:   min_count={min_count}, window={window}, over {epochs} epochs.')
        print(f'There are {vocab_size} words the vocabulary, with a vector size {vector_size}.')
        if return_full:
            ans = 'full model'
        else:
            ans = 'model.wv'
        
        print(f'- output is {ans}')
        
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

def make_keras_embedding_layer(word2vec_model,X_sequences,embedding_matrix= None):
        """Creates an embedding layer for Kera's neural networks using the 
        embedding matrix and text X_sequences"""
        if embedding_matrix is None:
            embedding_matrix = make_embedding_matrix(word2vec_model,verbose=0)
        
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
    