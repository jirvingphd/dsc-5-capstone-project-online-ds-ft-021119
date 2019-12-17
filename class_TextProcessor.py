class TextsProcessor(object):
    """text_proc = TextsProcessor(df_source, text_column='content')
    
    ## If you want to add or remove stopwords from list
    text_proc.update_stopwords(add_words=['word','to','add'], remove_words=['remove','me'])
    text_proc.prepare_corpus()
    https://colab.research.google.com/drive/1REMxM-keRZsrnnVwZuTI2hS3SKVTbogN"""
    
    ## Set default regexp_pattern, verbose
    regexp_pattern = "([a-zA-Z]+(?:'[a-z]+)?)"
    verbose=1
    
    @property
    def corpus(self):
        return self._corpus
    
    @corpus.setter
    def corpus(self,corpus):
        self._corpus=corpus
    
    @property
    def stopwords(self):
        return self._stopwords_
    
    @property
    def df(self):
        return self._df
    
    @df.setter
    def df(self,df):
        self._df =  df
        
    custom_stopwords  = ['http','https','...','…','``','co','“','’','‘','”',
                                         "n't","''",'u','s',"'s",'|','\\|','amp',"i'm",'rt']
    
    

    
    def __init__(self, df_text_source, text_column=None, verbose=1):
        import inspect
        import pandas as pd
        
        self._df = df_text_source.copy()
        self._text_column = text_column
        self.verbose = verbose
        
        
        ## SAVE INPUT TEXT AS .text_data
        # if input is df:
        if isinstance(df_text_source, pd.DataFrame)==True:
            
            # if no text_column given, 
            if text_column is None:
                
                num_text_cols = df_text_source.get_dtype_counts()['object']
                
                # check if there is only 1 string column, if so use that.
                if num_text_cols==1:
                    col_types = df_text_source.dtypes
                    text_column = col_types.loc[col_types==True].index
                else:
                    raise Exception('Must specify `text_column` if >1 string column.')                
            
            ## save series as text_data    
            self.text_data = self._df[text_column].copy()#df_text_source[text_column].copy()      

        
        elif isinstance(df_text_source, list):
            ## save text_data as list
            self.text_data = df_text_source
            
        elif isinstance(df_text_source, str):
            ## save text_data as list of str
            self.text_data = [str]
            
        else:
            raise Exception('Input was neither a dataframe, string, or list')
            
            
        ## Make stopwords list
        self._make_stopwords()
       
            

    def prepare_corpus(self,source_text=None, return_result=False,verbose=verbose):
        """
        Creates a Bag of Words For All Texts in the DataFrame combined
        """
        if source_text is None:
            source_text = self.text_data
            
        ## Join series or lists with >1 element    
        if len(source_text)>1:
            #1. Combine all strings from series
            text_data = ' '.join(source_text)
        else:
            text_data = source_text[0]

        #2. Regexp Tokenize
        text_tokens = self.regexp_tokenize(text_data)
        
        #3. Remove stopwords
        stopped_text = self.remove_stopwords(text_tokens)

        # stopped_text = [word.lower() for word in text_tokens if word not in self.stopwordslist]
        self.corpus = stopped_text
        
        # lemmas
        self.lemmatize_text()
        
        ## Get unique words for vocabulary
        vocab = list(set(stopped_text))
        self.vocab = vocab
        
        
        if verbose>0:
            print('[i] Text has been regexp_tokenized, stopwords have been removed.\n\t- `self.corpus` = processed body of text.')
#         self.cleaned_text = stopped_text
        
        if return_result:
            return stopped_text

        
    def prepare_texts(self):
        """Prepare each row of the self._df separately and add to self._df
        "Processed Columns:
        1. 'tokens' = regexp_tokenized
        2. 'stopped' = 'tokens' with stopwords removed
        3. 'stopped_lemmas' = 'stopped' lemmatized with nltk
        """
        
        df= self._df
        text_column = self._text_column
        
        df['tokens'] = df[text_column].apply(lambda x: self.regexp_tokenize(text=x))
        df['stopped'] = df['tokens'].apply(lambda x: self.remove_stopwords(x))
        df['stopped_lemmas'] = df['stopped'].apply(lambda x:self.lemmatize_text(x))
        

        
        
    
    
    def regexp_tokenize(self,text,pattern=None):
        """
        [summary]
                
        Args:
            text_data ([type]): [description]
            pattern ([type], optional): [description]. Defaults to None.
        
        Returns:
            [type]: [description]
        """

        if pattern is None:
            pattern = self.regexp_pattern

        # CREATING TEXT DICT FOR FREQUENCY DISTRIBUTIONS
        from nltk import regexp_tokenize as reg_tok
        try:
            tokenized_data = reg_tok(text, pattern)
        except TypeError:
            print('TypeError: text_data is already tokenized.')
            tokenized_data = text
        return tokenized_data
    
            
    def _make_stopwords(self, incl_punc=True, incl_nums=True, 
                            custom_stopwords=None):
        """
        [summary]
        Args:
            incl_punc (bool, optional): [description]. Defaults to True.
            incl_nums (bool, optional): [description]. Defaults to True.
            custom_stopwords ([type], optional): [description]. Defaults to None.
        
        Returns:
            None: It updates ._stopwords_
        """
        from nltk.corpus import stopwords
        import string
        
        if custom_stopwords is None:
            custom_stopwords= self.custom_stopwords

        stopwords_list = stopwords.words('english')
        if incl_punc==True:
            stopwords_list += list(string.punctuation)
        stopwords_list += custom_stopwords #['http','https','...','…','``','co','“','’','‘','”',"n't","''",'u','s',"'s",'|','\\|','amp',"i'm"]
        if incl_nums==True:
            stopwords_list += [0,1,2,3,4,5,6,7,8,9]
            
        self._stopwords_ = stopwords_list


    def update_stopwords(self,add_words=[],remove_words=[],update_corpus=True):
        stopwords = self.stopwords
        [stopwords.append(x) for x in add_words]


        [stopwords.remove(x) for x in remove_words if x in stopwords]     

            
            
        self._stopwords_ = stopwords
        if update_corpus:
            self.prepare_corpus()

    def remove_stopwords(self,text_data=None):#,  regexp_tokenize=True, return_tokens=False):
        #"""EX: df['text_stopped'] = df['content'].apply(lambda x: remove_stopwords(stopwords,x))"""
        # if text_data is None:
            # text_data = self.corpus()
        
        stopped_text = [x.lower() for x in text_data if x.lower() not in self.stopwords]
        return stopped_text
        # if return_tokens==True:
        #     return regexp_tokenize(' '.join(stopped_text),pattern)
        # else:
        #     return ' '.join(stopped_text)
        
    def fit_FreqDist(self, prepared_text=None, plot=False):
        """ Fits nltk's FreqDist on tokenized text and saved as .freq_dist"""

        if prepared_text is None:
            prepared_text = self._corpus
                                
        from nltk import FreqDist
        freq_dist = FreqDist(prepared_text)
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
            
    def lemmatize_text(self, text_data=None,join_output=True,verbose=verbose):
        """Lemmatize text_data. 
        - If text_data is None, use self.corpus"""
        
        if text_data is None:
            used_corpus = True
            text_data = self.corpus
        else:
            used_corpus=False
            
        x = text_data
        if isinstance(x,str):
            pattern = self.regexp_pattern
            x = self.regexp_tokenize(x)
            
        ## Import Lemmatizer
        from nltk.stem import WordNetLemmatizer
        lemmatizer=WordNetLemmatizer()
        
        output = []
        for word in x:
            output.append(lemmatizer.lemmatize(word))

        ## If corpus was used, add .corpus_lemmas, no return
        if used_corpus:
            self.corpus_lemmas = output 
            vocab_lemmas = list(set(output))
            self.vocab_lemmas = vocab_lemmas
            if verbose>0:
                print('[i] Corpus lemmatized and stored as `self.corpus_lemmas`')
        else:
            if join_output:
                output = ' '.join(output)
        
            return output
        
    def fit_bigrams(self,text_data=None,show_top_bigrams=True, top_n=20):
        """If text_data is None, use self.corpus"""
        import pandas as pd
        from nltk.collocations import BigramAssocMeasures, BigramCollocationFinder, TrigramAssocMeasures, TrigramCollocationFinder
        
        if text_data is None:
            text_data = self.corpus
            
        
        ## Instantiate and fit bigram functions
        bigram_measures =BigramAssocMeasures()
        finder = BigramCollocationFinder.from_documents(text_data)
        scored = finder.score_ngrams(bigram_measures.raw_freq)
        self.bigrams = scored
        
        if show_top_bigrams:
            from IPython.display import display
            bigrams_to_show = scored[:top_n]
            col_names =['Bigram','Frequency']
            caption=f'Top {top_n} Bigrams'
            df = pd.DataFrame.from_records(bigrams_to_show,columns=col_names)
            dfs = df.style.set_caption(caption)
            display(dfs)


            
                           
            
    def summary(self,top_n_words = 10):
        dashes = '---'*20 
        print(dashes)
        print('\tSUMMARY REPORT:')
        print(dashes,'\n')
        
        num_words = len(self.vocab)#corpus)
        num_lemmas = len(self.vocab_lemmas)#corpus_lemmas)
        print(f'[i] There are {num_words} in corpus vocab (self.vocab).')
        
        if hasattr(self,'corpus_lemmas'):
            print(f'[i] There are {num_lemmas} lemmas in `self.vocav_lemmas`')
            
        print(f'\n[i] regexp_tokenize pattern = {self.regexp_pattern}')

        
        print('\n[i] Stopword List:')
        print('\t',self.stopwords)
        
        fit_freqdist = hasattr(self,'FreqDist')
        if fit_freqdist==True:
            icon = 'i'
            msg = 'is fit (self.FreqDist).'
        else:
            icon ='!'
            msg = 'has not yet been fit.\n\t- call `self.fit_FreqDist()` to do so.' 
        print(f'\n[{icon}] FreqDist {msg}')
        
        if fit_freqdist==True:
            print(f'- Top {top_n_words} most frequent words:')
            top_words = self.FreqDist.most_common(top_n_words)
            for i,word in enumerate(top_words):
                
                print(f"{i+1:{5}}. {word[0]:>{10}} ({word[1]})")
        
    
    # def fit_tokenizer(self):
    #     pass
    
    # def fit_word2vec(self):
    #     pass
    
    # def help(self):
    #     """
    #     Initialize TEXT with a df containing the text_data, text_labels(for classificaiton),
    #     and a word2vec model.
    #     >> txt = TEXT(df_combined,'content_stop',None,word_model)
        
    #     ## FOR GETTING X SEQUENCES FOR NEW INPUT TEXT
    #     * To get sequences for each row in a series:
    #     >> text_series = df_combined['content_min_clean']
    #     >> X_seq = txt.text_to_sequences(text_series,regexp_tokenize=True)
        
    #     * To revert generated sequneces back to text:
    #     >> text_from_seq = txt.sequences_to_text(X_seq)
        
    #     ## TO GET CORPUS-WIDE NLP PROCESSING:
    #     * for word frequencies:
    #     >> txt.freq_dist.#[anything you can get from nltk's FreqDist]
    #     """
    #     pass
    
    
