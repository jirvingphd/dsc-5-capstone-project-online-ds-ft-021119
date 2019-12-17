### SOURCE:
```python
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

```
### SOURCE:
```python
def get_wv_from_word2vec(word2vec_model):
    """Checks if model is full word2vec or .wv attribute. 
    Returns wv."""
    import gensim 
    if isinstance(word2vec_model,gensim.models.word2vec.Word2Vec):
        wv = word2vec_model.wv
    elif isinstance(word2vec_model,gensim.models.keyedvectors.Word2VecKeyedVectors):
        wv = word2vec_model
    return wv

```
### SOURCE:
```python
def get_w2v_kwargs(params):
    ## get kwargs for make_word2vecmodel
    kwarg_keys =['sg','hs','negative','ns_exponent']
    kwargs = {k:params[k] for k in kwarg_keys}
    return kwargs

```
### SOURCE:
```python
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

```
### SOURCE:
```python
def get_wv_from_word2vec(word2vec_model):
    """Checks if model is full word2vec or .wv attribute. 
    Returns wv."""
    import gensim 
    if isinstance(word2vec_model,gensim.models.word2vec.Word2Vec):
        wv = word2vec_model.wv
    elif isinstance(word2vec_model,gensim.models.keyedvectors.Word2VecKeyedVectors):
        wv = word2vec_model
    return wv

```
### SOURCE:
```python
def get_w2v_kwargs(params):
    ## get kwargs for make_word2vecmodel
    kwarg_keys =['sg','hs','negative','ns_exponent']
    kwargs = {k:params[k] for k in kwarg_keys}
    return kwargs

```
### SOURCE:
```python
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

```
### SOURCE:
```python
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

```
### SOURCE:
```python
def get_wv_from_word2vec(word2vec_model):
    """Checks if model is full word2vec or .wv attribute. 
    Returns wv."""
    import gensim 
    if isinstance(word2vec_model,gensim.models.word2vec.Word2Vec):
        wv = word2vec_model.wv
    elif isinstance(word2vec_model,gensim.models.keyedvectors.Word2VecKeyedVectors):
        wv = word2vec_model
    return wv

```
### SOURCE:
```python
def get_w2v_kwargs(params):
    ## get kwargs for make_word2vecmodel
    kwarg_keys =['sg','hs','negative','ns_exponent']
    kwargs = {k:params[k] for k in kwarg_keys}
    return kwargs

```
### SOURCE:
```python
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

```
### SOURCE:
```python
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

```
### SOURCE:
```python
def get_wv_from_word2vec(word2vec_model):
    """Checks if model is full word2vec or .wv attribute. 
    Returns wv."""
    import gensim 
    if isinstance(word2vec_model,gensim.models.word2vec.Word2Vec):
        wv = word2vec_model.wv
    elif isinstance(word2vec_model,gensim.models.keyedvectors.Word2VecKeyedVectors):
        wv = word2vec_model
    return wv

```
### SOURCE:
```python
def get_w2v_kwargs(params):
    ## get kwargs for make_word2vecmodel
    kwarg_keys =['sg','hs','negative','ns_exponent']
    kwargs = {k:params[k] for k in kwarg_keys}
    return kwargs

```
### SOURCE:
```python
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

```
### SOURCE:
```python
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

```
### SOURCE:
```python
def get_wv_from_word2vec(word2vec_model):
    """Checks if model is full word2vec or .wv attribute. 
    Returns wv."""
    import gensim 
    if isinstance(word2vec_model,gensim.models.word2vec.Word2Vec):
        wv = word2vec_model.wv
    elif isinstance(word2vec_model,gensim.models.keyedvectors.Word2VecKeyedVectors):
        wv = word2vec_model
    return wv

```
### SOURCE:
```python
def get_w2v_kwargs(params):
    ## get kwargs for make_word2vecmodel
    kwarg_keys =['sg','hs','negative','ns_exponent']
    kwargs = {k:params[k] for k in kwarg_keys}
    return kwargs

```
### SOURCE:
```python
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

```
### SOURCE:
```python
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

```
### SOURCE:
```python
def get_wv_from_word2vec(word2vec_model):
    """Checks if model is full word2vec or .wv attribute. 
    Returns wv."""
    import gensim 
    if isinstance(word2vec_model,gensim.models.word2vec.Word2Vec):
        wv = word2vec_model.wv
    elif isinstance(word2vec_model,gensim.models.keyedvectors.Word2VecKeyedVectors):
        wv = word2vec_model
    return wv

```
### SOURCE:
```python
def get_w2v_kwargs(params):
    ## get kwargs for make_word2vecmodel
    kwarg_keys =['sg','hs','negative','ns_exponent']
    kwargs = {k:params[k] for k in kwarg_keys}
    return kwargs

```
### SOURCE:
```python
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

```
### SOURCE:
```python
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

```
### SOURCE:
```python
def get_wv_from_word2vec(word2vec_model):
    """Checks if model is full word2vec or .wv attribute. 
    Returns wv."""
    import gensim 
    if isinstance(word2vec_model,gensim.models.word2vec.Word2Vec):
        wv = word2vec_model.wv
    elif isinstance(word2vec_model,gensim.models.keyedvectors.Word2VecKeyedVectors):
        wv = word2vec_model
    return wv

```
### SOURCE:
```python
def get_w2v_kwargs(params):
    ## get kwargs for make_word2vecmodel
    kwarg_keys =['sg','hs','negative','ns_exponent']
    kwargs = {k:params[k] for k in kwarg_keys}
    return kwargs

```
### SOURCE:
```python
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

```
### SOURCE:
```python
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

```
### SOURCE:
```python
def get_wv_from_word2vec(word2vec_model):
    """Checks if model is full word2vec or .wv attribute. 
    Returns wv."""
    import gensim 
    if isinstance(word2vec_model,gensim.models.word2vec.Word2Vec):
        wv = word2vec_model.wv
    elif isinstance(word2vec_model,gensim.models.keyedvectors.Word2VecKeyedVectors):
        wv = word2vec_model
    return wv

```
### SOURCE:
```python
def get_w2v_kwargs(params):
    ## get kwargs for make_word2vecmodel
    kwarg_keys =['sg','hs','negative','ns_exponent']
    kwargs = {k:params[k] for k in kwarg_keys}
    return kwargs

```
### SOURCE:
```python
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

```
### SOURCE:
```python
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

```
### SOURCE:
```python
def get_wv_from_word2vec(word2vec_model):
    """Checks if model is full word2vec or .wv attribute. 
    Returns wv."""
    import gensim 
    if isinstance(word2vec_model,gensim.models.word2vec.Word2Vec):
        wv = word2vec_model.wv
    elif isinstance(word2vec_model,gensim.models.keyedvectors.Word2VecKeyedVectors):
        wv = word2vec_model
    return wv

```
### SOURCE:
```python
def get_w2v_kwargs(params):
    ## get kwargs for make_word2vecmodel
    kwarg_keys =['sg','hs','negative','ns_exponent']
    kwargs = {k:params[k] for k in kwarg_keys}
    return kwargs

```
### SOURCE:
```python
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

```
### SOURCE:
```python
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

```
### SOURCE:
```python
def get_wv_from_word2vec(word2vec_model):
    """Checks if model is full word2vec or .wv attribute. 
    Returns wv."""
    import gensim 
    if isinstance(word2vec_model,gensim.models.word2vec.Word2Vec):
        wv = word2vec_model.wv
    elif isinstance(word2vec_model,gensim.models.keyedvectors.Word2VecKeyedVectors):
        wv = word2vec_model
    return wv

```
### SOURCE:
```python
def get_w2v_kwargs(params):
    ## get kwargs for make_word2vecmodel
    kwarg_keys =['sg','hs','negative','ns_exponent']
    kwargs = {k:params[k] for k in kwarg_keys}
    return kwargs

```
### SOURCE:
```python
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

```
### SOURCE:
```python
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

```
### SOURCE:
```python
def get_wv_from_word2vec(word2vec_model):
    """Checks if model is full word2vec or .wv attribute. 
    Returns wv."""
    import gensim 
    if isinstance(word2vec_model,gensim.models.word2vec.Word2Vec):
        wv = word2vec_model.wv
    elif isinstance(word2vec_model,gensim.models.keyedvectors.Word2VecKeyedVectors):
        wv = word2vec_model
    return wv

```
### SOURCE:
```python
def get_w2v_kwargs(params):
    ## get kwargs for make_word2vecmodel
    kwarg_keys =['sg','hs','negative','ns_exponent']
    kwargs = {k:params[k] for k in kwarg_keys}
    return kwargs

```
### SOURCE:
```python
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

```
### SOURCE:
```python
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

```
### SOURCE:
```python
def get_wv_from_word2vec(word2vec_model):
    """Checks if model is full word2vec or .wv attribute. 
    Returns wv."""
    import gensim 
    if isinstance(word2vec_model,gensim.models.word2vec.Word2Vec):
        wv = word2vec_model.wv
    elif isinstance(word2vec_model,gensim.models.keyedvectors.Word2VecKeyedVectors):
        wv = word2vec_model
    return wv

```
### SOURCE:
```python
def get_w2v_kwargs(params):
    ## get kwargs for make_word2vecmodel
    kwarg_keys =['sg','hs','negative','ns_exponent']
    kwargs = {k:params[k] for k in kwarg_keys}
    return kwargs

```
### SOURCE:
```python
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

```
### SOURCE:
```python
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

```
### SOURCE:
```python
def get_wv_from_word2vec(word2vec_model):
    """Checks if model is full word2vec or .wv attribute. 
    Returns wv."""
    import gensim 
    if isinstance(word2vec_model,gensim.models.word2vec.Word2Vec):
        wv = word2vec_model.wv
    elif isinstance(word2vec_model,gensim.models.keyedvectors.Word2VecKeyedVectors):
        wv = word2vec_model
    return wv

```
### SOURCE:
```python
def get_w2v_kwargs(params):
    ## get kwargs for make_word2vecmodel
    kwarg_keys =['sg','hs','negative','ns_exponent']
    kwargs = {k:params[k] for k in kwarg_keys}
    return kwargs

```
### SOURCE:
```python
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

```
### SOURCE:
```python
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

```
### SOURCE:
```python
def get_wv_from_word2vec(word2vec_model):
    """Checks if model is full word2vec or .wv attribute. 
    Returns wv."""
    import gensim 
    if isinstance(word2vec_model,gensim.models.word2vec.Word2Vec):
        wv = word2vec_model.wv
    elif isinstance(word2vec_model,gensim.models.keyedvectors.Word2VecKeyedVectors):
        wv = word2vec_model
    return wv

```
### SOURCE:
```python
def get_w2v_kwargs(params):
    ## get kwargs for make_word2vecmodel
    kwarg_keys =['sg','hs','negative','ns_exponent']
    kwargs = {k:params[k] for k in kwarg_keys}
    return kwargs

```
### SOURCE:
```python
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

```
### SOURCE:
```python
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

```
### SOURCE:
```python
def get_wv_from_word2vec(word2vec_model):
    """Checks if model is full word2vec or .wv attribute. 
    Returns wv."""
    import gensim 
    if isinstance(word2vec_model,gensim.models.word2vec.Word2Vec):
        wv = word2vec_model.wv
    elif isinstance(word2vec_model,gensim.models.keyedvectors.Word2VecKeyedVectors):
        wv = word2vec_model
    return wv

```
### SOURCE:
```python
def get_w2v_kwargs(params):
    ## get kwargs for make_word2vecmodel
    kwarg_keys =['sg','hs','negative','ns_exponent']
    kwargs = {k:params[k] for k in kwarg_keys}
    return kwargs

```
### SOURCE:
```python
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

```
### SOURCE:
```python
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

```
### SOURCE:
```python
def get_wv_from_word2vec(word2vec_model):
    """Checks if model is full word2vec or .wv attribute. 
    Returns wv."""
    import gensim 
    if isinstance(word2vec_model,gensim.models.word2vec.Word2Vec):
        wv = word2vec_model.wv
    elif isinstance(word2vec_model,gensim.models.keyedvectors.Word2VecKeyedVectors):
        wv = word2vec_model
    return wv

```
### SOURCE:
```python
def get_w2v_kwargs(params):
    ## get kwargs for make_word2vecmodel
    kwarg_keys =['sg','hs','negative','ns_exponent']
    kwargs = {k:params[k] for k in kwarg_keys}
    return kwargs

```
### SOURCE:
```python
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

```
### SOURCE:
```python
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

```
### SOURCE:
```python
def get_wv_from_word2vec(word2vec_model):
    """Checks if model is full word2vec or .wv attribute. 
    Returns wv."""
    import gensim 
    if isinstance(word2vec_model,gensim.models.word2vec.Word2Vec):
        wv = word2vec_model.wv
    elif isinstance(word2vec_model,gensim.models.keyedvectors.Word2VecKeyedVectors):
        wv = word2vec_model
    return wv

```
### SOURCE:
```python
def get_w2v_kwargs(params):
    ## get kwargs for make_word2vecmodel
    kwarg_keys =['sg','hs','negative','ns_exponent']
    kwargs = {k:params[k] for k in kwarg_keys}
    return kwargs

```
### SOURCE:
```python
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

```
### SOURCE:
```python
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

```
### SOURCE:
```python
def get_wv_from_word2vec(word2vec_model):
    """Checks if model is full word2vec or .wv attribute. 
    Returns wv."""
    import gensim 
    if isinstance(word2vec_model,gensim.models.word2vec.Word2Vec):
        wv = word2vec_model.wv
    elif isinstance(word2vec_model,gensim.models.keyedvectors.Word2VecKeyedVectors):
        wv = word2vec_model
    return wv

```
### SOURCE:
```python
def get_w2v_kwargs(params):
    ## get kwargs for make_word2vecmodel
    kwarg_keys =['sg','hs','negative','ns_exponent']
    kwargs = {k:params[k] for k in kwarg_keys}
    return kwargs

```
### SOURCE:
```python
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

```
### SOURCE:
```python
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

```
### SOURCE:
```python
def get_wv_from_word2vec(word2vec_model):
    """Checks if model is full word2vec or .wv attribute. 
    Returns wv."""
    import gensim 
    if isinstance(word2vec_model,gensim.models.word2vec.Word2Vec):
        wv = word2vec_model.wv
    elif isinstance(word2vec_model,gensim.models.keyedvectors.Word2VecKeyedVectors):
        wv = word2vec_model
    return wv

```
### SOURCE:
```python
def get_w2v_kwargs(params):
    ## get kwargs for make_word2vecmodel
    kwarg_keys =['sg','hs','negative','ns_exponent']
    kwargs = {k:params[k] for k in kwarg_keys}
    return kwargs

```
### SOURCE:
```python
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

```
### SOURCE:
```python
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

```
### SOURCE:
```python
def get_wv_from_word2vec(word2vec_model):
    """Checks if model is full word2vec or .wv attribute. 
    Returns wv."""
    import gensim 
    if isinstance(word2vec_model,gensim.models.word2vec.Word2Vec):
        wv = word2vec_model.wv
    elif isinstance(word2vec_model,gensim.models.keyedvectors.Word2VecKeyedVectors):
        wv = word2vec_model
    return wv

```
### SOURCE:
```python
def get_w2v_kwargs(params):
    ## get kwargs for make_word2vecmodel
    kwarg_keys =['sg','hs','negative','ns_exponent']
    kwargs = {k:params[k] for k in kwarg_keys}
    return kwargs

```
### SOURCE:
```python
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

```
### SOURCE:
```python
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

```
### SOURCE:
```python
def get_wv_from_word2vec(word2vec_model):
    """Checks if model is full word2vec or .wv attribute. 
    Returns wv."""
    import gensim 
    if isinstance(word2vec_model,gensim.models.word2vec.Word2Vec):
        wv = word2vec_model.wv
    elif isinstance(word2vec_model,gensim.models.keyedvectors.Word2VecKeyedVectors):
        wv = word2vec_model
    return wv

```
### SOURCE:
```python
def get_w2v_kwargs(params):
    ## get kwargs for make_word2vecmodel
    kwarg_keys =['sg','hs','negative','ns_exponent']
    kwargs = {k:params[k] for k in kwarg_keys}
    return kwargs

```
### SOURCE:
```python
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

```
### SOURCE:
```python
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

```
### SOURCE:
```python
def get_wv_from_word2vec(word2vec_model):
    """Checks if model is full word2vec or .wv attribute. 
    Returns wv."""
    import gensim 
    if isinstance(word2vec_model,gensim.models.word2vec.Word2Vec):
        wv = word2vec_model.wv
    elif isinstance(word2vec_model,gensim.models.keyedvectors.Word2VecKeyedVectors):
        wv = word2vec_model
    return wv

```
### SOURCE:
```python
def get_w2v_kwargs(params):
    ## get kwargs for make_word2vecmodel
    kwarg_keys =['sg','hs','negative','ns_exponent']
    kwargs = {k:params[k] for k in kwarg_keys}
    return kwargs

```
### SOURCE:
```python
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

```
### SOURCE:
```python
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

```
### SOURCE:
```python
def get_wv_from_word2vec(word2vec_model):
    """Checks if model is full word2vec or .wv attribute. 
    Returns wv."""
    import gensim 
    if isinstance(word2vec_model,gensim.models.word2vec.Word2Vec):
        wv = word2vec_model.wv
    elif isinstance(word2vec_model,gensim.models.keyedvectors.Word2VecKeyedVectors):
        wv = word2vec_model
    return wv

```
### SOURCE:
```python
def get_w2v_kwargs(params):
    ## get kwargs for make_word2vecmodel
    kwarg_keys =['sg','hs','negative','ns_exponent']
    kwargs = {k:params[k] for k in kwarg_keys}
    return kwargs

```
### SOURCE:
```python
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

```
### SOURCE:
```python
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

```
### SOURCE:
```python
def get_wv_from_word2vec(word2vec_model):
    """Checks if model is full word2vec or .wv attribute. 
    Returns wv."""
    import gensim 
    if isinstance(word2vec_model,gensim.models.word2vec.Word2Vec):
        wv = word2vec_model.wv
    elif isinstance(word2vec_model,gensim.models.keyedvectors.Word2VecKeyedVectors):
        wv = word2vec_model
    return wv

```
### SOURCE:
```python
def get_w2v_kwargs(params):
    ## get kwargs for make_word2vecmodel
    kwarg_keys =['sg','hs','negative','ns_exponent']
    kwargs = {k:params[k] for k in kwarg_keys}
    return kwargs

```
### SOURCE:
```python
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

```
### SOURCE:
```python
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

```
### SOURCE:
```python
def get_wv_from_word2vec(word2vec_model):
    """Checks if model is full word2vec or .wv attribute. 
    Returns wv."""
    import gensim 
    if isinstance(word2vec_model,gensim.models.word2vec.Word2Vec):
        wv = word2vec_model.wv
    elif isinstance(word2vec_model,gensim.models.keyedvectors.Word2VecKeyedVectors):
        wv = word2vec_model
    return wv

```
### SOURCE:
```python
def get_w2v_kwargs(params):
    ## get kwargs for make_word2vecmodel
    kwarg_keys =['sg','hs','negative','ns_exponent']
    kwargs = {k:params[k] for k in kwarg_keys}
    return kwargs

```
### SOURCE:
```python
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

```
