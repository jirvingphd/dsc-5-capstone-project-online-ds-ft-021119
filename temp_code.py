
## FROM YELP API LAB _ EXAMPLE PAGINATIONS
def page_results(twtiter_df,iloc_index,n,page,displayed_index=None):
    num = len(idx_search)
    print(f'{num} total matches found.')#.format(num))
    dfs = []

    if displayed_index is not None:
        last_displayed = displayed_index.iloc[-1]

        cur = iloc_index.loc[]
    else:
        cur = 0


    while cur < num: #and cur < nun:
        # url_params['offset'] = cur
        idx_show = idx_search.iloc[cur:cur+n]
        dfs.append(yelp_call(url_params, api_key))
        time.sleep(1) #Wait a second
        cur += 50
    df = pd.concat(dfs, ignore_index=True)
    return df


##
def display_results_as_md_df(tweet,as_df=False,as_md=True):
    if as_df == False and as_md==False:

            ## for each tweet:
            for i in range(num_tweets):

                print(f'\n>> TWEETED ON {print_index[i] }') ##tweet.index[i]}')
                if is_series:
                    tweet_to_print = tweet
                else:
                    tweet_to_print=tweet.loc[print_index[i]]

                # print each column for dataframes
                for col in columns:
                    print(f"\n\t[col='{col}']:")
                    # col_text = tweet.loc[col] 
                    print('\t\t"',tweet_to_print[col],'"')

                if num_tweets>1:
                    print('\n','---'*10)


        elif as_md:

            for ii in range(len(print_index)):#range(num_tweets):
                i = print_index[ii]
                tweet_to_print = df_sampled.loc[i]

                if num_tweets>1:
                    display(Markdown("<br>"))
                    display(Markdown(f'### Tweet #{ii+1} of {len(print_index)}: sent @ *{i}*:'))#index[i]
                else:
                    display(Markdown(f'#### TWEET FROM {i}:'))#index[i]
                # print(f'* TWEETED ON {df_sampled.index[i]}')
                for col in columns:

                    col_name = f'* **["{col}"] column:**<p><blockquote>***"{tweet_to_print[col]}"***'
                    
                    display(Markdown(col_name))

                
        elif as_df:

            # for i in range(num_tweets):
            for ii in range(len(print_index)):#range(num_tweets):
                i = print_index[ii]
                tweet_to_print = df_sampled.loc[i]

                df = pd.DataFrame(columns=['tweet'],index=columns)
                df.index.name = 'column'
                for col in columns:
                    df['tweet'].loc[col] = tweet_to_print[col]#[i]
                
                with pd.option_context('display.max_colwidth',0, 'display.colheader_justify','left'):#,'colheader_justify','left'):
                    # caption = f'Tweet #{ii+1}  = {i}'
                    capt_text  = f'Tweet #{ii+1} of {len(print_index)}: sent @ *{i}*:'
                    table_style =[{'selector':'caption',
                    'props':[('font-size','1.2em'),('color','darkblue'),('font-weight','bold'),
                    ('vertical-align','0%')]}]
                    dfs = df.style.set_caption(capt_text).set_properties(subset=['tweet'],
                    **{'width':'600px',
                    'text-align':'center',
                    'padding':'1em',
                    'font-size':'1.2em'}).set_table_styles(table_style)
                    display(dfs)
        return 