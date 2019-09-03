def evaluate_regression_model(model, history, train_generator, test_generator,true_train_series,
true_test_series,include_train_data=True,return_preds_df = False, save_history=False, history_filename ='results/keras_history.png', save_summary=False, 
                            summary_filename = 'results/model_summary.txt',auto_unique_filenames=True):

    """Evaluates kera's model's performance, plots model's history,displays classification report,
    and plots a confusion matrix. 
    conf_matrix_classes are the labels for the matrix. [negative, positive]
    Returns df of classification report and fig object for  confusion matrix's plot."""

    from sklearn.metrics import roc_auc_score, roc_curve, classification_report,confusion_matrix
    import bs_ds as bs
    import functions_combined_BEST as ji
    from IPython.display import display
    import pandas as pd
    import matplotlib as mpl
    numFmt = '.4f'
    num_dashes = 30

    # results_list=[['Metric','Value']]
    # metric_list = ['accuracy','precision','recall','f1']
    print('---'*num_dashes)
    print('\tTRAINING HISTORY:')
    print('---'*num_dashes)

    if auto_unique_filenames:
        ## Get same time suffix for all files
        time_suffix = ji.auto_filename_time(fname_friendly=True)

        filename_dict= {'history':history_filename,'summary':summary_filename}
        ## update filenames 
        for filetype,filename in filename_dict.items():
            if '.' in filename:
                filename_dict[filetype] = filename.split('.')[0]+time_suffix + '.'+filename.split('.')[-1]
            else:
                if filetype =='summary':
                    ext='.txt'
                else:
                    ext='.png'
                filename_dict[filetype] = filename+time_suffix + ext


        history_filename = filename_dict['history']
        summary_filename = filename_dict['summary']


    ## PLOT HISTORY
    ji.plot_keras_history( history,filename_base=history_filename, save_fig=save_history,title_text='')

    print('\n')
    print('---'*num_dashes)
    print('\tEVALUATE MODEL:')
    print('---'*num_dashes)

        # # EVALUATE MODEL PREDICTIONS FROM GENERATOR 
    print('Evaluating Train Generator:')
    model_metrics_train = model.evaluate_generator(train_generator,verbose=1)
    print('Evaluating Test Generator:')
    model_metrics_test = model.evaluate_generator(test_generator,verbose=1)


    x_window = test_generator.length
    n_features = test_generator.data[0].shape[0]
    gen_df = ji.get_model_preds_from_gen(model=model, test_generator=test_generator,true_test_data=true_test_series,
        n_input=x_window, n_features=n_features,  suffix='_from_gen',return_df=True)

    regr_results = ji.evaluate_regression_model(y_true=gen_df['true_from_gen'], y_test=gen_df['pred_from_gen'],
                                metrics=['r2', 'RMSE', 'U'], show_results_df=True, return_as_df=True)


    if save_summary:
        with open(summary_filename,'w') as f:
            model.summary(print_fn=lambda x: f.write(x+"\n"))
            f.write(f"\nSaved at {time_suffix}\n")


    if include_train_data:
        true_train_series=true_train_series.rename('true_train_price')
        df_all_preds=pd.concat([true_train_series,gen_df],axis=1)
    else:
        df_all_preds = gen_df

    if return_preds_df:
        return df_all_preds


