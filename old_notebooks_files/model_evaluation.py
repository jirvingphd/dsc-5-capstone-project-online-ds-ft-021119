

# ### LEARNCO BUILDING TREES SKLEARN
# acc = accuracy_score(y_test,y_pred) * 100
# print("Accuracy is :{0}".format(acc))

# # Check the AUC for predictions
# false_positive_rate, true_positive_rate, thresholds = roc_curve(y_test, y_pred)
# roc_auc = auc(false_positive_rate, true_positive_rate)
# print("\nAUC is :{0}".format(round(roc_auc,2)))

# # Create and print a confusion matrix 
# print('\nConfusion Matrix')
# print('----------------')
# pd.crosstab(y_test, y_pred, rownames=['True'], colnames=['Predicted'], margins=True)
# ####


## James' Tree Classifier/Regressor

# def tune_params_trees (helpers: performance_r2_mse, performance_roc_auc)
def performance_r2_mse(y_true, y_pred):
    """ Calculates and returns the performance score between
        true and predicted values based on the metric chosen. """
    from sklearn.metrics import r2_score
    from sklearn.metrics import mean_squared_error as mse

    r2 = r2_score(y_true,y_pred)
    MSE = mse(y_true,y_pred)
    return r2, MSE

# def performance_roc_auc(X_test,y_test,dtc,verbose=False):
def performance_roc_auc(y_true,y_pred):
    """Tests the results of an already-fit classifer.
    Takes y_true (test split), and y_pred (model.predict()), returns the AUC for the roc_curve as a %"""
    from sklearn.metrics import roc_curve, auc
    FP_rate, TP_rate, _ = roc_curve(y_true,y_pred)
    roc_auc = auc(FP_rate,TP_rate)
    roc_auc_perc = round(roc_auc*100,3)
    return roc_auc_perc

def tune_params_trees(param_name, param_values, DecisionTreeObject, X,Y,test_size=0.25,perform_metric='r2_mse'):
    '''Test Decision Tree Regressor or Classifier parameter with the values in param_values
     Displays color-coed dataframe of perfomance results and subplot line graphs.
    Parameters:
        parame_name (str)
            name of parameter to test with values param_values
        param_values (list/array),
            list of parameter values to try
        DecisionTreeObject,
            Existing DecisionTreeObject instance.
        perform_metric
            Can either 'r2_mse' or 'roc_auc'.

    Returns:
    - df of results
    - Displays styled-df
    - Displays subplots of performance metrics.
    '''

    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(test_size=test_size)

    # Create results depending on performance metric
    if perform_metric=='r2_mse':
        results = [['param_name','param_value','r2_test','MSE_test']]

    elif perform_metric=='roc_auc':
        results =  [['param_name','param_value','roc_auc_test']]
    print(f'Using performance metrics: {perform_metric}')

    # Rename Deicision Tree for looping
    dtr_tune =  DecisionTreeObject

    # Loop through each param_value
    for value in param_values:

        # Set the parameters and fit the model
        dtr_tune.set_params(**{param_name:value})
        dtr_tune.fit(X_train,y_train)

        # Get predicitons and test_performance
        y_preds = dtr_tune.predict(X_test)

        # Perform correct performance metric and append results
        if perform_metric=='r2_mse':

            r2_test, mse_test = performance_r2_mse(y_test,y_preds)
            results.append([param_name,value,r2_test,mse_test])

        elif perform_metric=='roc_auc':

            roc_auc_test = performance_roc_auc(y_test,y_preds)
            results.append([param_name,value,roc_auc_test])


    # Convert results to dataframe, set index
    df_results = list2df(results)
    df_results.set_index('param_value',inplace=True)


    # Plot the values in results
    df_results.plot(subplots=True,sharex=True)

    # Style dataframe for easy visualization
    import seaborn as sns
    cm = sns.light_palette("green", as_cmap=True)
    df_style = df_results.style.background_gradient(cmap=cm,subset=['r2_test','MSE_test'])#,low=results.min(),high=results.max())
    # Display styled dataframe
    from IPython.display import display
    display(df_style)

    return df_results


# Display graphviz tree
def viz_tree(tree_object):
    '''Takes a Sklearn Decision Tree and returns a png image using graph_viz and pydotplus.'''
    # Visualize the decision tree using graph viz library
    from sklearn.externals.six import StringIO
    from IPython.display import Image
    from sklearn.tree import export_graphviz
    import pydotplus
    dot_data = StringIO()
    export_graphviz(tree_object, out_file=dot_data, filled=True, rounded=True,special_characters=True)
    graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
    tree_viz = Image(graph.create_png())
    return tree_viz









## classification model framework

# clock = bs.Clock(verbose=0)
# print('---'*40)
# print('\tFITTING MODEL:')
# print('---'*40,'\n')

# clock.tic('starting keras .fit')

# # num_epochs = 4
# # history = model2.fit(X_train, y_train, epochs=num_epochs, verbose=True, validation_split=0.1,
# #                      callbacks=callbacks,batch_size=300)#, validation_data=(X_val))

# clock.toc(f'completed {num_epochs} epochs')
def evaluate_classification_model(model,X_train, y_train, X_test, y_test):
    print('\n')
    print('---'*40)
    print('\tEVALUATE MODEL:')
    print('---'*40)

    from sklearn.metrics import roc_auc_score, roc_curve, classification_report

    loss, accuracy = model2.evaluate(X_train, y_train, verbose=True)
    print(f'Training Accuracy:{accuracy}')

    loss, accuracy = model2.evaluate(X_test, y_test, verbose=True)
    print(f'Testing Accuracy:{accuracy}\n')

    y_hat_test = model2.predict_classes(X_test)
    print('---'*40)
    print('CLASSIFICATION REPORT:')
    print('---'*40)

    print(classification_report(y_test,y_hat_test))

    from sklearn.metrics import confusion_matrix
    conf_mat = confusion_matrix(y_test, y_hat_test)

    import bs_ds as bs

    mpl.rcParams['figure.figsize'] = (8,4)
    bs.plot_confusion_matrix(conf_mat,classes=['Stock Increase','Stock Decrease'])
