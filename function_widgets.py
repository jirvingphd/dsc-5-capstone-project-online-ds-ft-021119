# def make_model_params_df(model_params, multi_index=True):
#     pass
#     import bs_ds as bs
#     import functions_combined_BEST as ji
#     import pandas as pd

#     # data_fields = 
#     model_layer_list=model_config_dict['layers']
#     output = [['#','layer_name', 'layer_config_level','layer_param','param_value']]#,'param_sub_value','param_sub_value_details' ]]

#     for num,layer_dict in enumerate(model_layer_list):


def get_model_config_df(model1, multi_index=True):

    import bs_ds as bs
    import functions_combined_BEST as ji
    import pandas as pd
    pd.set_option('display.max_rows',None)

    model_config_dict = model1.get_config()
    model_layer_list=model_config_dict['layers']
    output = [['#','layer_name', 'layer_config_level','layer_param','param_value']]#,'param_sub_value','param_sub_value_details' ]]

    for num,layer_dict in enumerate(model_layer_list):
    #     layer_dict = model_layer_list[0]


        # layer_dict['config'].keys()
        # config_keys = list(layer_dict.keys())
        # combine class and name into 1 column
        layer_class = layer_dict['class_name']
        layer_name = layer_dict['config'].pop('name')

        # col_000 = f"{num}: {layer_class}"
        # col_00 = layer_name#f"{layer_class} ({layer_name})"

        # get layer's config dict
        layer_config = layer_dict['config']


        # config_keys = list(layer_config.keys())


        # for each parameter in layer_config
        for param_name,col2_v_or_dict in layer_config.items():
            # col_1 is the key( name of param)
        #     col_1 = param_name

            
            col_000 = f"{num}: {layer_class}"

            ### DETERMINE LAYER_NAME WITH UNITS OF 
            if 'units' in layer_config.keys():
                units = layer_config['units'] #col2_v_or_dict
                col_00 = layer_name+' ('+str(units)+' units)'
                
            elif 'batch_input_shape' in layer_config.keys():
                input_length =  layer_config['input_length']
                output_dim =  layer_config['output_dim']
                col_00 = layer_name+' \n('+str(input_length)+' words, '+str(output_dim)+')'
            else:
                col_00 = layer_name#+' '+f"({}"#f"{layer_class} ({layer_name})"

            # check the contents of col2_:

            # if list, append col2_, fill blank cols
            if isinstance(col2_v_or_dict,dict)==False:


                col_0 = 'top-level'
                col_1 = param_name
                col_2 = col2_v_or_dict

                output.append([col_000,col_00,col_0,col_1 ,col_2])#,col_3,col_4])


            # else, set col_2 as the param name,
            if isinstance(col2_v_or_dict,dict):

                param_sub_type = col2_v_or_dict['class_name']
                col_0 = param_name +'  ('+param_sub_type+'):'

                # then loop through keys,vals of col_2's dict for col3,4
                param_dict = col2_v_or_dict['config']

                for sub_param,sub_param_val in param_dict.items():
                    col_1 =sub_param
                    col_2 = sub_param_val
                    # col_3 = ''


                    output.append([col_000,col_00,col_0, col_1 ,col_2])#,col_3,col_4])
        
    df = bs.list2df(output)    
    if multi_index==True:
        df.sort_values(by=['#','layer_config_level'], ascending=False,inplace=True)
        df.set_index(['#','layer_name','layer_config_level','layer_param'],inplace=True) #=pd.MultiIndex()
        df.sort_index(level=0, inplace=True)
    return df



# def view(df=df,layer_num='',param_level=''):
#             import pandas as pd
#             idx = pd.IndexSlice

#             if layer_num=='All': 
#                 # df = df.sort_index(by='#')
#                 if param_level=='All':
#                     # return display(df.sort_index(by='#'))
#                     # return display(df.sort_index(by='#'))
#                     df_out=df
#                 else:
#                     # return display(df.loc[idx[:,:,param_level],:])#display(df.xs(param_level,level=2).sort_index(by='#'))
#                     # return display(df.loc[idx[:,:,param_level],:].sort_index(by='#'))#display(df.xs(param_level,level=2).sort_index(by='#'))
#                     df_out = df.loc[idx[:,:,param_level],:]

#             else:
#                 if param_level=='All':
#                     # return display(df.loc[idx[layer_num,:,:],:])#display(df.xs(layer_num,level=0))
#                     # return display(df.loc[idx[layer_num,:,:],:].sort_index(by='#'))#display(df.xs(layer_num,level=0))
#                     df_out = df.loc[idx[layer_num,:,:],:]
#                 else:
#                     # return display(df.loc[idx[layer_num,:, param_level],:])#display(df.loc[layer][:][level]) #[df.xs(layer)])
#                     # return display(df.loc[idx[layer_num,:, param_level],:].sort_index(by='#'))#display(df.loc[layer][:][level]) #[df.xs(layer)])
#                     df_out = df.loc[idx[layer_num,:, param_level],:]

#             display(df_out.sort_index(by='#').style.set_caption('Model Layer Parameters'))
#             return df_out

## REFERNCE FOR CONTENTS OF CONFIG (for writing function below)
def make_model_menu(model1, multi_index=True):
    import bs_ds as bs
    import functions_combined_BEST as ji
    import pandas as pd
    from IPython.display import display
    import ipywidgets as widgets
    from ipywidgets import interact, interactive, interactive_output
    
    def get_model_config_df(model1, multi_index=True):
        model_config_dict = model1.get_config()
        model_layer_list=model_config_dict['layers']
        output = [['#','layer_name', 'layer_config_level','layer_param','param_value']]#,'param_sub_value','param_sub_value_details' ]]

        for num,layer_dict in enumerate(model_layer_list):
        #     layer_dict = model_layer_list[0]


            # layer_dict['config'].keys()
            # config_keys = list(layer_dict.keys())
            # combine class and name into 1 column
            layer_class = layer_dict['class_name']
            layer_name = layer_dict['config'].pop('name')
            col_000 = f"{num}: {layer_class}"
            col_00 = layer_name#f"{layer_class} ({layer_name})"

            # get layer's config dict
            layer_config = layer_dict['config']


            # config_keys = list(layer_config.keys())


            # for each parameter in layer_config
            for param_name,col2_v_or_dict in layer_config.items():
                # col_1 is the key( name of param)
            #     col_1 = param_name


                # check the contents of col2_:

                # if list, append col2_, fill blank cols
                if isinstance(col2_v_or_dict,dict)==False:
                    col_0 = 'top-level'
                    col_1 = param_name
                    col_2 = col2_v_or_dict

                    output.append([col_000,col_00,col_0,col_1 ,col_2])#,col_3,col_4])


                # else, set col_2 as the param name,
                if isinstance(col2_v_or_dict,dict):

                    param_sub_type = col2_v_or_dict['class_name']
                    col_0 = param_name +'  ('+param_sub_type+'):'

                    # then loop through keys,vals of col_2's dict for col3,4
                    param_dict = col2_v_or_dict['config']

                    for sub_param,sub_param_val in param_dict.items():
                        col_1 =sub_param
                        col_2 = sub_param_val
                        col_3 = ''


                        output.append([col_000,col_00,col_0, col_1 ,col_2])#,col_3,col_4])
            
        df = bs.list2df(output)    
        if multi_index==True:
            df.sort_values(by=['#','layer_config_level'], ascending=False,inplace=True)
            df.set_index(['#','layer_name','layer_config_level','layer_param'],inplace=True) #=pd.MultiIndex()
        return df


    # https://blog.ouseful.info/2016/12/29/simple-view-controls-for-pandas-dataframes-using-ipython-widgets/
    def model_layer_config_menu(df):
        import ipywidgets as widgets
        from IPython.display import display
        from ipywidgets import interact, interactive
        # from IPython.html.widgets import interactive


        ## SOLUION for getting values https://stackoverflow.com/questions/53927460/select-rows-in-pandas-multiindex-dataframe
        layer_names = pd.MultiIndex.get_level_values(df.index,level=0).unique().to_list()
        param_level = pd.MultiIndex.get_level_values(df.index, level=2).unique().to_list()

        # items = ['All']+sorted(df['layer_name'].unique().tolist())
        # layer_names.append('All') # 'All'+layer_names[:]#+param_level]
        layer_names=sorted(layer_names)
        layer_names.append('All')

        layer_levels = param_level
        layer_levels.append('All')
        
        
        def view(df=df,layer_num='',param_level=''):
            import pandas as pd
            idx = pd.IndexSlice

            if layer_num=='All': 
                # df = df.sort_index(by='#')
                if param_level=='All':
                    # return display(df.sort_index(by='#'))
                    # return display(df.sort_index(by='#'))
                    df_out=df
                else:
                    # return display(df.loc[idx[:,:,param_level],:])#display(df.xs(param_level,level=2).sort_index(by='#'))
                    # return display(df.loc[idx[:,:,param_level],:].sort_index(by='#'))#display(df.xs(param_level,level=2).sort_index(by='#'))
                    df_out = df.loc[idx[:,:,param_level],:]

            else:
                if param_level=='All':
                    # return display(df.loc[idx[layer_num,:,:],:])#display(df.xs(layer_num,level=0))
                    # return display(df.loc[idx[layer_num,:,:],:].sort_index(by='#'))#display(df.xs(layer_num,level=0))
                    df_out = df.loc[idx[layer_num,:,:],:]
                else:
                    # return display(df.loc[idx[layer_num,:, param_level],:])#display(df.loc[layer][:][level]) #[df.xs(layer)])
                    # return display(df.loc[idx[layer_num,:, param_level],:].sort_index(by='#'))#display(df.loc[layer][:][level]) #[df.xs(layer)])
                    df_out = df.loc[idx[layer_num,:, param_level],:]

            display(df_out.sort_index(by='#').style.set_caption('Model Layer Parameters'))
            return df_out

            
        w = widgets.Select(options=layer_names,value='All',description='Layer #')
        # interactive(view,layer=w)

        w2 = widgets.Select(options=layer_levels,value='All',desription='Level')
        # interactive(view,layer=w,level=w2)

        out= widgets.interactive_output(view,{'layer_num':w,'param_level':w2})
        return widgets.VBox([widgets.HBox([w,w2]),out])
    
    ## APPLYING FUNCTIONS
    df = get_model_config_df(model1,multi_index=True)

    return model_layer_config_menu(df)

# interactive(view, Menu) #layer=Menu.children[0],level=Menu.children[1])

# df.head()
def make_qgrid_model_menu(model, return_df = False):

    df=get_model_config_df(model)
    import qgrid
    from IPython.display import display
    import pandas as pd

    pd.set_option('display.max_rows',None)

    qgrid_menu = qgrid.show_grid(df,  grid_options={'highlightSelectedCell':True}, show_toolbar=True)
    
    display(qgrid_menu)
    if return_df:
        return df
    else:
        return 





# class VariableInspectorWindow(object):
#     instance = None

#     ### WAS OUTSIDE CLASS ORIGINALLY
#     ### https://github.com/jupyter-widgets/ipywidgets/blob/master/docs/source/examples/Variable%20Inspector.ipynb
#     import ipywidgets as widgets # Loads the Widget framework.
#     from IPython.core.magics.namespace import NamespaceMagics # Used to query namespace.
#     from IPython import get_ipython

#     # For this example, hide these names, just to avoid polluting the namespace further
#     get_ipython().user_ns_hidden['widgets'] = widgets
#     get_ipython().user_ns_hidden['NamespaceMagics'] = NamespaceMagics 


#     def __init__(self, ipython):
#         """Public constructor."""
#         if VariableInspectorWindow.instance is not None:
#             raise Exception("""Only one instance of the Variable Inspector can exist at a 
#                 time.  Call close() on the active instance before creating a new instance.
#                 If you have lost the handle to the active instance, you can re-obtain it
#                 via `VariableInspectorWindow.instance`.""")
#         from IPython.core.magics.namespace import NamespaceMagics # Used to query namespace.
#         from IPython import get_ipython
#         import ipywidgets as widgets

#         get_ipython().user_ns_hidden['widgets'] = widgets
#         get_ipython().user_ns_hidden['NamespaceMagics'] = NamespaceMagics 
#         VariableInspectorWindow.instance = self
#         self.closed = False
#         self.namespace = NamespaceMagics()
#         self.namespace.shell = ipython.kernel.shell
        
#         self._box = widgets.Box()
#         self._box.layout.overflow_y = 'scroll'
#         self._table = widgets.HTML(value = 'Not hooked')
#         self._box.children = [self._table]
        
#         self._ipython = ipython
#         self._ipython.events.register('post_run_cell', self._fill)
        
#     def close(self):
#         """Close and remove hooks."""
#         if not self.closed:
#             self._ipython.events.unregister('post_run_cell', self._fill)
#             self._box.close()
#             self.closed = True
#             VariableInspectorWindow.instance = None

#     def _fill(self):
#         """Fill self with variable information."""
#         values = self.namespace.who_ls()
#         self._table.value = '<div class="rendered_html jp-RenderedHTMLCommon"><table><thead><tr><th>Name</th><th>Type</th><th>Value</th></tr></thead><tr><td>' + \
#             '</td></tr><tr><td>'.join(['{0}</td><td>{1}</td><td>{2}'.format(v, type(eval(v)).__name__, str(eval(v))) for v in values]) + \
#             '</td></tr></table></div>'

#     def _ipython_display_(self):
#         """Called when display() or pyout is used to display the Variable 
#         Inspector."""
#         self._box._ipython_display_()


# def inspect_variables():
#     ### https://github.com/jupyter-widgets/ipywidgets/blob/master/docs/source/examples/Variable%20Inspector.ipynb
#     import ipywidgets as widgets # Loads the Widget framework.
#     from IPython.core.magics.namespace import NamespaceMagics # Used to query namespace.
#     from Ipython import get_ipython
#     # For this example, hide these names, just to avoid polluting the namespace further
#     get_ipython().user_ns_hidden['widgets'] = widgets
#     get_ipython().user_ns_hidden['NamespaceMagics'] = NamespaceMagics 
#     inspector = VariableInspectorWindow(get_ipython())
#     return inspector



def display_df_dict_menu(dict_to_display, selected_key=None):
    import ipywidgets as widgets
    from IPython.display import display
    from ipywidgets import interact, interactive
    import pandas as pd


    key_list = list(dict_to_display.keys())
    key_list.append('_All_')

    if selected_key is not None:
        selected_key = selected_key
        
    def view(eval_dict=dict_to_display,selected_key=''):
        
        from IPython.display import display
        from pprint import pprint
        
        if selected_key=='_All_':
            
            key_list = list(eval_dict.keys())
            outputs=[]
            
            for k in key_list:
                
                if type(eval_dict[k]) == pd.DataFrame:
                    outputs.append(eval_dict[k])
                    display(eval_dict[k].style.set_caption(k))
                else:
                    outputs.append(f"{k}:\n{eval_dict[k]}\n\n")
                    pprint('\n',eval_dict[k])
                
            return outputs#pprint(outputs)

        else:
                k = selected_key
#                 if type(eval_dict(k)) == pd.DataFrame:
                if type(eval_dict[k]) == pd.DataFrame:
                     display(eval_dict[k].style.set_caption(k))
                else:
                    pprint(eval_dict[k])
                return [eval_dict[k]]

    w= widgets.Dropdown(options=key_list,value='_All_', description='Key Word')
    
    # old, simple
    out = widgets.interactive_output(view, {'selected_key':w})
    
    
    # new, flashier
    output = widgets.Output(layout={'border': '1px solid black'})
    if type(out)==list:
        output.append_display_data(out)
#         out =widgets.HBox([x for x in out])
    else:
        output = out
#     widgets.HBox([])

    return widgets.VBox([widgets.HBox([w]),output])#out])