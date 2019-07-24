class Clock2(object):
    """A clock meant to be used as a timer for functions using local time.
    Clock.tic() starts the timer, .lap() adds the current laps time to clock._list_lap_times, .toc() stops the timer.
    If user initiializes with verbose =0, only start and final end times are displays.
        If verbose=1, print each lap's info at the end of each lap.
        If verbose=2 (default, display instruction line, return datafarme of results.)
    """
    
    from datetime import datetime
    from pytz import timezone
    from tzlocal import get_localzone
    from bs_ds import list2df
    # from bs_ds import list2df

    def get_time(self,local=True):
        """Returns current time, in local time zone by default (local=True)."""
        from datetime import datetime
        from pytz import timezone
        from tzlocal import get_localzone

        _now_utc_=datetime.now(timezone('UTC'))
        _now_local_=_now_utc_.astimezone(self._timezone_)
        if local==True:
            time_now = _now_local_

            return time_now#_now_local_
        else:
            return _now_utc_


    def __init__(self, display_final_time_as_minutes=True, verbose=2):

        from datetime import datetime
        from pytz import timezone
        from tzlocal import get_localzone

        self._strformat_ = []
        self._timezone_ = []
        self._timezone_ = get_localzone()
        self._start_time_ = []
        self._lap_label_ = []
        self._lap_end_time_ = []
        self._verbose_ = verbose
        self._lap_duration_ = []
        self._verbose_ = verbose
        self._prior_start_time_ = []
        self._display_as_minutes_ = display_final_time_as_minutes

        strformat = "%m/%d/%y - %I:%M:%S %p"
        self._strformat_ = strformat

    #         if self._verbose_ > 0:
    #             print(f'Clock created at {self.get_time().strftime(strformat)}.')

    #         if self._verbose_>1:
    #             print(f'\tStart: clock.tic()\tMark lap: clock.lap()\tStop: c_lap_times_list_lock.toc()\n')



    def mark_lap_list(self, label=None):
        """Used internally, appends the current laps' information when called by .lap()
        self._lap_times_list_ = [['Lap #' , 'Start Time','Stop Time', 'Stop Label', 'Duration']]"""
        import bs_ds as bs
#         print(self._prior_start_time_, self._lap_end_time_)
        if label is None:
            label='--'

        duration = self._lap_duration_.total_seconds()
        self._lap_times_list_.append([ self._lap_counter_ , # Lap #
                                      (self._prior_start_time_).strftime(self._strformat_), # This Lap's Start Time
                                      self._lap_end_time_,#.strftime(self._strformat_), # stop clock time
                                      label,#self._lap_label_, # The Label passed with .lap()
                                      f'{duration:.3f} sec']) # the lap duration


    def tic(self, label=None ):
        "Start the timer and display current time, appends label to the _list_lap_times."
        from datetime import datetime
        from pytz import timezone

        self._start_time_ = self.get_time()
        self._start_label_ = label
        self._lap_counter_ = 0
        self._prior_start_time_=self._start_time_
        self._lap_times_list_=[]

        # Initiate lap counter and list
        self._lap_times_list_ = [['Lap #','Start Time','Stop Time', 'Label', 'Duration']]
        self._lap_counter_ = 0
        self._decorate_ = '--- '
        decorate=self._decorate_
        base_msg = f'{decorate}CLOCK STARTED @: {self._start_time_.strftime(self._strformat_):>{25}}'
        
        if label == None:
            display_msg = base_msg+' '+ decorate
            label='--'
        else:
            spacer = ' '
            display_msg = base_msg+f'{spacer:{10}} Label: {label:{10}} {decorate}'
        if self._verbose_>0:
            print(display_msg)#f'---- Clock started @: {self._start_time_.strftime(self._strformat_):>{25}} {spacer:{10}} label: {label:{20}}  ----')

    def toc(self,label=None, summary=True):
        """Stop the timer and displays results, appends label to final _list_lap_times entry"""
        if label == None:
            label='--'
        from datetime import datetime
        from pytz import timezone
        from tzlocal import get_localzone
        from bs_ds import list2df
        if label is None:
            label='--'

        _final_end_time_ = self.get_time()
        _total_time_ = _final_end_time_ - self._start_time_
        _end_label_ = label

        self._lap_counter_+=1
        self._final_end_time_ = _final_end_time_
        self._lap_label_=_end_label_
        self._lap_end_time_ = _final_end_time_.strftime(self._strformat_)
        self._lap_duration_ = _final_end_time_ - self._prior_start_time_
        self._total_time_ = _total_time_
        self.mark_lap_list(label=label)
        decorate=self._decorate_
        # Append Summary Line
        # print(f'Lap #{self._lap_counter_} done @ {self._lap_end_time_:>{20}} label: {self._lap_label_:>{10}} duration: {self._lap_duration_.total_seconds()} sec)')
        # total_time_to_display = self._total_time_.total_seconds()
        if self._display_as_minutes_ == True:
            total_seconds = self._total_time_.total_seconds()
            total_mins = int(total_seconds // 60)
            sec_remain = total_seconds % 60
            total_time_to_display = f'{total_mins} min, {sec_remain:.3f} sec'
        else:

            total_seconds = self._total_time_.total_seconds()
            sec_remain = round(total_seconds % 60,3)

            total_time_to_display = f'{sec_remain} sec'

        # self._lap_times_list_.append(['TOTAL',self._start_time_.strftime(self._strformat_), self._final_end_time_.strftime(self._strformat_),total_time_to_display]) #'Total Time: ', total_time_to_display])

        # print(self._lap_times_list_[-1])
        # print('')
        if self._verbose_>0:
            print(f'--- TOTAL DURATION   =  {total_time_to_display:>{15}} {decorate}')
        
        if summary:
            self.summary()
            # df_lap_times = list2df(self._lap_times_list_)#,index_col='Lap #')
            # return df_lap_times.style.hide_index()



    def lap(self, label=None):
        """Records time, duration, and label for current lap. Output display varies with clock verbose level.
        Calls .mark_lap_list() to document results in clock._list_lap_ times."""
        from datetime import datetime
        if label is None:
            label='--'
        _end_time_ = self.get_time()

        # Append the lap attribute list and counter
        self._lap_label_ = label
        self._lap_end_time_ = _end_time_.strftime(self._strformat_)
        self._lap_counter_+=1
        self._lap_duration_ = (_end_time_ - self._prior_start_time_)
        # Now update the record
        self.mark_lap_list()

        # Now set next lap's new _prior_start
        self._prior_start_time_=_end_time_
        spacer = ' '

        if self._verbose_>0:
            print(f'       - Lap # {self._lap_counter_} @:  {self._lap_end_time_:>{25}} {spacer:{5}} Dur: {self._lap_duration_.total_seconds():.3f} sec. {spacer:{5}}Label:  {self._lap_label_:{20}}')
    
    def summary(self):
        from bs_ds import list2df
        import pandas as pd
        from IPython.display import display
        df_lap_times = list2df(self._lap_times_list_)#,index_col='Lap #')
        df_lap_times.drop('Stop Time',axis=1,inplace=True)
        df_lap_times = df_lap_times[['Lap #','Start Time','Duration','Label']]
        # with pd.option_context('display.colheader_justify','left'):
        dfs = df_lap_times.style.hide_index().set_caption('Summary Table of Clocked Processes').set_properties(subset=['Start Time','Duration'],**{'width':'140px'})
        # display(dfs.set_table_styles([dict(selector='th', props=[('text-align', 'center')])]))
        display(dfs.set_table_styles([dict(selector='table, th', props=[('text-align', 'center')])]))

