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


    def __init__(self, verbose=2):

        from datetime import datetime
        from pytz import timezone
        from tzlocal import get_localzone

        self._strformat_ = []
        self._timezone_ = []
        self._timezone_ = get_localzone()
        self._start_time_ = []
        self._lap_label_ = []
        self._lap_end_time_ = []
        self._verbose_ = []
        self._lap_duration_ = []
        self._verbose_ = verbose
        self._prior_start_time_ = []

        strformat = "%m/%d/%y - %I:%M:%S %p"
        self._strformat_ = strformat

    #         if self._verbose_ > 0:
    #             print(f'Clock created at {self.get_time().strftime(strformat)}.')

    #         if self._verbose_>1:
    #             print(f'\tStart: clock.tic()\tMark lap: clock.lap()\tStop: clock.toc()\n')



    def mark_lap_list(self, label=None):
        """Used internally, appends the current laps' information when called by .lap()
        self._lap_times_list_ = [['Lap #' , 'Start Time','Start Label','Stop Time', 'Stop Label', 'Duration']]"""
        import bs_ds as bs
#         print(self._prior_start_time_, self._lap_end_time_)
        self._lap_times_list_.append([ self._lap_counter_ , # Lap #
                                      (self._prior_start_time_).strftime(self._strformat_), # This Lap's Start Time
                                      self._start_label_, # the start label for tic
                                      self._lap_end_time_,#.strftime(self._strformat_), # stop clock time
                                      self._lap_label_, # The Label passed with .lap()
                                      self._lap_duration_.total_seconds()]) # the lap duration


    def tic(self, label=[] ):
        "Start the timer and display current time, appends label to the _list_lap_times."
        from datetime import datetime
        from pytz import timezone

        self._start_time_ = self.get_time()
        self._start_label_ = label
        self._lap_counter_ = 0
        self._prior_start_time_=self._start_time_
        self._lap_times_list_=[]

        # Initiate lap counter and list
        self._lap_times_list_ = [['Lap #','Start Time','Start Label','Stop Time', 'Stop Label', 'Duration']]
        self._lap_counter_ = 0
        print(f'Clock started at {self._start_time_.strftime(self._strformat_)}')

    def toc(self,label=[]):
        """Stop the timer and displays results, appends label to final _list_lap_times entry"""
        from datetime import datetime
        from pytz import timezone
        from tzlocal import get_localzone
        from bs_ds import list2df


        _final_end_time_ = self.get_time()
        _total_time_ = _final_end_time_ - self._start_time_
        _end_label_ = label

        self._lap_counter_+=1
        self._final_end_time_ = _final_end_time_
        self._lap_label_=_end_label_
        self._lap_end_time_ = _final_end_time_.strftime(self._strformat_)
        self._lap_duration_ = _final_end_time_ - self._prior_start_time_
        self._total_time_ = _total_time_
        self.mark_lap_list()

        # Append Summary Line
        print(f'\tLap #{self._lap_counter_} done @ {self._lap_end_time_}\tlabel: {self._lap_label_:>{20}}\tduration: {self._lap_duration_.total_seconds()} sec)')
        self._lap_times_list_.append(['Start-End',self._start_time_.strftime(self._strformat_), self._start_label_,self._final_end_time_.strftime(self._strformat_),'Total Time:', self._total_time_.total_seconds() ])

        df_lap_times = list2df(self._lap_times_list_,index_col='Lap #')
        print(f'Total Time: {_total_time_}.')
        if self._verbose_>1:
            return df_lap_times



    def lap(self, label=None):
        """Records time, duration, and label for current lap. Output display varies with clock verbose level.
        Calls .mark_lap_list() to document results in clock._list_lap_ times."""
        from datetime import datetime

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

        if self._verbose_>0:
            print(f'\tLap #{self._lap_counter_} done @ {self._lap_end_time_}\tlabel: {self._lap_label_:>{20}}\tduration: {self._lap_duration_.total_seconds()} sec)')
