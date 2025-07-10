import pandas as pd
from enum import Enum
import datetime
import zoneinfo  # For Python 3.9+
from dateutil.easter import easter
from dateutil.relativedelta import relativedelta, FR, MO, TH, WE  # for calculating nth weekday
import numpy as np

class StatusType(Enum):
    CLOSED_ONLY = 'Closed'
    OPEN_ONLY = 'Open'
    ALL = 'All'


# -----------------------------------------------------------------
#  Handy wrappers for pandas apply / vectorised use
# -----------------------------------------------------------------

business_hours = {
    "Monday": {"open": "09:00", "close": "19:00"},
    "Tuesday": {"open": "09:00", "close": "19:00"},
    "Wednesday": {"open": "09:00", "close": "19:00"},
    "Thursday": {"open": "09:00", "close": "19:00"},
    "Friday": {"open": "09:00", "close": "19:00"},
    "Saturday": {"open": "11:00", "close": "15:00"},
    "Sunday": {"open": "closed", "close": "closed"},
}


def get_eastern_time():
    """Return current time in Eastern Time."""
    eastern_tz = zoneinfo.ZoneInfo("America/New_York")
    return datetime.datetime.now(eastern_tz)


from zoneinfo import ZoneInfo   # ← add at top of file

EASTERN = ZoneInfo("America/New_York")

def _to_eastern(s: pd.Series) -> pd.Series:
    """
    Helper: parse, localise to UTC if naïve, then convert to Eastern.
    Keeps tz-awareness so downstream business-hour tests are correct.
    """
    # If value already contains timezone info pd.to_datetime(..., utc=True)
    # will respect it and convert to UTC first.
    t = pd.to_datetime(s, utc=True, errors="coerce")
    return t.dt.tz_convert(EASTERN)




# ---------------------------------------------------------------------
#                      Company Holiday Logic
# ---------------------------------------------------------------------

def get_company_holidays(year):
    """
    Return a set of datetime.date objects for all
    company-recognized holidays in a given year.
    The logic below accounts for floating holidays.
    """
    holidays = set()

    # Helper to add holiday easily
    def add_hday(dt):
        holidays.add(dt)

    # 1) New Year’s Day (observed)
    #    If Jan 1 falls on a Sunday, often observed Monday Jan 2, etc.
    #    For simplicity, assume the "observed" date is exactly Jan 1 if it’s a weekday,
    #    or the following Monday if it falls on a weekend:
    new_year = datetime.date(year, 1, 1)
    if new_year.weekday() == 5:  # Saturday
        observed_new_year = new_year + datetime.timedelta(days=2)
    elif new_year.weekday() == 6:  # Sunday
        observed_new_year = new_year + datetime.timedelta(days=1)
    else:
        observed_new_year = new_year
    add_hday(observed_new_year)

    # 2) Martin Luther King, Jr. Day: 3rd Monday of January
    #    dateutil.relativedelta can help:
    mlk = datetime.date(year, 1, 1) + relativedelta(weekday=MO(+3))
    add_hday(mlk)

    # 3) President’s Day: 3rd Monday of February
    presidents = datetime.date(year, 2, 1) + relativedelta(weekday=MO(+3))
    add_hday(presidents)

    # 4) Good Friday: 2 days before Easter
    #    Use dateutil.easter to get Easter Sunday, then subtract 2 days
    gf = easter(year) - datetime.timedelta(days=2)
    add_hday(gf)

    # 5) Memorial Day: last Monday of May
    #    Start from June 1, go back one Monday
    memorial = datetime.date(year, 6, 1) + relativedelta(days=-1, weekday=MO(-1))
    add_hday(memorial)

    # 6) Juneteenth: June 19 (if it falls on Sat/Sun, observe on Fri/Mon).
    juneteenth = datetime.date(year, 6, 19)
    # Example approach to observe on next Monday if Sunday,
    # or previous Friday if Saturday.
    if juneteenth.weekday() == 5:  # Saturday
        juneteenth_observed = juneteenth - datetime.timedelta(days=1)
    elif juneteenth.weekday() == 6:  # Sunday
        juneteenth_observed = juneteenth + datetime.timedelta(days=1)
    else:
        juneteenth_observed = juneteenth
    add_hday(juneteenth_observed)

    # 7) Independence Day: July 4 (similarly shift if on weekend)
    july4 = datetime.date(year, 7, 4)
    if july4.weekday() == 5:   # Saturday
        july4_obs = july4 - datetime.timedelta(days=1)
    elif july4.weekday() == 6: # Sunday
        july4_obs = july4 + datetime.timedelta(days=1)
    else:
        july4_obs = july4
    add_hday(july4_obs)

    # 8) Labor Day: 1st Monday of September
    labor = datetime.date(year, 9, 1) + relativedelta(weekday=MO(+1))
    add_hday(labor)

    # 9) Thanksgiving: 4th Thursday in November
    #    Then day after Thanksgiving is simply the next day (Friday)
    thanksgiving = datetime.date(year, 11, 1) + relativedelta(weekday=TH(+4))
    add_hday(thanksgiving)
    add_hday(thanksgiving + datetime.timedelta(days=1))  # Day after

    # 10) Christmas Day: Dec 25 (shift if on weekend)
    xmas = datetime.date(year, 12, 25)
    if xmas.weekday() == 5:   # Saturday
        xmas_obs = xmas - datetime.timedelta(days=1)
    elif xmas.weekday() == 6: # Sunday
        xmas_obs = xmas + datetime.timedelta(days=1)
    else:
        xmas_obs = xmas
    add_hday(xmas_obs)

    return holidays


def is_company_holiday(date):
    """
    Return True if 'date' (a datetime.datetime or datetime.date)
    is a recognized holiday in that year.
    """
    # Convert to date if needed
    dt = date.date() if isinstance(date, datetime.datetime) else date
    year = dt.year

    # Grab set of holidays for this year (you could memoize this to be more efficient)
    holidays_this_year = get_company_holidays(year)

    return dt in holidays_this_year


# ---------------------------------------------------------------------
#                 Existing Business Hours Logic
# ---------------------------------------------------------------------


def is_business_open(date):
    """Check if the business is open on the given date/time."""
    if pd.isnull(date) or not isinstance(date, datetime.datetime):
        return False  # or np.nan, depending on your needs

    if is_company_holiday(date):
        return False

    day_name = date.strftime("%A")
    hours = business_hours.get(day_name, {"open": "closed"})

    if hours["open"] == "closed":
        return False

    open_hour, open_minute = map(int, hours["open"].split(":"))
    close_hour, close_minute = map(int, hours["close"].split(":"))

    open_time = datetime.datetime(
        date.year, date.month, date.day,
        open_hour, open_minute, tzinfo=date.tzinfo
    )
    close_time = datetime.datetime(
        date.year, date.month, date.day,
        close_hour, close_minute, tzinfo=date.tzinfo
    )

    return open_time <= date < close_time



def is_business_day(dt: datetime.datetime) -> bool:
    """True iff the calendar date is a weekday we are open and it is NOT a holiday."""
    if pd.isnull(dt):
        return False
    return (
        business_hours.get(dt.strftime("%A"), {"open": "closed"})["open"] != "closed"
        and not is_company_holiday(dt)
    )


def is_business_hour(dt: datetime.datetime) -> bool:
    """
    True iff the **exact** timestamp is within opening hours on a
    business day.
    """
    return is_business_open(dt)          # re-uses your existing logic


class InsuredMindWorker(object):

    creation_date_column = 'Creation Date'
    quoted_date_column = ['Quoted Date']
    closed_date_column = ['Closed Date']
    new_lead_column = ['New Lead']
    live_columns = ['Attempted Contact','Gathering Info','Quoting','Quote Sent','Policy Binding','Policy Issued']
    pipeline_columns = new_lead_column + live_columns
    time_period_minutes_list = ['CreationToNewLead_min', 'CreationToClosed_min', 'CreationToQuoted_min',
                                'CreationToAttemptedContact_min', 'CreationToQuoteSent_min',
                                'CreationToPolicyIssued_min',
                                'QuotedToPolicyIssued_min']
    agent_column = 'Agent'
    AGENT_KEY_COLUMN = 'PrimaryAgent'
    contacted_columns = ['Gathering Info','Quoting','Quote Sent','Policy Binding','Policy Issued']
    quoted_columns =['Quote Sent','Policy Binding','Policy Issued']

    day_name_column_order = ['Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday']

    def __init__(self,card_file_name,account_file_name,segment_file_name):
        self._card_file_name = card_file_name
        self._account_file_name = account_file_name
        self._segment_file_name = segment_file_name
        self._init_data()


    def _init_data(self):
        data = pd.read_excel(self._card_file_name)
        data['TL_id'] = data.index
        self.data = data
        self.data = self.transform_leads_to_quotes(self.data)
        self.data = self.clean_day_columns(self.data)
        self.data = self.clean_date_columns(self.data)
        self.data = self.clean_agent_columns(self.data)
        self.data = self.expand_multi_label_columns(self.data,columns_with_prefix={'Label':'label_','Category':'category_'})
        self.data = self.calculate_pipeline_time_periods(self.data)
        self.data = self.add_card_status_column(self.data)


        data_acct = pd.read_csv(self._account_file_name,dtype={'Zip': str})
        data_acct['Zip5'] = data_acct['Zip'].str[:5]
        self.data_acct = data_acct
        account_merge_columns = ['Account Name','Email','City','State','Zip5','Account Type','Account Status']
        self.data = pd.merge(self.data,self.data_acct[account_merge_columns] ,left_on=['Account Name','Email'],right_on=['Account Name','Email'],how='left')

        # Merge Segments
        df_z= pd.read_csv(self._segment_file_name)
        df_z['Zip5'] = df_z['Zip Code'].astype(str).str.zfill(5)
        self.data_census = df_z
        self.data = pd.merge(left=self.data, right=df_z[['Zip5', 'fseg', 'SegmentName', 'category', 'segment', 'order']],
                      how='left', left_on='Zip5', right_on='Zip5')



    def transform_leads_to_quotes(self,df: pd.DataFrame,quote_cols=None,delimiter: str = ",",
                                  lead_id_col: str = "lead_id",
                                  quote_id_col: str = "quote_id") -> pd.DataFrame:
        """
        Convert a *lead‑level* dataframe into a *quote‑level* dataframe.

        Parameters
        ----------
        df : pd.DataFrame
            The input dataframe containing one row per lead.
        quote_cols : list[str] | None
            Columns that hold comma‑separated, quote‑level information.
            If *None*, the default set is::
                ['Category', 'Status', 'Carrier Name',
                 'Closed Date', 'Quoted Date', 'Lost Reason']
        delimiter : str, default “,”
            Character used to separate quote values inside each cell.
        lead_id_col : str, default “lead_id”
            Name of the new column that will hold a unique ID per lead.
        quote_id_col : str, default “quote_id”
            Name of the new column that will hold a unique ID per quote.

        Returns
        -------
        pd.DataFrame
            A new dataframe with one row per quote and the two extra ID columns.

        Notes
        -----
        * All non‑quote columns are carried over unchanged.
        * Missing or shorter lists are padded with ``NaN`` so column lengths match.
        * Date columns containing the word “Date” are parsed to ``datetime64``.
        """
        if quote_cols is None:
            quote_cols = [
                "Category",
                "Status",
                "Carrier Name",
                "Closed Date",
                "Quoted Date",
                "Lost Reason",
            ]

        df = df.copy()

        # Assign a unique ID to every lead
        df[lead_id_col] = range(1, len(df) + 1)

        # Make sure the quote columns are strings and replace NaNs with ""
        for col in quote_cols:
            df[col] = df[col].fillna("").astype(str)

        exploded_rows = []

        for _, row in df.iterrows():
            # Split each quote column into a list
            split_lists = {
                col: [v.strip() for v in row[col].split(delimiter)] if row[col] else [""]
                for col in quote_cols
            }
            max_len = max(len(lst) for lst in split_lists.values())

            # Pad shorter lists so they all have the same length
            for col, lst in split_lists.items():
                if len(lst) < max_len:
                    lst.extend([""] * (max_len - len(lst)))

            # Build one new row per quote
            for i in range(max_len):
                new_row = row.drop(quote_cols).to_dict()  # lead‑level columns
                for col in quote_cols:
                    value = split_lists[col][i]
                    new_row[col] = value if value != "" else np.nan
                new_row[lead_id_col] = row[lead_id_col]
                new_row[quote_id_col] = f"{row[lead_id_col]}-{i + 1}"
                exploded_rows.append(new_row)

        quote_df = pd.DataFrame(exploded_rows)

        # Parse date‑like quote columns
        for col in quote_cols:
            if "Date" in col:
                quote_df[col] = pd.to_datetime(quote_df[col], errors="coerce")

        return quote_df


    def clean_day_columns(self,df):
        # New Lead > Initial Contact > Gathering Info > Quoting > Quote Sent > Policy Binding > Policy Issued

        for column in self.pipeline_columns:
            df[column] = df[column].str.extract('(\d+)')

        return df


    def clean_date_columns(self,df):
        # New Lead > Initial Contact > Gathering Info > Quoting > Quote Sent > Policy Binding > Policy Issued

        df[self.creation_date_column] = _to_eastern(df[self.creation_date_column])
        df['creation_date'] = df[self.creation_date_column].dt.date
        df['creation_day_name'] = df[self.creation_date_column].dt.day_name()
        df['creation_day_of_week'] = df[self.creation_date_column].dt.dayofweek
        df['creation_hour'] = df[self.creation_date_column].dt.hour
        df['creation_year'] = df[self.creation_date_column].dt.year
        df['creation_month'] = df[self.creation_date_column].dt.month
        df['creation_month_end'] = pd.to_datetime(df['creation_year'].astype(str) + '-' +df['creation_month'].astype(str).str.zfill(2)) + pd.offsets.MonthEnd(0)
        df['creation_is_business_open'] = df[self.creation_date_column].apply(is_business_open)
        df['creation_is_business_day']  = df[self.creation_date_column].apply(is_business_day)
        df['creation_is_business_hour']  = df[self.creation_date_column].apply(is_business_hour)


        # for column in self.quoted_date_column+self.closed_date_column:
        #     date_list_column = '{0}_list'.format(column)
        #     df[date_list_column] = df[column].apply(lambda x: x.split(',') if isinstance(x,str) else [])
        #     dates_expanded = pd.DataFrame(df[date_list_column].to_list(),index=df.index)
        #     dates_expanded.columns = [column +'_{}'.format(i+1) for i in range(dates_expanded.shape[1])]
        #     for col in dates_expanded.columns:
        #         # dates_expanded[col] = pd.to_datetime(dates_expanded[col],format='mixed') # date format started to become mixed
        #         dates_expanded[col] = _to_eastern(dates_expanded[col])
        #     df = pd.concat([df, dates_expanded], axis=1)

        df['Closed Date'] = _to_eastern(df['Closed Date'])
        df['Quoted Date'] = _to_eastern(df['Quoted Date'])

        for column in self.pipeline_columns:
            column_moved_on = 'Moved On ' + column
            df[column_moved_on] = _to_eastern(df[column_moved_on])

        df['New Lead Date'] = df['Moved On New Lead'].dt.floor('d')
        df['FirstActionTime'] = df[['Moved On '+ c for c in self.live_columns]].min(axis=1)
        df['FirstActionTime'] = pd.to_datetime(df['FirstActionTime'], errors='coerce', utc=True)
        df['FirstActionTime_is_business_open'] = df['FirstActionTime'].apply(is_business_open)
        df['FirstActionTime_is_business_day'] = df['FirstActionTime'].apply(is_business_day)
        df['FirstActionTime_is_business_hour'] = df['FirstActionTime'].apply(is_business_hour)

        return df

    def clean_agent_columns(self,df):
        column = self.agent_column
        agent_list_column = '{0}_list'.format(column)
        df[agent_list_column]=df[column].apply(lambda x: x.split(',') if isinstance(x,str) else [])
        agents_expanded = pd.DataFrame(df[agent_list_column].to_list(),index=df.index)
        agents_expanded.columns = [column+'_{}'.format(i+1) for i in range(agents_expanded.shape[1])]
        agents_expanded=agents_expanded.rename({'Agent_1':'PrimaryAgent'},axis=1)
        df = pd.concat([df,agents_expanded],axis=1)

        return df

    def expand_multi_label_columns(self,df, columns_with_prefix):
        for column, prefix in columns_with_prefix.items():
            label_lists = df[column].fillna("").apply(
                lambda x: [label.strip() for label in x.split(",") if label.strip()])
            all_labels = set(label for sublist in label_lists for label in sublist)

            for label in all_labels:
                col_name = f"{prefix}{label}"
                df[col_name] = label_lists.apply(lambda labels: label in labels)

        return df


    def add_card_status_column(self,df):
        # card_status_column_name = 'Card Status'
        # df[card_status_column_name] = None
        # idx_won = df['Status'].str.contains("won")
        # idx_inProcess = df['Status'].str.contains("inProcess")
        # idx_lost = df['Status'].str.contains("lost")
        #
        # df.loc[idx_won,card_status_column_name] = "won"
        # df.loc[~idx_won&idx_inProcess, card_status_column_name] = "inProcess"
        # df.loc[(~idx_won)&(~idx_inProcess)&idx_lost,card_status_column_name] = "lost"
        # print("Number of rows with Card Status N/A: {0}".format(df.loc[df[card_status_column_name].isna()].shape[0]))

        card_status_column_name = 'Card Status'

        df[card_status_column_name] = df['Status']
        df['is_won'] = (df['Card Status'].str.lower() == 'won')#.astype(int)
        condition_is_won = df['is_won'] == True

        quoted_columns = ['Moved On ' + c for c in self.quoted_columns]
        condition_is_quoted = df['Quoted Date'].notna()|df[quoted_columns].notna().any(axis=1)|condition_is_won
        df['is_quoted'] = condition_is_quoted

        # Contacted
        contacted_columns = ['Moved On ' + c for c in self.contacted_columns]
        condition_status_not_na = df[contacted_columns].notna().any(axis=1)

        df['is_contacted'] = condition_status_not_na|condition_is_quoted|condition_is_won

        return df


    def add_pipeline_columns(self,df):
        return df

    def calculate_pipeline_time_periods(self,data):

        data['CreationToFirstAction_timedelta'] = data['FirstActionTime']-data['Creation Date']
        data['CreationToFirstAction_min'] = data['CreationToFirstAction_timedelta'] / datetime.timedelta(minutes=1)
        data['CreationToNewLead_timedelta'] = data['Moved On New Lead'] - data['Creation Date']
        data['CreationToNewLead_min'] = data['CreationToNewLead_timedelta']/datetime.timedelta(minutes=1)

        for i in range(len(self.pipeline_columns)-1):
            col_a = 'Moved On '+self.pipeline_columns[i]
            col_b = 'Moved On '+self.pipeline_columns[i+1]

            data[col_b+'_timedelta'] = data[col_b]-data[col_a]
            data[col_b + '_min'] = data[col_b+'_timedelta']/datetime.timedelta(minutes=1)

        data['CreationToClosed_timedelta'] = data['Closed Date']-data['Creation Date']
        data['CreationToClosed_min'] = data['CreationToClosed_timedelta']/datetime.timedelta(minutes=1)
        data['CreationToQuoted_timedelta'] = data['Quoted Date'] - data['Creation Date']
        data['CreationToQuoted_min'] = data['CreationToQuoted_timedelta'] /datetime.timedelta(minutes=1)
        data['CreationToAttemptedContact_timedelta'] = data['Moved On Attempted Contact'] - data['Creation Date']
        data['CreationToAttemptedContact_min'] = data['CreationToAttemptedContact_timedelta'] /datetime.timedelta(minutes=1)
        data['CreationToGatheringInfo_timedelta'] = data['Moved On Gathering Info'] - data['Creation Date']
        data['CreationToGatheringInfo_min'] = data['CreationToGatheringInfo_timedelta'] /datetime.timedelta(minutes=1)
        data['CreationToQuoting_timedelta'] = data['Moved On Quoting'] - data['Creation Date']
        data['CreationToQuoting_min'] = data['CreationToQuoting_timedelta'] /datetime.timedelta(minutes=1)
        data['CreationToQuoteSent_timedelta'] = data['Moved On Quote Sent'] - data['Creation Date']
        data['CreationToQuoteSent_min'] = data['CreationToQuoteSent_timedelta'] /datetime.timedelta(minutes=1)
        data['CreationToPolicyBinding_timedelta'] = data['Moved On Policy Binding'] - data['Creation Date']
        data['CreationToPolicyBinding_min'] = data['CreationToPolicyBinding_timedelta']/datetime.timedelta(minutes=1)
        data['CreationToPolicyIssued_timedelta'] = data['Moved On Policy Issued'] - data['Creation Date']
        data['CreationToPolicyIssued_min'] = data['CreationToPolicyIssued_timedelta']/datetime.timedelta(minutes=1)
        data['QuotedToPolicyIssued_timedelta'] = data['Moved On Policy Issued'] - data['Quoted Date']
        data['QuotedToPolicyIssued_min'] = data['QuotedToPolicyIssued_timedelta']/datetime.timedelta(minutes=1)
        data['QuotedToClosed_timedelta'] = data['Closed Date'] - data['Quoted Date']
        data['QuotedToClosed_min'] = data['QuotedToClosed_timedelta']/datetime.timedelta(minutes=1)


        return data

    def _standard_filter(self,data,status_type=StatusType.ALL,start_date=None, end_date=None,is_business_hours=False,
                         source_list=None):
        if start_date:
            msk = data['Creation Date']>=start_date
            data = data.loc[msk]
        if end_date:
            msk = data['Creation Date']<=end_date
            data = data.loc[msk]

        if status_type==StatusType.CLOSED_ONLY:
            data = data.loc[data['Card Status'].str.contains("lost|won")]
            # investigate status: won, lost
        elif status_type==StatusType.OPEN_ONLY:
            # Does not contain lost or won, as some deals have both "inProcess" and "lost"/"won"
            data = data.loc[data['Card Status'].isin(['inProcess'])]

        if is_business_hours:
            data = data.loc[data['creation_is_business_open']]

        if source_list:
            data = data.loc[data['Source Name'].isin(source_list)]
        return data

    def simple_ai_filter(self):
        data = self.data
        fltr_agent = data[self.AGENT_KEY_COLUMN].isin(['Winchell Regodon','Zach Kamran'])
        fltr_source = data['Source Name'].isin(['FloridaInsuranceQuotes.net'])
        #fltr_status = data['Status'].isin(['inProcess'])&data['Stage'].isin(['New Lead','First Call','Attempted Contact'])
        #fltr_stage = data['Status'].isin(['inProcess']) & data['Stage'].isin(['Attempted Contact'])
        fltr_stage = data['Stage'].isin(['Attempted Contact'])&(data['CreationToAttemptedContact_timedelta']<pd.Timedelta(days=10))
        data = data.loc[fltr_agent&fltr_source&fltr_stage]

        return data

    def get_source_win_rate(self,source_name=None,status_type=StatusType.ALL,start_date=None, end_date=None):
        data =self.data


        data = self._standard_filter(data,status_type=status_type,start_date=start_date,end_date=end_date)

        if source_name is not None:
            data = data.loc[data['Source Name'].isin([source_name])]

        df = data.groupby(['Source Name', 'Card Status']).agg({'TL_id': 'count'}).rename({'TL_id': 'count'},
                                                        axis=1).reset_index().sort_values(by=['Source Name', 'Card Status'])

        df_count = data.groupby(['Source Name']).agg({'TL_id': 'count'}).rename({'TL_id': 'source_count'},axis=1)

        if source_name is not None:
            df['pct'] = df['count'].divide(df['count'].sum()) * 100
        else:
            df = pd.merge(df,
                          df_count,
                          how='left', left_on='Source Name', right_index=True)
            df['pct_source'] = df['count'] / df['source_count'] * 100
        return df

    def get_source_loss_reason(self,source_list=None,start_date=None, end_date=None):
        data = self.data

        data = self._standard_filter(data, start_date=start_date, end_date=end_date)

        if source_list:
            data = data.loc[data['Source Name'].isin(source_list)]


        data = data.loc[data['Card Status'].isin(['lost'])]

        df = data.groupby(['Source Name','Lost Reason']).agg({'TL_id': 'count'}).rename({'TL_id': 'count'},axis=1).reset_index()
        df_count = data.groupby(['Source Name']).agg({'TL_id': 'count'}).rename({'TL_id': 'source_count'}, axis=1)

        df['pct'] = df['count'].divide(df['count'].sum()) * 100
        df = pd.merge(df, df_count, how='left', left_on=['Source Name'], right_index=True)
        df['pct_source'] = df['count'] / df['source_count'] * 100


        return df

    def get_source_loss_stage(self,source_name,status_type=StatusType.ALL):
        # New Lead > Initial Contact > Gathering Info > Quoting > Quote Sent > Policy Binding > Policy Issued
        df = self.data
        return df

    def get_source_active_stage(self,source_name,status_type=StatusType.ALL):
        df = self.data
        return df

    def get_source_category_stage(self,source_name=None,status_type=StatusType.ALL,start_date=None, end_date=None):
        data =self.data


        data = self._standard_filter(data,status_type=status_type,start_date=start_date,end_date=end_date)

        if source_name is not None:
            data = data.loc[data['Source Name'].isin([source_name])]

        df = data.groupby(['Source Name', 'Category']).agg({'TL_id': 'count'}).rename({'TL_id': 'count'},
                                                        axis=1).reset_index().sort_values(by=['Source Name', 'Category'])

        df_count = data.groupby(['Source Name']).agg({'TL_id': 'count'}).rename({'TL_id': 'source_count'},axis=1)

        if source_name is not None:
            df['pct'] = df['count'].divide(df['count'].sum()) * 100
        else:
            df = pd.merge(df,
                          df_count,
                          how='left', left_on='Source Name', right_index=True)
            df['pct_source'] = df['count'] / df['source_count'] * 100
        return df



    def get_win_time_line(self):
        return self.data

    def get_category_win_rate(self):
        return self.data

    def get_segment_win_rate(self,status_type=StatusType.ALL,start_date=None, end_date=None):
        data = self.data

        data = self._standard_filter(data, status_type=status_type, start_date=start_date, end_date=end_date)


        df = data.groupby(['order','SegmentName','Card Status']).agg({'TL_id': 'count'}).rename({'TL_id': 'count'},
                                                        axis=1).reset_index().sort_values(by=['order', 'Card Status'])

        df_count = data.groupby(['SegmentName']).agg({'TL_id': 'count'}).rename({'TL_id': 'segment_count'}, axis=1)

        df['pct'] = df['count'].divide(df['count'].sum()) * 100
        df = pd.merge(df,df_count,how='left',left_on='SegmentName',right_index=True)
        df['pct_seg'] = df['count']/df['segment_count']*100
        return df



    def get_segment_by_source(self,start_date=None, end_date=None):
        data = self.data

        data = self._standard_filter(data, start_date=start_date, end_date=end_date)
        df = data.groupby(['Source Name','order','SegmentName']).agg({'TL_id': 'count'}).rename({'TL_id': 'count'},
                                                        axis=1).reset_index().sort_values(by=['Source Name', 'order'])

        df_count_source = data.groupby(['Source Name']).agg({'TL_id': 'count'}).rename({'TL_id': 'source_count'}, axis=1)
        df_count_segment = data.groupby(['SegmentName']).agg({'TL_id': 'count'}).rename({'TL_id': 'segment_count'},                                                                                       axis=1)

        df['pct'] = df['count'].divide(df['count'].sum()) * 100
        df = pd.merge(df,df_count_source,how='left',left_on='Source Name',right_index=True)
        df = pd.merge(df, df_count_segment, how='left', left_on='SegmentName', right_index=True)
        df['pct_source'] = df['count']/df['source_count']*100
        df['pct_segment'] = df['count'] / df['segment_count'] * 100

        return df

    def get_time_summary_by_stage_by_agent(self,status_type=StatusType.ALL,start_date=None, end_date=None):
        data = self.data
        data = self._standard_filter(data, status_type=status_type, start_date=start_date, end_date=end_date)
        time_period_minutes_list = ['CreationToNewLead_min','CreationToClosed_min','CreationToQuoted_min',
                                    'CreationToAttemptedContact_min','CreationToGatheringInfo_min',
                                    'CreationToQuoting_min','CreationToQuoteSent_min','CreationToPolicyBinding_min',
                                    'CreationToPolicyIssued_min',
                                    'QuotedToPolicyIssued_min']

        assemble_dict = {key:'describe' for key in time_period_minutes_list}
        df = data.groupby([self.AGENT_KEY_COLUMN]).agg(assemble_dict)

        return df


    def get_time_summary_by_stage_by_status(self,status_type=StatusType.ALL,start_date=None, end_date=None):
        data = self.data
        data = self._standard_filter(data, status_type=status_type, start_date=start_date, end_date=end_date)
        time_period_minutes_list = ['CreationToNewLead_min','CreationToClosed_min','CreationToQuoted_min',
                                    'CreationToAttemptedContact_min','CreationToGatheringInfo_min',
                                    'CreationToQuoting_min','CreationToQuoteSent_min','CreationToPolicyBinding_min',
                                    'CreationToPolicyIssued_min',
                                    'QuotedToPolicyIssued_min']
        assemble_dict = {key:'describe' for key in time_period_minutes_list}
        df = data.groupby(['Card Status']).agg(assemble_dict)

        return df

    def get_time_summary_by_day_by_status(self,status_type=StatusType.ALL,start_date=None, end_date=None,is_business_hours=True):
        data = self.data
        data = self._standard_filter(data, status_type=status_type, start_date=start_date, end_date=end_date)
        time_period_minutes_list = ['CreationToNewLead_min','CreationToClosed_min','CreationToQuoted_min',
                                    'CreationToAttemptedContact_min','CreationToGatheringInfo_min',
                                    'CreationToQuoting_min','CreationToQuoteSent_min','CreationToPolicyBinding_min',
                                    'CreationToPolicyIssued_min',
                                    'QuotedToPolicyIssued_min']
        assemble_dict = {key:'describe' for key in time_period_minutes_list}
        df = data.groupby(['creation_day_of_week','creation_day_name','Card Status']).agg(assemble_dict)
        df = df.reset_index().sort_values(by=['creation_day_of_week', 'creation_day_name', 'Card Status'],
                                          ascending=True)

        return df

    def get_time_summary_by_dayofweek(self,status_type=StatusType.ALL,start_date=None, end_date=None,is_business_hours=True):
        data = self.data
        data = self._standard_filter(data, status_type=status_type, start_date=start_date, end_date=end_date)
        time_period_minutes_list = ['CreationToNewLead_min','CreationToClosed_min','CreationToQuoted_min',
                                    'CreationToAttemptedContact_min','CreationToGatheringInfo_min',
                                    'CreationToQuoting_min','CreationToQuoteSent_min','CreationToPolicyBinding_min',
                                    'CreationToPolicyIssued_min',
                                    'QuotedToPolicyIssued_min']
        assemble_dict = {key:'describe' for key in time_period_minutes_list}
        df = data.groupby(['creation_day_of_week','creation_day_name']).agg(assemble_dict)
        df = df.reset_index().sort_values(by=['creation_day_of_week', 'creation_day_name', 'creation_hour'],ascending=True)

        return df


    def get_time_summary_by_day_by_hour(self,status_type=StatusType.ALL,start_date=None, end_date=None,is_business_hours=True,source_list=None):
        data = self.data
        data = self._standard_filter(data, status_type=status_type, start_date=start_date, end_date=end_date,is_business_hours=is_business_hours,source_list=source_list)
        df = data.groupby(['creation_day_of_week','creation_day_name','creation_hour']).agg({'CreationToFirstAction_min':'describe'})
        # df = df.reset_index().sort_values(by=['creation_day_of_week','creation_day_name','creation_hour'],ascending=True)
        df = df.sort_index()
        return df

    def get_time_summary_by_date(self,status_type=StatusType.ALL,start_date=None, end_date=None,is_business_hours=True):
        data = self.data
        data = self._standard_filter(data, status_type=status_type, start_date=start_date, end_date=end_date,is_business_hours=is_business_hours)
        df = data.groupby([self.creation_date_column,'creation_day_of_week']).agg({'CreationToFirstAction_min': 'describe'})
        df = df.reset_index().sort_values(by=[self.creation_date_column],ascending=[True])
        return df


    def get_time_summary_by_hour(self, status_type=StatusType.ALL, start_date=None, end_date=None,is_business_hours=True):
        data = self.data
        data = self._standard_filter(data, status_type=status_type, start_date=start_date, end_date=end_date,is_business_hours=is_business_hours)
        df = data.groupby(['creation_hour']).agg({'CreationToFirstAction_min':'describe'})

        return df

    def get_median_time_by_day_by_hour_pivot(self,status_type=StatusType.ALL, start_date=None, end_date=None,
                                             is_business_hours=True,source_list=None):
        data = self.get_time_summary_by_day_by_hour(status_type=StatusType.ALL, start_date=None, end_date=None,is_business_hours=True,source_list=source_list)
        df = pd.pivot(data=data['CreationToFirstAction_min'].reset_index(), index='creation_hour',columns='creation_day_name', values='50%')
        # Only keep columns from the desired order that are actually present in the DataFrame
        existing_columns = [col for col in self.day_name_column_order if col in df.columns]
        df = df[existing_columns]
        return df

    def get_time_summary_by_day_by_hour_by_status(self, status_type=StatusType.ALL, start_date=None, end_date=None,is_business_hours=True):
        data = self.data
        data = self._standard_filter(data, status_type=status_type, start_date=start_date, end_date=end_date,is_business_hours=is_business_hours)
        data['creation_day_of_week'] = data['Creation Date'].dt.day_name()
        data['creation_hour'] = data['Creation Date'].dt.hour
        df = data.groupby(['creation_day_of_week','creation_hour','Card Status']).agg({'CreationToFirstAction_min':'describe'})
        return df

    def get_time_summary_by_stage(self,status_type=StatusType.ALL,start_date=None, end_date=None):
        data = self.data
        data = self._standard_filter(data, status_type=status_type, start_date=start_date, end_date=end_date)

        df = data[self.time_period_minutes_list].describe()
        return df


    def get_agent_win_rate(self,agent_list=None,status_type=StatusType.ALL,start_date=None, end_date=None,is_business_hours=False):
        data = self.data

        data = self._standard_filter(data, status_type=status_type, start_date=start_date, end_date=end_date,is_business_hours=is_business_hours)
        if agent_list:
            data = data.loc[data[self.AGENT_KEY_COLUMN].isin(agent_list)]

        df = data.groupby([self.AGENT_KEY_COLUMN,'Card Status']).agg({'TL_id': 'count'}).rename({'TL_id': 'count'},
                                                        axis=1).reset_index().sort_values(by=[self.AGENT_KEY_COLUMN, 'Card Status'])

        df_count = data.groupby([self.AGENT_KEY_COLUMN]).agg({'TL_id': 'count'}).rename({'TL_id': 'agent_count'}, axis=1)

        df['pct'] = df['count'].divide(df['count'].sum()) * 100
        df = pd.merge(df,df_count,how='left',left_on=self.AGENT_KEY_COLUMN,right_index=True)
        df['pct_agent'] = df['count']/df['agent_count']*100
        return df

    def get_agent_win_rate_by_source(self,agent_list=None,source_list=None,status_type=StatusType.ALL,start_date=None, end_date=None,is_business_hours=False):
        data = self.data

        data = self._standard_filter(data, status_type=status_type, start_date=start_date, end_date=end_date,is_business_hours=is_business_hours)

        if agent_list:
            if source_list:
                data = data.loc[data[self.AGENT_KEY_COLUMN].isin(agent_list)&data['Source Name'].isin(source_list)]
            else:
                data = data.loc[data[self.AGENT_KEY_COLUMN].isin(agent_list)]
        elif source_list:
            data = data.loc[data['Source Name'].isin(source_list)]

        df = data.groupby([self.AGENT_KEY_COLUMN,'Source Name', 'Card Status']).agg({'TL_id': 'count'}).rename({'TL_id': 'count'},
                                                                                   axis=1).reset_index().sort_values(
            by=[self.AGENT_KEY_COLUMN,'Source Name','Card Status'])

        df_count = data.groupby([self.AGENT_KEY_COLUMN,'Source Name']).agg({'TL_id': 'count'}).rename({'TL_id': 'agent_source_count'}, axis=1)

        df['pct'] = df['count'].divide(df['count'].sum()) * 100
        df = pd.merge(df, df_count, how='left', left_on=[self.AGENT_KEY_COLUMN,'Source Name'], right_index=True)
        df['pct_agent_source'] = df['count'] / df['agent_source_count'] * 100
        return df


    def get_win_rate_by_day_and_time(self,status_type=StatusType.ALL,start_date=None, end_date=None,is_business_hours=False):
        data = self.data

        data = self._standard_filter(data, status_type=status_type, start_date=start_date, end_date=end_date,is_business_hours=is_business_hours)

        df = data.groupby(['creation_day_of_week','creation_hour','Card Status']).agg({'TL_id': 'count'}).rename({'TL_id': 'count'},
                                                                                   axis=1).reset_index().sort_values(
            by=['creation_day_of_week','creation_hour','Card Status'])

        df_count = data.groupby(['creation_day_of_week','creation_hour']).agg({'TL_id': 'count'}).rename({'TL_id': 'day_time_count'}, axis=1)

        df['pct'] = df['count'].divide(df['count'].sum()) * 100
        df = pd.merge(df, df_count, how='left', left_on=['creation_day_of_week','creation_hour'], right_index=True)
        df['pct_day_time'] = df['count'] / df['day_time_count'] * 100

        df = pd.pivot(data=df,
                 index='creation_hour',
                 columns=['creation_day_of_week','Card Status'], values='pct_day_time')
        return df


if __name__ == '__main__':
    file_date = datetime.date(2025,6,25)
    ime = InsuredMindWorker(card_file_name='C:/users/Luiso/data/insured_mine/Dealcards_{}.xlsx'.format(file_date.strftime('%m_%d_%Y')),
                            account_file_name='C:/users/Luiso/data/insured_mine/Accounts_{}.csv'.format(file_date.strftime('%m_%d_%Y')),
                            segment_file_name='C:/users/Luiso/data/census/SegmentsAll_20180809.csv')

    start_date = datetime.datetime(2025,1,1,tzinfo=ZoneInfo("America/New_York"))
    end_date = datetime.datetime(2025,5,31,tzinfo=ZoneInfo("America/New_York"))
    data = ime.data
    pvt = ime.get_median_time_by_day_by_hour_pivot(start_date=start_date,end_date=end_date,source_list=['FloridaInsuranceQuotes.net'])
    ime.data.to_excel('ime_data_full_{}.xlsx'.format(file_date.strftime('%m_%d_%Y')))
    print(data)
