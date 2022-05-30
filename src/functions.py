import pandas as pd
import matplotlib.pyplot as plt
import re
from sklearn.linear_model import LinearRegression
import numpy as np



def badrate_summary_categories(df, col, bad_col):
    summary = df.groupby(col)[bad_col].agg(['sum', 'count'])
    summary['rate'] = 100*summary['sum']/summary['count']
    summary.rename(columns = {'sum': 'n_positive', 'count': 'n_records'}, inplace=True)
    return summary


def plot_distro_per_cat(df, dist_col, cat_col, plt_args = {}, figsize= (15,10)):
    plt.figure(figsize=figsize)
    ax = plt.gca()
    df.loc[:, [cat_col, dist_col]].pivot(columns=cat_col)[dist_col].plot(
            kind="hist", ax=ax, **plt_args, title = 'distribution of {} per category of {}'.format(dist_col, cat_col))
            
def process_cabin(data):
    """generate some info based on the cabin data"""
    data['J_cabin_letter'] = data['Cabin'].fillna('MISSING').apply(lambda x: re.sub(r'\d+', '',x ))
    data.loc[data['J_cabin_letter'].isin(['B B', 'B B B B', 'B B B']), 'J_cabin_letter']= 'B'
    data.loc[data['J_cabin_letter'].isin(['C C C', 'C C']), 'J_cabin_letter']= 'C'
    data.loc[data['J_cabin_letter'].isin(['F G', 'F', 'G', 'F E']), 'J_cabin_letter']= 'F/G'
    data.loc[data['J_cabin_letter'] == 'T', 'J_cabin_letter']= 'MISSING'
    data.loc[data['J_cabin_letter'] == 'D D', 'J_cabin_letter']= 'D'
    data['J_room_number'] = data['Cabin'].fillna('MISSING').apply(lambda x: re.sub(r'[A-Z]', '', x))
    return data
    
def process_Ticket(data):
    """generate some info based on ticket data"""
    ticket_info = pd.DataFrame()
    ticket_info['raw'] = data['Ticket']
    ticket_info['prefix'] = ['_'.join(i[0:2]) if len(i)>2 else i[0] if len(i)==2 else 'NO_PREFIX' for i in data['Ticket'].str.split(' ')]
    ticket_info['ticket_number'] = [i[2] if len(i)>2 else i[1] if len(i)==2 else i[0] for i in data['Ticket'].str.split(' ')]
    ticket_info['prefix'] = ticket_info['prefix'].str.replace('\.','', regex=True).str.upper()
    
    
    ticket_info['prefix'].value_counts()
    ticket_info['J_prefix'] = ticket_info['prefix'].copy()
    ticket_info['J_ticket_location'] = ''
    ticket_info.loc[ticket_info['J_prefix'].str.contains('|'.join(['SOTON','STON'])), 'J_ticket_location'] = 'southampton'
    ticket_info.loc[ticket_info['J_prefix'].str.contains('|'.join(['PARIS'])), 'J_ticket_location'] = 'paris'
    # looks like we have people from southampton? STON SOTON 
    tidy_titles={
        'CA':['C.A.', 'CA.', 'CA', 'CA/SOTON'],
    'OQ': ['O.Q', 'OQ', 'SOTON/OQ'],
    'PP': ['PP', 'P.P.', 'SO/PP', 'P/PP', 'SW/PP'],
    'WC': ['W./C.', 'W/C'],
    'A4': ['A/4', 'A4', 'SC/A4'],
    'A5': ['A5','A.5.', 'A/5'],
    'WEP': ['WEP','WE/P'],
        'O2': ['O2', 'O_2', 'SOTON/O2', 'STON/O2', 'STON/O_2'],
        'SC': ['SC', 'SC/PARIS', 'SC/AH', 'SC/AH_BASLE'],
        'SOC': ['SO/C', 'SOC']
    }

    for k,v in tidy_titles.items():
        ticket_info.loc[ticket_info['J_prefix'].isin(v), 'J_prefix'] = k
    
    return ticket_info

def process_name(data):
    """take the name col, and generate a grouped var for title, and #people with same name as them"""
    data_names = data['Name'].copy()
    name_df = pd.DataFrame()
    
    
    # can i split out the surname using ','
    split_comma = data_names.str.split(',')
    #split_comma.apply(lambda x: len(x)).value_counts()
    name_df['surname'] = split_comma.apply(lambda x: x[0]).str.upper()
    name_df['title_forenames'] = split_comma.apply(lambda x: x[1].strip())
    name_df['title'] = name_df['title_forenames'].str.split('.').apply(lambda x: x[0])
# originally did this splitting on ' ' but it gives basically the same data, except for missing the countess.
    name_df['J_title_grped'] = name_df['title']
    name_df.loc[name_df['J_title_grped'].isin(['Major', 'Col', 'Capt']), 'J_title_grped'] = 'Military'
    name_df.loc[name_df['J_title_grped']== 'Mlle', 'J_title_grped'] = 'Miss'
    name_df.loc[name_df['J_title_grped']== 'Don', 'J_title_grped'] = 'Mr'
    name_df.loc[name_df['J_title_grped']== 'Ms', 'J_title_grped'] = 'Miss'
    name_df.loc[name_df['J_title_grped']== 'Mme', 'J_title_grped'] = 'Mrs'
    name_df.loc[name_df['J_title_grped'].isin(['Lady', 'Jonkheer', 'Sir', 'the Countess']), 'J_title_grped'] = 'Nobility'

    # grouping the small categories together, they are statistically insignificant
    name_df.loc[name_df['J_title_grped'].isin(['Military', 'Nobility', 'Dr', 'Rev']), 'J_title_grped'] = 'Nobility/Job Prefix'
    n_relatives = name_df['surname'].value_counts().to_frame('J_nrelatives') - 1
    # lets assume everyone with the same surname are all related
    name_df = name_df.merge(n_relatives, left_on='surname', right_index=True, how='left')
    
    return name_df.loc[:,['J_title_grped', 'J_nrelatives']]
    
    
def process_siblings_spouses(data):
    data['J_n_siblings_spouses'] = data['SibSp']+data['Parch']
    return data
    

def calculate_woe_for_column(newX, var,target = 'died', eps= 1e-10, min_rate = 0.05):
    """Calculate the weight of evidence per bin for a variable, and also check what is the gradient of the woe"""
    WoEs = []
    IV = -1
    trend = -1
    intercept = -1
    maxdiff = -1
    woe_trend = -1



    counts = newX.groupby(by=var, observed=True)[
        target].value_counts().sort_index()


    counts = counts.unstack(level=-1).fillna(0)
    counts.columns = counts.columns.to_series().map({0: "Good",
                                                     1: "Bad"})
    total = (counts["Good"]+counts["Bad"]).sum()
    total_good = counts["Good"].sum()
    total_bad = counts["Bad"].sum()
    # rate of bads per bin
    counts["Bad rate"] = counts["Bad"]/(counts["Good"]+counts["Bad"])
    # rate of entries per bin
    counts["Good+Bad"] = (counts["Good"]+counts["Bad"])/total
    # rate of goods over total of goods per bin
    counts["Good"] = counts["Good"]/total_good
    # rate of bads over total of bads per bin
    counts["Bad"] = counts["Bad"]/total_bad
    counts["Good%-Bad%"] = 100*(counts["Good"]-counts["Bad"])

    counts = counts[(counts["Good+Bad"] > min_rate) & (
        counts["Bad"] > 0) & (counts["Good"] > 0)].copy()

    counts["WoE"] = np.log((counts["Good"]+eps)/(counts["Bad"]+eps))
    counts["IV_i"] = 100*(counts["Good"]-counts["Bad"])*counts["WoE"]
    IV = counts["IV_i"].sum()

    # what way does our variable 'pull'?
    Yvals = counts.sort_index().loc[:, "WoE"].values
    Xvals = np.arange(len(Yvals))

    linear_reg = LinearRegression()
    linear_reg.fit(Xvals.reshape(-1, 1), Yvals.reshape(-1, 1))
    coef = linear_reg.coef_
    WoEs = Yvals.tolist()
    values_fit = Xvals*coef[0][0] + linear_reg.intercept_[0]
    trend = coef[0][0]
    intercept = linear_reg.intercept_[0]
    maxval = max(values_fit)
    minval = min(values_fit)
    maxdiff = maxval - minval

    woe_trend = get_woe_trend(counts.copy())

    return counts, woe_trend

def get_woe_trend(df):
    if df.shape[0] < 2:
        return 0
    elif (df['WoE'] == df['WoE'].cummax()).all():
        return +1
    elif (df['WoE'] == df['WoE'].cummin()).all():
        return -1
    else:
        return 0