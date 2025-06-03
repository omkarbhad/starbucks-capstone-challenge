import pandas as pd
import numpy as np
from ipywidgets import IntProgress, HTML, VBox
from IPython.display import display
import logging

# -------------------------------
# Progress-bar utility
# -------------------------------
def log_progress(sequence, every=None, size=None, name='Items'):
    """Display a progress bar for iterating over a sequence."""
    is_iterator = False
    if size is None:
        try:
            size = len(sequence)
        except TypeError:
            is_iterator = True

    if size is not None:
        if every is None:
            every = 1 if size <= 200 else int(size / 200)
    else:
        assert every is not None, 'Sequence is iterator; set every'

    progress = IntProgress(min=0, max=1 if is_iterator else size, value=0)
    progress.bar_style = 'info' if is_iterator else ''
    label = HTML()
    box = VBox(children=[label, progress])
    display(box)

    index = 0
    try:
        for index, record in enumerate(sequence, 1):
            if index == 1 or index % every == 0:
                if is_iterator:
                    label.value = f'{name}: {index} / ?'
                else:
                    label.value = f'{name}: {index} / {size}'
                    progress.value = index
            yield record
    except Exception:
        progress.bar_style = 'danger'
        raise
    else:
        progress.bar_style = 'success'
        progress.value = index
        label.value = f"{name}: {index or '?'}"


# -------------------------------
# Data-loading & cleaning
# -------------------------------
def load_data():
    """Load JSON data files into pandas DataFrames."""
    portfolio = pd.read_json('portfolio.json', orient='records', lines=True)
    profile = pd.read_json('profile.json', orient='records', lines=True)
    transcript = pd.read_json('transcript.json', orient='records', lines=True)
    return portfolio, profile, transcript

def clean_portfolio(portfolio):
    """Clean and preprocess portfolio data."""
    portfolio = portfolio.rename(columns={'id': 'offer_id'})
    channels = ['web', 'email', 'social', 'mobile']
    for channel in channels:
        portfolio[channel] = portfolio['channels'].apply(lambda x: 1 if channel in x else 0)
    portfolio = portfolio.drop(columns=['channels'])

    offer_types = ['bogo', 'informational', 'discount']
    for offer_type in offer_types:
        portfolio[offer_type] = portfolio['offer_type'].apply(lambda x: 1 if x == offer_type else 0)
    portfolio = portfolio.drop(columns=['offer_type'])

    return portfolio

def clean_profile(profile):
    """Clean and preprocess profile data."""
    profile = profile.rename(columns={'id': 'customer_id'})
    profile['age'] = profile['age'].replace(118, np.nan)
    profile = profile.dropna(subset=['gender', 'income', 'age'])
    profile['became_member_on'] = pd.to_datetime(
        profile['became_member_on'].astype(str), format='%Y%m%d'
    )
    profile['gender'] = profile['gender'].apply(lambda x: 1 if x == 'M' else 0)
    profile['start_year'] = profile['became_member_on'].dt.year
    profile['start_month'] = profile['became_member_on'].dt.month
    return profile

def clean_transcript(transcript, profile):
    """Clean and preprocess transcript data."""
    transcript = transcript.rename(columns={'person': 'customer_id'})
    transcript['offer_id'] = transcript['value'].apply(
        lambda x: x.get('offer id', x.get('offer_id')) if isinstance(x, dict) else None
    )
    transcript['amount'] = transcript['value'].apply(
        lambda x: x.get('amount') if isinstance(x, dict) else None
    )
    transcript = transcript.drop(columns=['value'])
    transcript = transcript[transcript['customer_id'].isin(profile['customer_id'])]
    transcript['time'] = transcript['time'] / 24.0
    transcript = transcript.drop_duplicates().reset_index(drop=True)
    return transcript

def segregate_transcript(transcript):
    """Segregate transcript into transaction and offer DataFrames."""
    transaction_df = transcript[transcript['event'] == 'transaction'][['customer_id', 'time', 'amount']].copy()

    offers_df = transcript[transcript['event'] != 'transaction'].copy()
    offers_df['received'] = offers_df['event'].apply(lambda x: 1 if x == 'offer received' else 0)
    offers_df['viewed'] = offers_df['event'].apply(lambda x: 1 if x == 'offer viewed' else 0)
    offers_df['completed'] = offers_df['event'].apply(lambda x: 1 if x == 'offer completed' else 0)
    offers_df = offers_df.drop(columns=['event', 'amount'])
    return transaction_df, offers_df

def merge_data(profile, portfolio, offers_df, transaction_df):
    """Merge portfolio, profile, offers, and transaction data."""
    data_rows = []
    customer_ids = offers_df['customer_id'].unique()

    for cid in log_progress(customer_ids, name='Customers'):
        cust_profile = profile[profile['customer_id'] == cid].iloc[0]
        cust_offers = offers_df[offers_df['customer_id'] == cid]
        cust_txns = transaction_df[transaction_df['customer_id'] == cid]

        # Find all received events
        received_events = cust_offers[cust_offers['received'] == 1]
        for _, offer in received_events.iterrows():
            offer_id = offer['offer_id']
            offer_portfolio = portfolio[portfolio['offer_id'] == offer_id].iloc[0]
            start_time = offer['time']
            duration = offer_portfolio['duration']
            end_time = start_time + duration

            # Check if viewed AND completed within the window
            completed_within = cust_offers[
                (cust_offers['completed'] == 1) &
                (cust_offers['time'] >= start_time) &
                (cust_offers['time'] <= end_time)
            ]
            viewed_within = cust_offers[
                (cust_offers['viewed'] == 1) &
                (cust_offers['time'] >= start_time) &
                (cust_offers['time'] <= end_time)
            ]
            offer_successful = int((not viewed_within.empty) and (not completed_within.empty))

            # Sum transactions in that window
            txns_within = cust_txns[
                (cust_txns['time'] >= start_time) &
                (cust_txns['time'] <= end_time)
            ]
            total_amount = txns_within['amount'].sum()

            row = {
                'offer_id': offer_id,
                'customer_id': cid,
                'time': start_time,
                'total_amount': total_amount,
                'offer_successful': offer_successful
            }
            row.update(offer_portfolio.to_dict())
            row.update(cust_profile.to_dict())
            data_rows.append(row)

    merged_df = pd.DataFrame(data_rows)
    return merged_df

# -------------------------------
# Main
# -------------------------------
def main():
    """Execute preprocessing pipeline to generate master_offer_analysis.csv."""
    logging.basicConfig(level=logging.INFO)
    np.random.seed(42)

    print("Loading data...")
    portfolio, profile, transcript = load_data()

    print("Cleaning data...")
    portfolio = clean_portfolio(portfolio)
    profile = clean_profile(profile)
    transcript = clean_transcript(transcript, profile)

    print("Segregating transcript data...")
    transaction_df, offers_df = segregate_transcript(transcript)

    print("Merging data...")
    merged_df = merge_data(profile, portfolio, offers_df, transaction_df)
    merged_df.to_csv('master_offer_analysis.csv', index=False)

    print("Preprocessing complete. Data saved as 'master_offer_analysis.csv'.")

if __name__ == "__main__":
    main()