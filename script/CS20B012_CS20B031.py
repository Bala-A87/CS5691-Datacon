# TODO: Try One-Hot Encoding, perform precomputation after train_test_split

# Import required libraries

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor

# Define some constants

BD = "bookings_data"
B = "bookings"
CD = "customer_data"
HD = "hotels_data"
PD = "payments_data"
SS = "sample_submission_5"
TD = "train_data"

base_path = r"../data/"

# Define helper functions

def load_from_csv(filename):
    file_path = base_path + filename + ".csv"
    return pd.read_csv(file_path)

def get_hash(val_list):
    hashmap = {}
    for i,val in enumerate(val_list):
        hashmap[val] = i
    return hashmap

def trim_pred(preds):
    for i in range(preds.shape[0]):
        if preds[i] > 5:
            preds[i] = 5
        elif preds[i] < 1:
            preds[i] = 1
    return preds

# Load the data

bookings_data = load_from_csv(BD)
bookings = load_from_csv(B)
customer_data = load_from_csv(CD)
hotels_data = load_from_csv(HD)
payments_data = load_from_csv(PD)
sample_submission = load_from_csv(SS)
train_data = load_from_csv(TD)

# Modify data to handle non-numeric values

bookings['booking_create_timestamp'] = pd.to_datetime(bookings['booking_create_timestamp'])
bookings['booking_approved_at'] = pd.to_datetime(bookings['booking_approved_at'])
bookings['booking_checkin_customer_date'] = pd.to_datetime(bookings['booking_checkin_customer_date'])
bookings_data['booking_expiry_date'] = pd.to_datetime(bookings_data['booking_expiry_date'])

unique_ids = payments_data['booking_id'].unique()
hash_val = get_hash(unique_ids)
counts = np.zeros(unique_ids.shape)
total_installments = np.zeros(unique_ids.shape)
value_sum = np.zeros(unique_ids.shape)
for i in range(payments_data.shape[0]):
    booking_id = payments_data.at[i, 'booking_id']
    installments = payments_data.at[i, 'payment_installments']
    payment_value = payments_data.at[i, 'payment_value']
    counts[hash_val[booking_id]] += 1
    total_installments[hash_val[booking_id]] += installments
    value_sum[hash_val[booking_id]] += payment_value
payments_data_modified = []
for booking_id in unique_ids:
    payments_made =  counts[hash_val[booking_id]]
    payments_installments = total_installments[hash_val[booking_id]]
    payments_value = value_sum[hash_val[booking_id]]
    payments_data_modified.append([booking_id, payments_made, payments_installments, payments_value])
payments_data_modified = pd.DataFrame(data=payments_data_modified, columns=['booking_id', 'payments_made', 'payments_installments', 'payments_value'])

unique_ids = hotels_data['hotel_id'].unique()
hotel_id_hash = get_hash(unique_ids)
hotels_data_modified = []
for i in range(hotels_data.shape[0]):
    hotel_id = hotel_id_hash[hotels_data.at[i, 'hotel_id']]
    category = hotels_data.at[i, 'hotel_category']
    name_length = hotels_data.at[i, 'hotel_name_length']
    description_length = hotels_data.at[i, 'hotel_description_length']
    photos_qty = hotels_data.at[i, 'hotel_photos_qty']
    hotels_data_modified.append([hotel_id, category, name_length, description_length, photos_qty])
hotels_data_modified = pd.DataFrame(data=hotels_data_modified, columns=[
    'hotel_id', 'hotel_category', 'hotel_name_length', 'hotel_description_length', 'hotel_photos_qty'])
hotels_data_modified.fillna(0, inplace=True)

unique_unique_ids = customer_data['customer_unique_id'].unique()
unique_id_hash = get_hash(unique_unique_ids)
unique_countries = customer_data['country'].unique()
country_hash = get_hash(unique_countries)
customer_data_modified = []
for i in range(customer_data.shape[0]):
    customer_id = customer_data.at[i, 'customer_id']
    unique_id_num = unique_id_hash[customer_data.at[i, 'customer_unique_id']]
    country_num = country_hash[customer_data.at[i, 'country']]
    customer_data_modified.append([customer_id, unique_id_num, country_num])
customer_data_modified = pd.DataFrame(data=customer_data_modified, columns=['customer_id', 'unique_id_num', 'country_num'])

unique_booking_status = bookings['booking_status'].unique()
status_hash = get_hash(unique_booking_status)
bookings_modified = []
for i in range(bookings.shape[0]):
    booking_id = bookings.at[i, 'booking_id']
    customer_id = bookings.at[i, 'customer_id']
    booking_status_num = status_hash[bookings.at[i, 'booking_status']]
    create_date = bookings.at[i, 'booking_create_timestamp']
    approved_date = bookings.at[i, 'booking_approved_at']
    checkin_date = bookings.at[i, 'booking_checkin_customer_date']
    approval_time = (approved_date-create_date).total_seconds()/60
    checkin_time = (checkin_date-create_date).total_seconds()/1440
    bookings_modified.append([booking_id, customer_id, booking_status_num, create_date, approval_time, checkin_time])
bookings_modified = pd.DataFrame(data=bookings_modified, columns=[
    'booking_id', 'customer_id', 'booking_status_num', 'booking_create_timestamp', 'booking_approval_time', 'booking_checkin_time'])
bookings_modified.fillna(-1, inplace=True)

unique_ids = bookings_data['booking_id'].unique()
hash_val = get_hash(unique_ids)
unique_agents = bookings_data['seller_agent_id'].unique()
agent_hash = get_hash(unique_agents)
counts = np.zeros(unique_ids.shape)
hotel_ids = np.zeros(unique_ids.shape).tolist()
seller_agent_ids = np.zeros(unique_ids.shape).tolist()
booking_expiry_dates = np.zeros(unique_ids.shape).tolist()
prices = np.zeros(unique_ids.shape).tolist()
agent_feess = np.zeros(unique_ids.shape).tolist()
for i in range(bookings_data.shape[0]):
    booking_id = bookings_data.at[i, 'booking_id']
    counts[hash_val[booking_id]] += 1
    hotel_ids[hash_val[booking_id]] = hotel_id_hash[bookings_data.at[i, 'hotel_id']]
    seller_agent_ids[hash_val[booking_id]] = agent_hash[bookings_data.at[i, 'seller_agent_id']]
    booking_expiry_dates[hash_val[booking_id]] = bookings_data.at[i, 'booking_expiry_date']
    prices[hash_val[booking_id]] = bookings_data.at[i, 'price']
    agent_feess[hash_val[booking_id]] = bookings_data.at[i, 'agent_fees']
bookings_data_modified = []
for booking_id in unique_ids:
    sub_requests = counts[hash_val[booking_id]]
    hotel_id = hotel_ids[hash_val[booking_id]]
    seller_agent_id = seller_agent_ids[hash_val[booking_id]]
    booking_expiry_date = booking_expiry_dates[hash_val[booking_id]]
    price = prices[hash_val[booking_id]]
    agent_fees = agent_feess[hash_val[booking_id]]
    bookings_data_modified.append([booking_id, sub_requests, hotel_id, seller_agent_id, booking_expiry_date, price, agent_fees])
bookings_data_modified =  pd.DataFrame(data=bookings_data_modified, columns=[
    'booking_id', 'sub_requests', 'hotel_id', 'seller_agent_id_num', 'booking_expiry_date', 'price', 'agent_fees'
])

# Merge the separated data to get entire train and test data

train_data_full = pd.merge(left=train_data, right=bookings_modified, how='left', on='booking_id')
train_data_full = pd.merge(left=train_data_full, right=bookings_data_modified, how='left', on='booking_id')
train_data_full = pd.merge(left=train_data_full, right=customer_data_modified, how='left', on='customer_id')
train_data_full = pd.merge(left=train_data_full, right=hotels_data_modified, how='left', on='hotel_id')
train_data_full = pd.merge(left=train_data_full, right=payments_data_modified, how='left', on='booking_id')
booking_expiry = []
unique_ids = train_data_full['booking_id'].unique()
id_hash = get_hash(unique_ids)
expiry_times = np.zeros(unique_ids.shape)
for i in range(train_data_full.shape[0]):
    booking_id = train_data_full.at[i, 'booking_id']
    expiry_date = train_data_full.at[i, 'booking_expiry_date']
    create_date = train_data_full.at[i, 'booking_create_timestamp']
    expiry_time = (expiry_date-create_date).total_seconds()/1440
    expiry_times[id_hash[booking_id]] = expiry_time
for booking_id in unique_ids:
    expiry_time = expiry_times[id_hash[booking_id]]
    booking_expiry.append([booking_id, expiry_time])
booking_expiry = pd.DataFrame(data=booking_expiry, columns=['booking_id', 'booking_expiry_time'])
train_data_full = pd.merge(left=train_data_full, right=booking_expiry, how='left', on='booking_id')
train_data_full.drop(labels=['booking_create_timestamp', 'booking_expiry_date'], axis=1, inplace=True)

test_data = sample_submission['booking_id']
test_data = pd.merge(left=test_data, right=bookings_modified, how='left', on='booking_id')
test_data = pd.merge(left=test_data, right=bookings_data_modified, how='left', on='booking_id')
test_data = pd.merge(left=test_data, right=customer_data_modified, how='left', on='customer_id')
test_data = pd.merge(left=test_data, right=hotels_data_modified, how='left', on='hotel_id')
test_data = pd.merge(left=test_data, right=payments_data_modified, how='left', on='booking_id')
booking_expiry = []
unique_ids = test_data['booking_id'].unique()
id_hash = get_hash(unique_ids)
expiry_times = np.zeros(unique_ids.shape)
for i in range(test_data.shape[0]):
    booking_id = test_data.at[i, 'booking_id']
    expiry_date = test_data.at[i, 'booking_expiry_date']
    create_date = test_data.at[i, 'booking_create_timestamp']
    expiry_time = (expiry_date-create_date).total_seconds()/1440
    expiry_times[id_hash[booking_id]] = expiry_time
for booking_id in unique_ids:
    expiry_time = expiry_times[id_hash[booking_id]]
    booking_expiry.append([booking_id, expiry_time])
booking_expiry = pd.DataFrame(data=booking_expiry, columns=['booking_id', 'booking_expiry_time'])
test_data = pd.merge(left=test_data, right=booking_expiry, how='left', on='booking_id')
test_data.drop(labels=['booking_create_timestamp', 'booking_expiry_date'], axis=1, inplace=True)

train_labels = train_data_full['rating_score']
train_data_full.drop(labels=['rating_score', 'booking_id', 'customer_id'], axis=1, inplace=True)
train_data_full.fillna(-1, inplace=True)

test_ids = test_data['booking_id']
test_data.drop(labels=['booking_id', 'customer_id'], axis=1, inplace=True)
test_data.fillna(-1, inplace=True)

# Fit train data and predict on test data

rfrgr = RandomForestRegressor(n_estimators=1000, max_features=0.5)
rfrgr.fit(train_data_full, train_labels)
test_pred = trim_pred(np.reshape(rfrgr.predict(test_data), test_ids.shape))
test_pred = pd.DataFrame(data=test_pred, columns=['rating_score'])
test_sub = pd.concat([test_ids, test_pred], axis=1)

test_sub.to_csv("../output/CS20B012_CS20B031.csv", index=False)
