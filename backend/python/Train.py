import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
import joblib

input_file = "data/cleaned_real_estate_data.csv"
zip_file = "data/uszips.csv"
city_file = "data/uscities.csv"
cities_file = "data/city_list.txt"
df = pd.read_csv(input_file)
zip_to_coordinate = pd.read_csv(zip_file)
city_center = pd.read_csv(city_file)

df_cities = pd.read_csv(cities_file, header=None, names=['city'])
city_center = df_cities.merge(city_center[['city', 'lat', 'lng']], left_on=['city'], right_on=['city'], how='left')
city_center = city_center.rename(columns={'lat': 'center_lat', 'lng': 'center_lng'})

zip_to_coordinate['zip_code'] = zip_to_coordinate['zip']
df['zip_code'] = df['zip_code'].astype(str).str.zfill(5)
zip_to_coordinate['zip_code'] = zip_to_coordinate['zip_code'].astype(str).str.zfill(5)

df = df.merge(zip_to_coordinate[['zip_code', 'lat', 'lng']], on='zip_code', how='left')

df = df.merge(city_center[['city', 'center_lat', 'center_lng']], on='city', how='left')


def haversine(row):
    if pd.isna(row['center_lat']) or pd.isna(row['center_lng']):
        return 0

    lat1, lng1 = np.radians(row['center_lat']), np.radians(row['center_lng'])
    lat2, lng2 = np.radians(row['lat']), np.radians(row['lng'])

    dlat = lat2 - lat1
    dlng = lng2 - lng1

    a = np.sin(dlat / 2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlng / 2)**2
    c = 2 * np.arcsin(np.sqrt(a))

    return 6371 * c


df['prev_sold_date'] = pd.to_datetime(df['prev_sold_date'], errors='coerce')
df['days_from_sold'] = (pd.Timestamp.today() - df['prev_sold_date']).dt.days
df['log_price'] = np.log1p(df['price'])
zip_num_dict = df['zip_code'].value_counts().to_dict()
df['zip_freq'] = df['zip_code'].map(zip_num_dict)
city_num_dict = df['city'].value_counts().to_dict()
df['city_freq'] = df['city'].map(city_num_dict)
state_num_dict = df['city'].value_counts().to_dict()
df['city_freq'] = df['city'].map(city_num_dict)
city_median_price = df.groupby('city')['price'].median().reset_index()
city_median_price.columns = ['city', 'city_median_price']
df = df.merge(city_median_price, on='city', how='left')
city_median_sqft = df.groupby('city')['house_size'].median().reset_index()
city_median_sqft.columns = ['city', 'city_median_sqft']
df = df.merge(city_median_sqft, on='city', how='left')
state_median_price = df.groupby('state')['price'].median().reset_index()
state_median_price.columns = ['state', 'state_median_price']
df = df.merge(state_median_price, on='state', how='left')
df['total_rooms'] = df['bed'] + df['bath']
df['bed_to_bath_ratio'] = df['bed'] / df['bath']
df['house_to_lot_ratio'] = df['house_size'] / (df['acre_lot'] + 10)
df['city_price_per_sqft'] = df['city_median_price'] / df['city_median_sqft']
df['size_efficiency'] = df['house_size'] / (df['bed'] + df['bath'])
df['distance_to_center'] = df.apply(haversine, axis=1)

rfr = RandomForestRegressor(n_estimators=200, max_depth=25, max_features='sqrt', min_samples_split=5, min_samples_leaf=2
                            , n_jobs=-1, random_state=42, verbose=1)

check_cols = ['bed', 'bath', 'acre_lot', 'total_rooms', 'bed_to_bath_ratio', 'house_size', 'size_efficiency',
              'house_to_lot_ratio', 'city_median_price', 'city_price_per_sqft', 'days_from_sold', 'zip_freq']
df = df.dropna(subset=check_cols)
X_all = df[['bed', 'bath', 'acre_lot', 'total_rooms', 'bed_to_bath_ratio', 'house_size', 'size_efficiency',
            'house_to_lot_ratio', 'state_median_price', 'city_median_price', 'city_price_per_sqft', 'days_from_sold',
            'zip_freq', 'city_freq', 'distance_to_center', 'lat', 'lng']]
y_all = df['log_price']
rfr.fit(X_all, y_all)

joblib.dump(rfr, "rfr_model.pkl")
