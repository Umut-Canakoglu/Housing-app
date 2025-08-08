import pandas as pd

input_file = 'data/cleaned_real_estate_data.csv'
df = pd.read_csv(input_file)
unique_cities = df['city'].unique()
city_num_dict = df['city'].value_counts().to_dict()
df['city_freq'] = df['city'].map(city_num_dict)
city_median_price = df.groupby('city')['price'].median().reset_index()
city_median_price.columns = ['city', 'city_median_price']
df = df.merge(city_median_price, on='city', how='left')
city_median_sqft = df.groupby('city')['house_size'].median().reset_index()
city_median_sqft.columns = ['city', 'city_median_sqft']
df = df.merge(city_median_sqft, on='city', how='left')
df['city_price_per_sqft'] = df['city_median_price'] / df['city_median_sqft']
city_summary = df[['city', 'city_freq', 'city_median_price', 'city_price_per_sqft']]
city_summary = city_summary.drop_duplicates(subset=['city'])
city_location = "uscities.csv"
city_loc_df = pd.read_csv(city_location)
city_loc_df = city_loc_df[['city', 'lat', 'lng']]
city_summary = city_summary.merge(city_loc_df, on='city', how='left')
city_summary.rename(columns={"lat": "center_lat", "lng": "center_lng"}, inplace=True)
city_summary.to_csv('city_summary.csv')
