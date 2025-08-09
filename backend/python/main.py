from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import joblib
import numpy as np
import pandas as pd
from geopy.geocoders import Nominatim
from pydantic import BaseModel
import math

app = FastAPI()
rfr_model = joblib.load("rfr_model.pkl")

input_file = 'data/all_summary.csv'
df = pd.read_csv(input_file, index_col=0).iloc[:, 0]
allMedian = df['median_price']
allArea = df['median_area']
allPricePerSqft = allMedian / allArea

class InputData(BaseModel):
    house_size: float
    lot_area: float
    bed: int
    bath: int
    age_in_years: int
    zip_code: str
    street: str
    city: str
    state: str


def addresslocation(street, city, state, zip_code):
    geolocator = Nominatim(user_agent="geoapi")
    zip_code = zip_code.zfill(5)
    address = street + ", " + city.capitalize() + ", " + state + " " + zip_code + ", USA"
    location = geolocator.geocode(address)
    if location:
        return location.latitude, location.longitude
    else:
        return None, None


def city_data(city):
    city_input = "data/city_summary.csv"
    df_city = pd.read_csv(city_input)
    city_dict = {}
    if df_city[df_city['city'] == city]:
        city_row = df_city[df_city['city'] == city].iloc[0]
        city_dict = {
            'city_median_price': city_row['city_median_price'],
            'city_price_per_sqft': city_row['city_price_per_sqft'],
            'city_freq': city_row['city_freq'],
            'center_lat': city_row['center_lat'],
            'center_lng': city_row['center_lng']
        }
    else:
        city_dict['city_median_price'] = allMedian
        city_dict['city_price_per_sqft'] = allPricePerSqft
        city_dict['city_freq'] = 0
    return city_dict


def zip_data(zip_code):
    zip_code = zip_code.zfill(5)
    zip_input = "data/zip_data.csv"
    df_zip = pd.read_csv(zip_input)
    zip_dict = dict(zip(df_zip['zip_code'].astype(str).str.zfill(5), df_zip['zip_freq']))
    return zip_dict.get(zip_code, 0)


def state_data(state):
    state_input = "data/state_data.csv"
    df_state = pd.read_csv(state_input)
    state_dict = dict(zip(df_state['state'], df_state['state_median_price']))
    return state_dict.get(state, allMedian)


def haversine(location_dict):
    if location_dict['center_lat'] is None or location_dict['center_lng'] is None:
        return 0
    lat1, lng1 = math.radians(location_dict['center_lat']), math.radians(location_dict['center_lng'])
    lat2, lng2 = math.radians(location_dict['lat']), math.radians(location_dict['lng'])

    dlat = lat2 - lat1
    dlng = lng2 - lng1

    a = math.sin(dlat / 2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlng / 2)**2
    c = 2 * math.asin(math.sqrt(a))

    return 6371 * c


@app.post("/predict")


def predict(data: InputData):
    lat, lng = addresslocation(data.street, data.city, data.state, data.zip_code)
    if lat is None or lng is None:
        return {"error": "Address not found"}
    city_dict = city_data(data.city)
    if city_dict is None:
        return {"error": "City data not found"}
    zip_freq = zip_data(data.zip_code)
    if zip_freq is None:
        zip_freq = 0
    state_median_price = state_data(data.state)
    total_rooms = data.bed + data.bath
    bed_to_bath_ratio = data.bed / data.bath
    size_efficiency = data.house_size / total_rooms
    house_to_lot_ratio = data.house_size / (data.lot_area + 10)
    days_from_sold = data.age_in_years * 365
    location_dict = {
        "lat": lat,
        "lng": lng,
        "center_lat": city_dict["center_lat"],
        "center_lng": city_dict["center_lng"]
    }
    distance_to_center = haversine(location_dict)
    features = np.array([[data.bed, data.bath, total_rooms, data.lot_area, bed_to_bath_ratio, data.house_size, size_efficiency,
                          house_to_lot_ratio, state_median_price, city_dict['city_median_price'],
                          city_dict['city_price_per_sqft'], days_from_sold, zip_freq, city_dict['city_freq'],
                          distance_to_center, lat, lng]])
    prediction = rfr_model.predict(features)
    return math.expm1(prediction)


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
