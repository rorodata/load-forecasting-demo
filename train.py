import joblib

from dateutil.relativedelta import relativedelta
import requests
import zipfile
import joblib
import datetime
import os

from download_data import get_periods
from download_data import download_load_data
from download_data import download_weather_data
from transform_data import return_load_df
from transform_data import load_df_pipeline
from transform_data import return_station_df
from transform_data import zones
from transform_data import counties2zones
from transform_data import get_state_and_zones_wbans
from transform_data import return_weather_df
from transform_data import combine_load_weather_df
from transform_data import process_df_for_training


##### DOWNLOAD DATA
DOWNLOAD_PATH = '/volumes/data/downloaded'

dates = get_periods()

download_load_data(dates, save_path = DOWNLOAD_PATH)
download_weather_data(dates, save_path = DOWNLOAD_PATH)


##### TRANSFORM DATA / FEATURE ENGINNERING
PROJECT_PATH = '/volumes/data/project_files'
SAVE_PATH = '/volumes/data/downloaded'

if not os.path.exists(PROJECT_PATH):
    os.mkdir(PROJECT_PATH)

load_df = return_load_df(os.path.join(SAVE_PATH, 'load_data'), grid='N.Y.C.', save = PROJECT_PATH)

county_load_df_rolling = load_df_pipeline(load_df, save = PROJECT_PATH)

station_df = return_station_df(os.path.join(SAVE_PATH, 'weather_data'), save = PROJECT_PATH)
                    
nyc_weather_stations = get_state_and_zones_wbans(
    station_df, state='NY', zone='N.Y.C.', save = PROJECT_PATH)

weather_features_hourly = return_weather_df(
    os.path.join(SAVE_PATH, 'weather_data'),
    nyc_weather_stations, 
    save = PROJECT_PATH)

load_and_weather_data = combine_load_weather_df(
    county_load_df_rolling, weather_features_hourly, save = PROJECT_PATH)

load_and_weather_data_final, features, scaler_dict = process_df_for_training(
    load_and_weather_data['2010-01-01':'2016-12-31'], save = PROJECT_PATH, save_list=[
        'features_to_normalize', 'scaler_dict', 'load_and_weather_data_final'])



##### TRAINING

#load_and_weather_data_final = joblib.load('/volumes/data/notebook2/load_and_weather_data_final')

"""
# features to use in training
features = [
    feature for feature in load_and_weather_data_final.columns
    if ('dewpoint_' in feature or 
        'temperature_' in feature or
        'month_' in feature or 
        'weekday_' in feature or
        'day_' in feature or
        'year' == feature or
        'load' == feature or
        'hour' == feature)]
"""

def train_model(load_and_weather_data_final, features, save=None):
    """
    Train a Random Forest Model on the load and weather features.
    Args:
        load_and_weather_data_final: Dataframe: The processed and standardized dataframe of load and
            weather features to be used for the training.
        features: List: List of features to be used for the training.
        save: String: Path to save the model to.

    Returns:
        regr: RandomForestRegressor: Trained model.
    """

    from sklearn.ensemble import RandomForestRegressor

    X_train, y_train = load_and_weather_data_final[features], load_and_weather_data_final['load_tomorrow']

    regr = RandomForestRegressor(
        n_estimators=1000,
        max_depth=40, 
        random_state=42)

    regr.fit(X_train, y_train)

    if save:
        joblib.dump(regr, os.path.join(save, 'regr.model'))

    return regr


train_model(load_and_weather_data_final, features, save = PROJECT_PATH)
