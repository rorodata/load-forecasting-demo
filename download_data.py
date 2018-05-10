from dateutil.relativedelta import relativedelta
import requests
import zipfile
import joblib
import datetime
import os


def get_periods(start=(2010,1,1), end=None):
    """
    Returns a list of dates in the specific format required to download files.

    Args:
        start: Tuple: Form (YYYY, MM, DD)
        end: Tuple: Form (YYYY, MM, DD)

    Returns:
        dates: List: Dates in the format 'YYYYMMDD'
    """
    if not end:
        end_date = datetime.date.today()
    else:
        end_date = datetime.date(*end)
    
    dates = []
    date = datetime.date(*start)

    while date < end_date:
        dates.append(date.strftime('%Y%m%d'))
        date += relativedelta(months=1)

    return dates


#start = (2010, 1, 1)
#end = (2018, 1, 1)

#dates = get_periods(start, end)

def download_load_data(dates, save_path='/volumes/data/downloaded'):
    """
    Download load data from the internet.
    
    Args:
        dates: List: Dates to be used in the url substitution to download the data for the
            specific period.
        save_path: String: Path to save the downloaded files to.

    Returns:
        None
    """
    
    if not os.path.exists(save_path):
        os.mkdir(save_path)

    periods = dates

    for period in periods:
        
        try:
            zip_path = os.path.join(save_path, f'{period}pal_csv.zip')

            if os.path.exists(zip_path):
                print('File ' + os.path.basename(zip_path) + ' already exists. Not Downloading.')
                continue

            url = f'http://mis.nyiso.com/public/csv/pal/{period}pal_csv.zip'

            print(f"Retrieving load data...for period {period}")
            result = requests.get(url)

            with open(zip_path, 'wb') as f:  
                f.write(result.content)

            print("Extracting zipped contents...")
            zip_ref = zipfile.ZipFile(zip_path, 'r')
            zip_ref.extractall(os.path.join(save_path, 'load_data'))
            zip_ref.close()
            print("Done!")
        except:
            continue

#download_load_data(dates)

        
def download_weather_data(dates, save_path='/volumes/data/downloaded'):
    """
    Download weather data from the internet.
    
    Args:
        dates: List: Dates to be used in the url substitution to download the data for the
            specific period.
        save_path: String: Path to save the downloaded files to.

    Returns:
        None
    """
    
    if not os.path.exists(save_path):
        os.mkdir(save_path)

    periods = dates

    for period in periods:
        try:
            period = period[:6]
            zip_path = os.path.join(save_path, f'QCLD{period}.zip')

            if os.path.exists(zip_path):
                print('File ' + os.path.basename(zip_path) + ' already exists. Not Downloading.')
                continue

            url = f'https://www.ncdc.noaa.gov/orders/qclcd/QCLCD{period}.zip'

            print(f"Retrieving weather data...for period {period}")
            result = requests.get(url)

            with open(zip_path, 'wb') as f:  
                f.write(result.content)

            print("Extracting zipped contents...")
            zip_ref = zipfile.ZipFile(zip_path, 'r')
            zip_ref.extractall(os.path.join(save_path, 'weather_data'))
            zip_ref.close()
            print("Done!")
        except:
            continue
        
# download_weather_data(dates)

