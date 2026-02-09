import os
import csv
import math
from datetime import datetime


def map_func(data_dir, worker_id):
    """
    For each accident, create a key from (road_feature, weather, day/night)
    and calculate impact score.
    Returns: list of (key, impact_score) pairs
    """
    results = []
    
    for filename in os.listdir(data_dir):
        if not filename.endswith('.csv'):
            continue
            
        filepath = os.path.join(data_dir, filename)
        with open(filepath, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            
            for row in reader:
                road_feature = get_road_feature(row)
                weather = simplify_weather(row.get('Weather_Condition', 'Unknown'))
                time_of_day = row.get('Sunrise_Sunset', 'Day')
                
                key = f"{road_feature}_{weather}_{time_of_day}"
                
                try:
                    severity = float(row.get('Severity', 1))
                except:
                    severity = 1
                try:
                    distance = float(row.get('Distance(mi)', 0))
                except:
                    distance = 0
                duration = calculate_duration(row)
                
                impact = severity * (1 + distance) * (1 + duration)
                
                results.append((key, impact))
    
    return results


def get_road_feature(row):
    """Find which road feature was present at the accident."""
    features = ['Station','Junction', 'Crossing', 'Traffic_Signal', 'Stop', 'Railway', 'Roundabout']
    
    for feature in features:
        if row.get(feature, 'False') == 'True':
            return feature
    
    return 'None'


def simplify_weather(weather):
    """
    Group weather into 4 categories:
    - Fair
    - Cloudy (Mostly Cloudy + Cloudy)
    - PartlyClear (Clear + Partly Cloudy + Overcast)
    - BadWeather (Rain, Snow, Fog, etc.)
    """
    weather = weather.lower() if weather else ''
    
    if weather == 'fair' or weather == 'fair / windy':
        return 'Fair'
    elif 'mostly cloudy' in weather or weather == 'cloudy' or weather == 'cloudy / windy':
        return 'Cloudy'
    elif 'clear' in weather or 'partly cloudy' in weather or 'overcast' in weather:
        return 'PartlyClear'
    else:
        return 'BadWeather'


def calculate_duration(row):
    """Calculate accident duration in hours."""
    try:
        start = row.get('Start_Time', '')
        end = row.get('End_Time', '')
        start_dt = datetime.strptime(start[:19], '%Y-%m-%d %H:%M:%S')
        end_dt = datetime.strptime(end[:19], '%Y-%m-%d %H:%M:%S')
        return (end_dt - start_dt).total_seconds() / 3600
    except:
        return 0


def shuffle_func(key):
    """
    Decides which worker gets this key.
    Returns a list of integers, for each integer engine does: target % num_workers
    """
    weather = key.split('_')[1] if '_' in key else ''
    
    if weather == 'Fair':
        return [0]
    elif weather == 'Cloudy':
        return [1]
    elif weather == 'PartlyClear':
        return [2]
    else:
        return [3]


def reduce_func(data, worker_id):
    """Processes data (in format: List[dict_item[key, values]]."""
    results = []
    for key, values in data:
        count = len(values)
        total_impact = sum(values)
        avg_impact = total_impact / count if count > 0 else 0
        min_impact = min(values) if values else 0
        max_impact = max(values) if values else 0
        log_factor = math.log(count + 1)
        danger_score = avg_impact * log_factor

        result = {
            'count': count,
            'log_count': round(log_factor, 2),
            'avg_impact': round(avg_impact, 2),
            'danger_score': round(danger_score, 2),
            'min_impact': round(min_impact, 2),
            'max_impact': round(max_impact, 2),
            'formula': f"{round(avg_impact, 2)} * {round(log_factor, 2)} = {round(danger_score, 2)}"
        }

        results.append((key, result))
    return results
