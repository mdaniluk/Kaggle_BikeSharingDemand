import pandas as pd
import numpy as np
class Loader:
    def __init__(self):
        self.data = {}
    
    def parse_data_time(self, data):
        date_time = pd.DatetimeIndex(data['datetime'])
        data.set_index(date_time, inplace = True)
        data['date'] = date_time.date
        data['day'] = date_time.day
        data['month'] = date_time.month
        data['year'] = date_time.year
        data['hour'] = date_time.hour
        return data
        
    def read_data(self, path):
        data = pd.read_csv(path)
        data = self.parse_data_time(data)
        return data
        
    def train_valid_split(self, data, split_day):
        train = data[data['day'] <= split_day]
        valid = data[data['day'] > split_day]
        return train, valid
        
    def load_data(self, path_train, path_test):
        data_train_valid = self.read_data(path_train)
        data_test = self.read_data(path_test)
        data_train, data_valid = self.train_valid_split(data_train_valid, split_day = 15)
        return data_train, data_valid, data_test
    
    def create_data(self, data, input_cols):
        X = data[input_cols].as_matrix()
        y = data['count'].as_matrix()
        y_registered = data['registered'].as_matrix()
        y_casual = data['casual'].as_matrix()
        return X, y, y_registered, y_casual
        
    
if __name__ == "__main__":
    my_loader = Loader()
    my_loader.load_data('data/train.csv', 'data/test.csv')