from loader import Loader
from sklearn.ensemble import RandomForestRegressor

import numpy as np

class Trainer:
    def get_rmsle(self, y_pred, y_actual):
        diff = np.log(y_pred + 1) - np.log(y_actual + 1)
        mean_error = np.square(diff).mean()
        return np.sqrt(mean_error)
    
    def random_forest_train(self, X, y):
        params = {'n_estimators': 1000, 'max_depth': 15, 'random_state': 0, 'min_samples_split' : 5, 'n_jobs': -1}
        random_forest_model = RandomForestRegressor(**params)
        model = random_forest_model.fit(X, y)
        return model
    
    def random_forest_predict(self, model, X):
        y_predict = model.predict(X)
        return y_predict
        
    def train(self):
        my_loader = Loader()
        train, valid, test = my_loader.load_data('data/train.csv', 'data/test.csv')
        input_cols = ['season', 'holiday', 'workingday', 'weather', 'temp', 'atemp', 'windspeed', 'hour', 'month', 'year']
        X_train, y_train, y_train_registered, y_train_casual = my_loader.create_data(train, input_cols)
        X_valid, y_valid, y_valid_registered, y_valid_casual = my_loader.create_data(valid, input_cols)
        
        model_registered = self.random_forest_train(X_train, y_train_registered)
        y_predict_registered = self.random_forest_predict(model_registered, X_valid)
        
        model_casual = self.random_forest_train(X_train, y_train_casual)
        y_predict_casual = self.random_forest_predict(model_casual, X_valid)
        
        y_predict_count = np.round(y_predict_registered + y_predict_casual)
        rmsle = self.get_rmsle(y_predict_count, y_valid)
        print(rmsle)
        
    
if __name__ == "__main__":
    my_trainer = Trainer()
    my_trainer.train()