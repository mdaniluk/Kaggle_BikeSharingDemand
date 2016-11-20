from loader import Loader
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor
from sklearn import linear_model, svm
from sklearn.grid_search import GridSearchCV
import numpy as np

class Trainer:
    def get_rmsle(self, y_pred, y_actual):
        diff = np.log(y_pred + 1) - np.log(y_actual + 1)
        mean_error = np.square(diff).mean()
        return np.sqrt(mean_error)
        
    def set_input_cols(self):
        self.input_cols = ['season', 'holiday', 'workingday', 'weather', 'temp', 'atemp', 'humidity', 'windspeed', 'hour', 'year', 'dayofweek']
        
    def grid_search(self, X, y):
        model = RandomForestRegressor(random_state=30)
        param_grid = { 'n_estimators': [800, 1000, 1200], 'max_depth': [10, 15], 'min_samples_split': [2,5] }
        CV_rfc = GridSearchCV(estimator=model, param_grid=param_grid, cv= 5)
        CV_rfc.fit(X, y)
        print (CV_rfc.best_params_)
    
    def gird_search_random_forest(self):
        my_loader = Loader()
        train, valid, test = my_loader.load_data('data/train.csv', 'data/test.csv')
        X_train, y_train, y_train_registered, y_train_casual = my_loader.create_data(train, self.input_cols)
        X_valid, y_valid, y_valid_registered, y_valid_casual = my_loader.create_data(valid, self.input_cols)
        self.grid_search(X_train, y_train_registered)
        self.grid_search(X_train, y_train_casual)
       
    def linear_regression_train(self, X, y):
        lin_reg_model = linear_model.LinearRegression()
        return lin_reg_model.fit(X,y)
    
    def adaboost_train(self, X, y):
        adaboost_model = AdaBoostRegressor(n_estimators=100)
        adaboost_model.fit(X,y)
        return adaboost_model
        
    def gradient_boosting_train(self, X, y):
        params = {'n_estimators': 100, 'max_depth': 5, 'random_state': 0, 'min_samples_leaf' : 10, 'learning_rate': 0.1, 'subsample': 0.7, 'loss': 'ls'}
        gb_model = GradientBoostingRegressor(**params)
        gb_model.fit(X,y)
        return gb_model
        
    def random_forest_train(self, X, y):
        params = {'n_estimators': 1000, 'max_depth': 15, 'random_state': 0, 'min_samples_split' : 5, 'n_jobs': -1}
        random_forest_model = RandomForestRegressor(**params)
        model = random_forest_model.fit(X, y)
        return model
    
    def model_predict(self, model, X):
        y_predict = model.predict(X)
        return y_predict
        
     
    def train(self):
        my_loader = Loader()
        train, valid, _ = my_loader.load_data('data/train.csv', 'data/test.csv')
        X_train, y_train, y_train_registered, y_train_casual = my_loader.create_data(train, self.input_cols)
        X_valid, y_valid, y_valid_registered, y_valid_casual = my_loader.create_data(valid, self.input_cols)
               
        model_registered = self.gradient_boosting_train(X_train, y_train_registered)
        y_predict_registered = np.exp(self.model_predict(model_registered, X_valid)) - 1
        
        model_casual = self.gradient_boosting_train(X_train, y_train_casual)
        y_predict_casual = np.exp(self.model_predict(model_casual, X_valid)) - 1
        
        y_predict_count = np.round(y_predict_registered + y_predict_casual)
        
        rmsle = self.get_rmsle(y_predict_count, np.exp(y_valid_registered) + np.exp(y_valid_casual) - 2)
        print(rmsle)
        self.predict('data/test.csv', model_registered, model_casual)
        
    def predict(self, path, model_registerd, model_casual):
        my_loader = Loader()
        test_data = my_loader.read_data(path)
        X = test_data[self.input_cols].as_matrix()
        y_registered = np.exp(model_registerd.predict(X)) - 1
        y_casual = np.exp(model_casual.predict(X)) - 1
        y_predict_count = np.round(y_registered + y_casual)
        
        test_data['count'] = y_predict_count
        output = test_data[['datetime', 'count']].copy()
        output.to_csv('submit.csv', index = False)
        print ('A')
        
        
        
        
    
if __name__ == "__main__":
    my_trainer = Trainer()
    my_trainer.set_input_cols()
    my_trainer.train()