import pandas as pd
import mlflow
import os

# Setting the MLflow tracking server
#mlflow.set_tracking_uri('http://training.itu.dk:5000/')

# Setting the requried environment variables
# os.environ['MLFLOW_S3_ENDPOINT_URL'] = 'http://130.226.140.28:5000'
# os.environ['AWS_ACCESS_KEY_ID'] = 'training-bucket-access-key'
# os.environ['AWS_SECRET_ACCESS_KEY'] = 'tqvdSsEDnBWTDuGkZYVsRKnTeu'

#mlflow.set_experiment("<gebu> - a2")
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split, TimeSeriesSplit
from sklearn.metrics import r2_score   
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import GridSearchCV
import json
import shutil
import warnings


def run(search, name):
    with mlflow.start_run(run_name= name) as run:
        df = pd.read_json("./dataset.json", orient="split")
        df = df.dropna()

        transformer = ColumnTransformer([
                ('standard_scaler', StandardScaler(), ['Speed']), #scales the data to mean 0 and std 1
                ('num_features', PolynomialFeatures(degree= 1), ['Speed']),
                ('Categorical', OneHotEncoder(), ['Direction'])
                ])

        pipeline = Pipeline(steps = [
                ('transformers', transformer),
                ('model', GradientBoostingRegressor())])

        X = df[["Speed","Direction"]]
        y = df["Total"]

        number_of_splits = 5

        #Splitting data intro training and test data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle= False)

        
        tscv = TimeSeriesSplit(n_splits=5)
        grid = GridSearchCV(pipeline, search, scoring= 'r2',verbose=1, cv=tscv)
        pipe = grid.fit(X_train, y_train)
        r2_score_test = pipe.score(X_test, y_test)
        r2_score = pipe.best_score_
        params = pipe.best_params_
        mlflow.log_metric('r2', r2_score_test)
        mlflow.log_params(params)

        artifact_path= f"mlruns/1/{run.info.run_id}"
        mlflow.sklearn.log_model(pipe, artifact_path=artifact_path)

#Updating best model 


        try: 
          with open("best_regressor.json", "r") as f:
            best_model = json.loads(f.read())

          if r2_score_test > best_model["score"]:
            shutil.rmtree('best_model')
            print(f"New model outperforms old model")
            mlflow.sklearn.save_model(
                pipe, path = 'best_model', conda_env="conda.yml")

            model_details = {
                "model_name": name,
                "score": r2_score_test,
                }
            with open("best_regressor.json", "w") as f:
              json.dump(model_details, f)
          else: 
            print(f"Model {name} does not perform better than current saved one")

        except:
          print(f"There is no model saved yet. Let's save model: {name}")
          mlflow.sklearn.save_model(
              pipe, path = 'best_model', conda_env="conda.yml")

          model_details = {
              "model_name": name,
              "score": r2_score_test,
              }
          with open("best_regressor.json", "w") as f:
            json.dump(model_details, f)

def main():
        warnings.filterwarnings("ignore")
        svr = [{'model': [SVR()],
         'model__gamma': ['scale', 'auto'],
         'model__C': [1,5,10]}]
        rfr = [{'model': [RandomForestRegressor()],
         'model__max_depth': [i for i in range(1,8)]}]
        gbr = [{'model': [GradientBoostingRegressor()],
         'model__learning_rate': [0.1,0.2,0.3],
         'model__n_estimators': [100,200,300]}]
        run(svr, 'SVR')
        run(rfr, 'RFR')
        run(gbr,'GBR')

        
if __name__ == "__main__":
  main()

