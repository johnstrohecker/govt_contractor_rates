# govt_contratctor_rates
Building a predictive model for government contractor rates using data from https://calc.gsa.gov/

The focus of this project is to develop a useful predictive model to forecast government contractor rates for firms working with the US federal government. 

The initial data provided was found on https://calc.gsa.gov/ and downloaded to CSV

Steps completed to date (8/11/19)
1.  added features to data set for keyworks located in Labor Category Description
2.  built Random Forest, Decision Tree, and Linear Regression Models
3.  built scoring mechanism for all models (RMSE)
4.  built plots to show relative accuracy for all models

Planned next steps:
1.  parse SINs into additional features of data set, and train models with new information
2.  Add xgboost model
3.  Tune hyperparameters for RF and DT
4.  Look for additional information to refine data set...data on contractors listed in file?
