# Using Machine Learning to Predict Prices in New York City Airbnbs

## Wenxuan Cai, Jerry Hong, Josh Jacobs, Jin Lee


![Header](https://github.com/joshuajacobs2020/AirbnbPricePrediction/blob/main/Figures/Header.jpg?raw=true)

## Introduction

Tourists visiting New York may have many available options for an Airbnb, with around 50k-60k options available in 2019. How are tourists supposed to know what areas around NYC give travelers the best deals? Additionally, what variables most impact the price of the listing? What factors seems to be the most appealing to tourists when getting an Airbnb.

Considering the amount of options, and the fact that the prices of accommodations can be very volatile and can greatly fluctuate. By better understanding these influences on prices, we can attempt to give reccomendations on where tourists should stay. We will do this by using machine learning models to predict the prices of Airbnbs and estimate whether a listing is a “good” or “bad” deal. This can help better inform users of making the right purchasing decision knowing the price accurately reflects the listing’s features.

We will be building this model from the New York City Airbnb Open Data dataset available on [Kaggle](https://www.kaggle.com/datasets/dgomonov/new-york-city-airbnb-open-data).

Predicting prices and evaluating the features tourists care most about has a wide variety of applications. For instance, Expedia does this for hotel bookings around the world by giving each room a rating from 1 to 5. There are many other possible applications of this work. In this process, we will be able to discover what models seem to perform the best at predicting prices. Price prediction has a wide application to other markets and services dedicated to giving consumers more information on products, and these models could be applied outside of Airbnb bookings.

## Summary Statistics - Predictors
The New York City Airbnb Open Data dataset contains both numerical and categorical data for a total 48,895 observations. The dataset contains 16 variables regarding Airbnb listings such as information regarding the host, review, location, details about the listing, and price. The variable being predicted was price and 16 features were used for our models which includes dummy variables we created (explained two sections down).

<img src="https://github.com/joshuajacobs2020/AirbnbPricePrediction/blob/main/Figures/SummaryTablePredictors.png?raw=true" alt="drawing" width="600"/>

## Summary Statistics - Price

The average price to stay in a New York Airbnb for one night is about \$152. But what is the overall distribution of prices? As shown in the histogram below, there is a significant rightward skew in prices with most Airbnbs being around \$100 a night and a portion being very expensive, even reaching up to \$10,000 a night.

![Histogram](https://github.com/joshuajacobs2020/AirbnbPricePrediction/blob/main/Figures/HistogramPrice.png?raw=true)

We can see if there are any geographic trends with Airbnb price levels. In the map shown below, we can that Manhattan is a very expensive to stay. This makes sense considering Times Square, Central Park, and many other tourist attractions are in this borough.

![Map](https://github.com/joshuajacobs2020/AirbnbPricePrediction/blob/main/Figures/MapPrices.png?raw=true)

## Setup - Creating Text Features

Each Airbnb observation comes with a unique name created by the host. These often include relevant information about the listing, such as "Clean and Quiet Home by the Park." These names may capture unmeasured factors about listings that may be related to price. In the process below, we use PCA to obtain 6 text features capturing different aspects of these texts.

1. Filter Airbnb names to high frequency, meaningful words
    - Reduces to 49 keywords

2. Create dummy variables for each keyword indicating whether the Airbnb name contains the keyword

3. Use PCA to reduce the dummy variables to 6 features

This setup is also shown in the `Data_Cleaning_Kaggle_Data.ipynb` file. Below is the code used to perform the dimension reduction.

```
# Make array of values and reduce dimensionality with PCR
text_array = np.array(airbnb_text.drop(["id","name","price","filtered_text"],axis=1))

import numpy as np
from sklearn.decomposition import PCA
pca = PCA(n_components=6)
pca.fit(text_array)
reduced = pca.fit_transform(text_array)
reduced_df_pca = pd.DataFrame(data=reduced,columns=["textfeat1","textfeat2","textfeat3","textfeat4","textfeat5","textfeat6"])
```
## Summary Statistics - Evaluating Predictors

In the heatmap below, we can see what predictors have the highest correlation with prices.

![Heatmap](https://github.com/joshuajacobs2020/AirbnbPricePrediction/blob/main/Figures/Heatmap.png?raw=true)

Most notably, we see that prices are positively correlated with Entire home/apt and Manhattan. This makes sense as entire homes/apartments would be nicer to stay in than a private or shared room. Additionally, Manhattan is a very popular tourist destination and likely to have a lot more demand for temporary housing.

Prices are negatively correlated with most of the text features, Brooklyn, Queens. This is interesting as it indicates the text features are capturing some negative aspect about the listings that drive down prices. However, these correlations are still relatively small, only being about -.14 for the first text feature.

Text features themselves appear to be most correlated with entire/home apt and the different boroughs. These text features, however, are also ideally correlated with unmeasured factors about each listing. This additional information should help us in predicting Airbnb prices.

## Setup - Experiment Design
Using Google Collab, we will use each predictor to predict prices in a variety of different models:

*   OLS Regression
*   Regularization Methods
*   Partial Least Squares
*   Decision Trees
*   Random Forest
*   Boosting

For each model we will select needed tuning parameters using cross validation.

We will then choose the model with the lowest test root mean squared error (RMSE) to predict the values of all Airbnb bookings in the sample. 

Our last step will be to assess the value of the booking by comparing the predicted to the actual price.


## Results

With the cleanned Dataset, we conducted the following experiments. We will utilize OLS regression, regularization methods, PCA/PLS, Decision Tree/Random Forests, and Boosting the determine the optimal method to predict Airbnb prices based on the features of our dataset. We will use the lowest train/test RMSE to determine the best performing model. In addition, we will compare these predicted prices to the actual prices and set a baseline value in the residuals to determine if the prices is a good deal.

### OLS Regression
As a baseline, we start by fitting an OLS linear regression to our training set. We utilize all numeric features that are related to our overarching question. First is to run to run a robust scaler to scale the features that are robust to outliers.

```
from sklearn.preprocessing import RobustScaler
from sklearn.linear_model import LinearRegression
r_scaler = RobustScaler()
X_train = r_scaler.fit_transform(X_train)
X_test = r_scaler.transform(X_test)

linreg = LinearRegression()
linreg.fit(X_train, y_train)
```
The results generated by this regression leads to a training RMSE of 193.41 and a testing RMSE of 142.81.

### Regularization

For using lasso and ridge regressions, the procedure is as follows:
1.   Standardize predictors
2.   Select λ≥0 with k=5 folds cross-validation
3.   Run regression on test set with optimal lambda
4.   Obtain Test and Training MSE

For lasso regression, the code for this procedure is shown below. Ridge regression is very similar, except we set `l1_ratio=1.

The full code is available in the `Regularization Methods.ipynb` notebook.

```
from sklearn.linear_model import ElasticNetCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

# Setting random_state, since I use LOOCV this shouldn't be too important
np.random.RandomState(123)

# Setting tuning parameters to check
lambdas = 10**np.linspace(4, -6, 101)

# Cross Validating for optimal lambda
ridgecv = ElasticNetCV(alphas=lambdas, cv = None, l1_ratio=0.) # perform LOOCV
scaler = StandardScaler(with_mean=True, with_std=True)

# Define pipeline steps to scale and then do cross-validation
pipeCV = Pipeline(steps=[('scaler', scaler), ('ridgecv', ridgecv)]);
pipeCV.fit(X_train, y_train);

# Get optimal lambda
tuned_ridge = pipeCV.named_steps['ridgecv']
tunedlambda_ridge  = tuned_ridge.alpha_
```
For lasso regression, the cross-validated value for lambda is `0.0001`. From this regression, we get a training RMSE of 222.82 and a test RMSE of 243.15.

Ridge regression yields an optimal lambda of ~0.0025 and identical training and test RSME.

In the graph below, we can see how the standardizded coefficients of our lasso regression changes as lambda changes.

![Shrinkages](https://github.com/joshuajacobs2020/AirbnbPricePrediction/blob/main/Figures/LassoShrinkages.png?raw=true)

In the table below, I compare the standardizded coefficients for the OLS regression with the coefficients at the optimal lambda values of the ridge and lasso regressions. As we can see, the coefficients are very close to each other. This makes sense considering our lambda values are very small. The table also shows that Airbnb locations (Brooklyn, longitude, etc.), entire home/apt, and the 5th text feature are most important in predicting prices.

Overall, we can conclude the regularization methods did not do much to improve the baseline OLS models.

![Coefficients](https://github.com/joshuajacobs2020/AirbnbPricePrediction/blob/main/Figures/CoefficientsRegularization.png?raw=true)

### PCA

To conduct the PCA, we first used the standardized predictors, then try the number of components from 1 to 20, and fit the PCA.

We calculate the training and testing RMSE and find the optimal testing RMSE here.

```
from sklearn.decomposition import PCA # type: ignore
from sklearn.linear_model import LinearRegression # type: ignore
from sklearn.metrics import mean_squared_error # type: ignore
import numpy as np # type: ignore
from sklearn.preprocessing import StandardScaler # type: ignore
df = data

X = df.drop('price', axis=1)
y = df['price']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train_standardized = (X_train - X_train.mean()) / X_train.std()
X_test_standardized = (X_test - X_train.mean()) / X_train.std()
MSE_train_pcr_arr = np.ones(20)
MSE_test_pcr_arr = np.ones(20)

for i in range(1,21):
    pca = PCA(n_components=i)
    X_train_pca = pca.fit_transform(X_train_standardized)
    X_test_pca = pca.fit_transform(X_test_standardized)

    # Fit the model
    model_pcr = LinearRegression()
    model_pcr.fit(X_train_pca, y_train)

    # get training MSE
    MSE_train_pcr = mean_squared_error(y_train, model_pcr.predict(X_train_pca))
    MSE_train_pcr_arr[i-1] = MSE_train_pcr
    # Get test MSE
    MSE_test_pcr = mean_squared_error(y_test, model_pcr.predict(X_test_pca))
    MSE_test_pcr_arr[i-1] = MSE_test_pcr

print("PCR Model with 1 to 20 Predictors:")

RMSE_train_pcr_arr = np.sqrt(MSE_train_pcr_arr)
RMSE_test_pcr_arr = np.sqrt(MSE_test_pcr_arr)
print("Training MSE:", RMSE_train_pcr_arr)
print("Test MSE:", RMSE_test_pcr_arr)

## This part of code generate the plot
import matplotlib.pyplot as plt
import numpy as np

i = np.arange(1, 21)
plt.figure()
plt.plot(i, RMSE_train_pcr_arr, label='RMSE Train PCR', marker='o', linestyle='-', color='b')
plt.plot(i, RMSE_test_pcr_arr, label='RMSE Test PCR', marker='x', linestyle='--', color='r')
plt.legend()
plt.xlabel('Number of Principal Components')
plt.ylabel('Root of Mean Squared Error')
plt.title('RMSE for Training and Testing with PCR')
plt.show()
```
Here we generate the plot of RMSE and the number of the principle components. We realized that $n = 6$ is already sufficient for testing dataset, with Train RMSE as 194.06, and Test RMSE 191.88.

## PLS
Along with PCR, we also utilized a PLS regression to our training data. First, the predictors were standardized in pipeline, then the optimal number of components for PLS was found through k=5 folds cross-validation. The model was refit using the optimal number of components and the training and test RMSE were found.

```
pls = PLSRegression(n_components=5)
scaler = StandardScaler(with_mean=True, with_std=True)

# defining a pipeline to scale then run pls
pipe = Pipeline([('scaler', scaler), ('pls', pls)])
pls.fit(X_train, y_train)

K=5
kfold = skm.KFold(K, random_state=1, shuffle=True)

param_grid = {'n_components':range(1, 15)}
grid = skm.GridSearchCV(pls,
                        param_grid,
                        cv=kfold,
                        scoring='neg_mean_squared_error')
grid.fit(X_train, y_train)

# rerunning PLS with optimal number of components
pls = PLSRegression(n_components=9,
                   scale=True)
pls.fit(X_train, y_train)
```

Refitting our model and calculating the train and test RMSE results in a train RMSE of 225.49 and test RMSE of 243.83.

### Decision Tree

Before running the decision tree, we first need to find the max depth to limit the number of split nodes of the tree to alleviate overfitting and poor genearlization of unseen data. We will use kfold of 5 cross validation to ensure the optimal number of split nodes for this tree.

```
from sklearn.tree import DecisionTreeRegressor as DTR
from sklearn.model_selection import GridSearchCV
import sklearn.model_selection as skm

tree = DTR(random_state=1)

kfold = skm.KFold(5,
                  shuffle=True,
                  random_state=1)

max_depth_range = range(1, 20)
param_grid = dict(max_depth=max_depth_range)
grid = GridSearchCV(tree, param_grid, refit=True, cv=kfold, scoring='neg_mean_squared_error')

grid.fit(X_train, y_train)

grid.best_params_
```

After running this, we found that the optimal max depth is 3. This falls within our expectations as we think that given the 22 features used in our data set, we think it is appropriate to use a 1/4 of the features or around 4-5 as the max depth.
With this, we can now proceed to run the decision tree model with this parameter.

```
# decision tree with max_depth
tree_optimal = DTR(max_depth = 3, random_state=1)
tree_optimal.fit(X_train, y_train)

y_pred_tree = tree_optimal.predict(X_test)
results_tree = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred_tree})
results_tree.head(10)

# important features for tree
feature_names = X.columns
feature_imp = pd.DataFrame({'importance':tree_optimal.feature_importances_}, index=feature_names)
feature_imp.sort_values(by='importance', ascending=False).head(10)
```
The results generated leads to a training RMSE of 182.95 and a testing RMSE of 142.72. We also checked the top ten most important features from our decision tree and the following features are:


1. days_since_last_review
2. Entire home/apt
3. longitude
4. minimum_nights
5. availability_365
6. textfeat5
7. latitude
8. textfeat4
9. textfeat3
10. textfeat2

We generally expect that location plays an important factor in the pricing of Airbnbs. However, the top two important features are worth discussing as they are not location-based nor logitics (i.e. nights, though it is the fourth- most important). The number of days since last review tops the list, which could make sense as the more recent review could sway the demand towards that listing, especially if it is a positive review. The dummy variable of whether the listing is of type home or apartment is something many travlers would highly consider. It makes sense that travlers would prefer spaces that can span an entire home or apartment for flexibility of living accommodations, which is why this feature is the second-most important.

### Random Forests

Before runnning the random forests model, we first need to find the maximum number of features for each tree iteration. This ensures that we have right balance of the features used in each tree while taking account overfitting and representation of all features. Since we are dealing with 22 features, having a relatively smaller number of max features allows the chance for all feature to be involved in the model. We also set the default number of tree iterations as 100 for simplicity. We iterate from the total number of features and return the test error for each number of features used in the model. We pick the max number that corresponds to the lowest testing error.

```
# find the optimal number of features
from sklearn.ensemble import RandomForestRegressor as RFR
test_errs = list()
num_estimators = range(1, X_train.shape[1])
for m in num_estimators:
    rf = RFR(max_features=m, random_state=0).fit(X_train, y_train)
    y_hat_RF = rf.predict(X_test)
    err = np.mean((y_test - y_hat_RF)**2)
    print(m, err)
    test_errs.append(err)
```
We found that the optimal number of features in the random forests model is set to five. From here, we will run the model using the 100 tree iterations, where each tree will randomly choose five features, and for each tree would have five different features. 

```
# random forest with max features set from above
rf = RFR(max_features=5, random_state=1)
rf.fit(X_train, y_train)

# get predictions
y_pred_rf = rf.predict(X_test)
results_rf = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred_rf})
results_rf.head(10)

# important features in random forest
feature_names = X.columns
feature_imp = pd.DataFrame({'importance':rf.feature_importances_}, index=feature_names)
feature_imp.sort_values(by='importance', ascending=False).head(10)
```

The results generated from the random forests model leads to a training RMSE of 71.26 and a testing RMSE of 124.74. Despite the considerable increase between the training and testing RMSE, the random forests model seems to perform the best out of the other models. This may be because since random forests take the average for every bootstrapped tree, it performs very well on the training set and testing sets relative to the other models. We also checked the ten most-important features:

1. longitude
2. days_since_last_review
3. minimum_nights
4. latitude
5. availability_365
6. textfeat6
7. textfeat1
8. textfeat3
9. textfeat2
10. reviews_per_month

We see from the random forest model, both the latitude and longitude features are among the top four most important, with longitude being the most important. This implies that the location, especially with the many neighborhoods and sections that divide up New York City plays an important role in the pricing of Airbnbs. We also see that most of the important features from the decision tree model are maintained in the random forest model, suggesting there is consistency between them. However, in the random forest model, the text features look to also play an important in pricing. The terms renters would use to market their listing can influence the pricing that best matches the descriptions of the accommodation. However, we may need to further look into what kind of keywords that hs the biggest influence in pricing.

### Boosting

Here we conducted a grid search to identify the optimal parameters for our Gradient Boosting Regressor model. Once you found the best parameters, we refitted the model with those settings. To evaluate performance, we calculated the root mean square error (RMSE) for both the training and test datasets.

```
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GridSearchCV, train_test_split

# Define parameter grid
param_grid = {
    'n_estimators': [10, 25, 50],
    'learning_rate': [0.05, 0.1],
    'max_depth': [5,10]
}

# Initialize GradientBoostingRegressor
gb_model = GradientBoostingRegressor(random_state=123)

# Perform grid search
grid_search = GridSearchCV(estimator=gb_model, param_grid=param_grid, cv=5, scoring='neg_mean_squared_error')
grid_search.fit(X_train, y_train)

# Get best parameters
best_params = grid_search.best_params_

# Fit model with best parameters
best_gb_model = GradientBoostingRegressor(**best_params, random_state=123)
best_gb_model.fit(X_train, y_train)

# Predict on training set
y_train_pred = best_gb_model.predict(X_train)
# Calculate training MSE
train_mse = mean_squared_error(y_train, y_train_pred)

# Predict on test set
y_test_pred = best_gb_model.predict(X_test)
# Calculate test MSE
test_mse = mean_squared_error(y_test, y_test_pred)

print("Best parameters:", best_params)
print("Training MSE:", train_mse)
print("Test MSE:", test_mse)
print("Training RMSE:", np.sqrt(train_mse))
print("Test RMSE:", np.sqrt(test_mse))
```

## Discussion

Now that we have discovered the Random Forest model is performing the best at predicting Airbnb prices, we can now evaluate the value of each Airbnb listing. We can obtain the residual prices for each listing as shown in the code below:
```
# random forest with max features set from cross validation
from sklearn.ensemble import RandomForestRegressor as RFR
rf = RFR(max_features=5, random_state=1)
rf.fit(X_train,y_train)

# Get predicted prices
df['predicted price'] = rf.predict(df[Xcols])

# Obtain residuals
df['resid'] = df['price'] - df['predicted price']
```
For each listing, if the residual is greater than 0, this suggests the host overpriced the booking. If residual is less than 0, this suggests the host underpriced the booking. A consumer can then look at these residuals to see the best value Airbnbs to stay in.

These residuals are relatively close to 0, suggesting the model tends to slightly overvalue the price of airbnb bookings. Most bookings are relatively close to the predicted value, but the standard deviation of prices is \$159. This suggests the outliers we observed in the histogram of prices may be very hard to predict, as some residuals even exceed \$9917.

<img src="https://github.com/joshuajacobs2020/AirbnbPricePrediction/blob/main/Figures/ResidsTable.png?raw=true" alt="drawing" width="200"/>


Below shows a map of the best value bookings across New York City. We can see that Manhattan, as predicted, has the most overpriced bookings. Still, there are also many underpriced bookings across the city. It's just about knowing which ones to choose. A tourist can use this model to find good deals on Airbnbs, even in the busiest parts of New York.

<img src="https://github.com/joshuajacobs2020/AirbnbPricePrediction/blob/main/Figures/MapResids.png?raw=true" alt="drawing" width="800"/>

## Future Directions

Regarding future directions for this study, there are a few factors to consider. One is to further explore the impact of seasonal and holiday dates. It is true during certain times of the year are peak travel seasons. Summer and winter vacation are two of the most high-volume periods where travelers would venture to their favorite places. This would lead for Airbnb tenants to raise prices to match the high demand during these peak seasons. Given the date of the listing, we would categorize if the it is during a peak traveling season and implement this factor to our models. Another is to consider the accessibility to popular tourist destinations. Besides the neighborhood and the borough the Airbnb is in, it would be worth considering to see if these tourist spots that are close proximity to the Airbnb would affect the pricing. Given visitors would want to visit these locations, they may be more willing to pay for a higher rate if the Airbnb is close by. Not just tourist spots but also neccessities like restaurants and transportation hubs would be viable for travelers and having an Airbnb that connects these locations would be highly considerable. Thus, prices may reflect the demand for locations that are accessible to these spots. Finally, due to the sheer number of listings in the Airbnb database, especially in a location like New York City, there are bound to be illegitimate listings to target uninformed travelers. We may an extension of this study to collect data on Airbnb scams and do a classifier study to determine if the features can indicate a legitimate Airbnb listing.

## References

New York City Airbnb Open Data (2019).
https://www.kaggle.com/datasets/dgomonov/new-york-city-airbnb-open-data
