# Kaggle M5 competition: Walmart store forecasting

Top 4% solution for the Kaggle M5 competition. The competition requires prediction store sales of individual
items over a prediction period of 28 days.

## Modelling approach

The code is quite short (<300 lines) and uses only fairly basic features in a LightGBM model. I didn't use any "magic" adjustment
factors. I also didn't use any custom metrics, just rmse. I think the evaluation metric is noisy, especially for features 
with short history, because random fluctuations in the day-to-day sales history can cause products to be weighted very 
differently even if they have similar long-term average. So I thought trying to optimise for this metric would lead to 
overfitting.

Rather than using a recursive approach, I trained separate models for each day of the forecasting horizon, and for each `n` I recalculated the features 
so that the `n`-day-ahead model is trained on data that has been lagged by `n` days. Based on discussions in the forum (specifically, 
this post https://www.kaggle.com/c/m5-forecasting-accuracy/discussion/144067#),
I decided that the recursive approach was only performing well on the training period by coincidence.

I noticed that in the test period, there are very few new products (i.e. product that have not been sold 
before the test period). So I excluded from the training set rows before the first sale date of a product in a 
store, and also excluded these rows when calculating aggregate features.

I used 3 years of data to calculate the features (to reduce noise and capture seasonal trends) and 1 year to actually 
train the model. I excluded December from the training period because of the effect of Christmas.

## Features 

The feature engineering is mainly common sense: as well as the obvious date features, just lagged sales at various 
levels of aggregation. For the aggregated features, I took the mean of sales at 3 levels of aggregation:
 - item and store
 - item (aggregated over all stores)
 - dept id and store id
 The idea of this was that the higher levels of aggregation provide a less noisy view of item-level and store-level trends.

Specifically, the features are:
 - dept_id and store_id (categorical)
 - day of week, month, snap (i.e. is today a snap day for the current store)
 - days since product first sold in that store
 - price relative to price 1 week and 2 weeks ago
 - item-level holiday adjustment factor (for each holiday and each item, calculate the average change in sales in the week
 leading up to the holiday and the holiday itself)
 - long-term mean and variance of sales at the 3 levels of aggregation
 - long-term mean and variance of sales at the 3 levels of aggregation for each day of week
 - average of last 7, 14, and 28 days of sales at the 3 levels of aggregation
 - average sales lagged 1-7 days at the 3 levels of aggregation
 