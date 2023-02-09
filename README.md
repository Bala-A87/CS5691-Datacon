# CS5691-Datacon
Script and submission for data contest conducted as part of CS5691 - Pattern Recognition and Machine Learning.

[Link](https://www.kaggle.com/competitions/datacon-22/overview) to data contest on Kaggle for reference.

The contest allows teams of 2 to participate, and the solution in this repository is the combined work of [Hemesh D J](https://github.com/HemeshDJ) and me.

## Brief description of the problem and approach
The problem is a hotel rating prediction problem, to predict the rating a customer would award a hotel, given all the data of their booking, the hotel, and basic demographic data. 

The problem can easily be modelled as a classification problem, as labels in the train data belong to $[1, 2, 3, 4, 5]$, but due to factors such as ordering among the labels and skew towards higher rating, formulation as a regression problem yielded significantly better results.

Our best solution was obtained by averaging the results of independently-trained similarly-parametrized [HistGradientBoostingRegressors](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.HistGradientBoostingRegressor.html) available in [sklearn](https://scikit-learn.org/stable/).

More information regarding our approach and conclusions can be found in the report.

## Repository outline

### data/
Contains the data for the contest. Distributed over 7 csv files - <em>bookings_data.csv</em>, <em>bookings.csv</em>, <em>customer_data.csv</em>, <em>hotels_data.csv</em>, <em>payments_data.csv</em>, <em>train_data.csv</em>, and <em>sample_submission_5.csv</em>. Each of the first 5 files contains some part of the information joinable using foreign keys. <em>train_data.csv</em> contains ids for train data and their corresponding labels. <em>sample_submission_5.csv</em> contains ids for test data and all 5's as dummy labels.

### output/
Our prediction for the test data, in the form of a csv file, <em>preds.csv</em>.

### script/
A python script, <em>predict.py</em>, to generate the output csv file from the input data.

### report.pdf
Report describing our method, result and findings.
