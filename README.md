# Car_price_predicion_with_service

I have successfully built a ridge regression model that can predict car prices with high result. The r_2 score on the test data was 0.88, which means that the model can explain 88% of the variance in the target variable. 

I have applied various data processing techniques to improve the performance of the model:

- removing outliers, transforming features and target variable with logarithms, and creating new features from existing ones. The most significant boost in quality came from data processing, especially outlier removal, logarithmic transformation, and adding categorical features.

- I have also used cross-validation to find the best hyperparameters for the model

- I have saved the parameters of the trained model and created a fast api service that can provide car price predictions based on user input.

Some of the challenges and limitations that I faced during the work were:

- The dataset had some missing values and noisy data that I had to deal with. I used different strategies for imputing or dropping missing values, and for detecting and removing outliers.

- The dataset had some features that were not normally distributed, which could affect the assumptions of the linear regression model. I used histograms and boxplots to visualize the distribution of the features, and applied logarithmic transformation to make them more normal.

- The dataset had some categorical features that had to be encoded into numerical values. For this I used one hot encoding.

Below you can see the result of my service 👽. 

