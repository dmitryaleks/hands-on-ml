# Notes for "Hands-On Machine Learning with Scikit-Learn &amp; TensorFlow"

[Jupyter Notebook for the Regression Model](regression/regression-model.ipynb)

## Data evaluation

First, consider plotting histograms for each feature and label values in order to learn about data ranges and distributions:

  * if some features are distributed in a tail-heavy manner - consider applying a transformation that would make them closer to a Gaussian distribution;
  * consider feature scaling if not all of the features are approximately on the same scale.

Identify missing data - consider filling missing values with median values for a given feature.

### Extract and set the Test Set aside

To avoid data snooping - extract and set aside the Test Set early on.

Selection of test set data points can be done:

  * by randomly sampling the full set: there is a risk of introducing a sampling bias if some categories in the original set will end up being underrepresented in the test set;

  * by appying a Stratified Random Sampling: this method ensures that necessary categories of features are represented in the test set with a minimum bias.

### Discover and Visualise the data

#### Looking for correlated features and correlation of some features with label values

Make a correlation matrix for the input features.

Pandas allows you to do so on a DataFrame as follows:
```
corr_matrix = data.corr()
```

Correlation matrix contain linear correlation coefficients (Pearson's r) for each pair of features.

Note that this is a coefficient of linear correlation and it will not indicate presence of non-linear correlations. Those can be discovered visually by means of examining Scatter Matrices.

Visualise feature/label distributions by using Pandas' scatter_matrix function:

```
from pandas.tools.plotting import scatter_matrix

scatter_matrix(data, figsize=(12,8))
```

This will visualise the scatter plot for each pair of features as well as label values.

### Clean-up and normalize the data

#### Filling a missing data

Can use pandas.DataFrame.fillna() to fill missing feature values with median/mean values.

Also, Imputer class from Scikit-Learn allows filling missing data with calculated median values for each feature automatically.

#### Converting Categorical features into a numberical format

  * LabelEncoder class converts text labels to numbers automatically (with this representation there is a risk of ML model thinking that some categories are closer to each other semantically based on their numberical value then they in fact are);
  * OneHotEncoder class converts each category type into a one-hot encoded vector. This is required as otherwise ;
  * LabeBinarizer coverts each categorical feature into a set of one-hot encoded vectors for each category.

#### Feature scaling

Two main types:
  * min-max scaling (normalization): converts to range [0, 1]: implemented by MinMaxScaler class;
  * Standardization: converts to range [-1, 1]: implemented by StandardScaler class.

#### Organizing data transformations into Pipelines

Pipeline is a class of Transformer type that allows organizing Transformers into an ordered sequence.

### Use Validation Set to tune hyper-parameters of the model

Sometimes it is beneficial to use Cross-Validation to evalue generalization power of the model. sklearn has a cross_val_score function that facilitates such evaluation.

### Try using alternative models

For example, for a regression problem one may easily try any of the following regressors:

  * LinearRegressor;
  * DecisionTreeRegressor;
  * RandomForestRegressor.

### Save various models and validation scores for the purpose of future reference

This can be done either using "pickle" package or using a "joblib" module.

### Fine-tune your model

Use GridSearchCV or RandomSearchCV to find the best combination of hyperparameters by looking at the results of cross-validation.

Consider using Ensemble Methods: where an ensemble of simpler top-performing models may produce a better result than a single sophisticated model.

Calculate relative importance of features

### Do evaluation on a Test Set

You'd expect to see scores that are slightly (or a lot) worse that when model was evaluated on a Validation Set as we picked the best hyperparameters to get the best performance on a Validation Set.

Do not tweak hyperparameters at this point. Though doing so would bring better score on the Test Set, it may lead to a worse perormance on the unseen new data.

### Launch, monitor and maintain your ML system

Make sure to evaluate the model performance consistently and regularly, as models tend to 'rot' as data evolves over time, unless models are regularly trained on fresh data.

Automate training of your model so that you could re-train your model pretty often (otherwise model performance will be inconsistent over time, improving only after infrequent re-trainings and degrading untill the next far standing retraining happens).

Monitor system inputs as they may get stale or start getting erroneous values.





