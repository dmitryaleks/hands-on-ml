# Notes for "Hands-On Machine Learning with Scikit-Learn &amp; TensorFlow"

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
