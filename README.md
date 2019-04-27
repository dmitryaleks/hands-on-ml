# Notes for "Hands-On Machine Learning with Scikit-Learn &amp; TensorFlow"

## Data evaluation

First, consider plotting histograms for each feature and label values in order to learn about data ranges and distributions:

  * if some features are distributed in a tail-heavy manner - consider applying a transformation that would make them closer to a Gaussian distribution;
  * consider feature scaling if not all of the features are approximately on the same scale.

Identify missing data - consider filling missing values with median values for a given feature.

### Set Test set aside

To avoid data snooping - extract and set aside the Test Set early on.

Selection of test set data points can be done:

  * by randomly sampling the full set: there is a risk of introducing a sampling bias if some categories in the original set will end up being underrepresented in the test set;

  * by appying a Stratified Random Sampling: this method ensures that necessary categories of features are represented in the test set with a minimum bias.
