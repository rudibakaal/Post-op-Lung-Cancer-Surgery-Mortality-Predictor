# Post-op-Lung-Cancer-Surgery-Mortality-Predictor

## Motivation
A binary classification neural network used to predict post-operative life expectancy in lung cancer patients

The data was collected retrospectively at Wroclaw Thoracic Surgery Centre for patients who underwent major lung resections for primary lung cancer,  class 0 - death within one year after surgery, class 1 - survival.[1].

One hot encoding of the categorical features was implemented with the pandas cat.codes class. All features were standardised via sklearn's StandardScaler class. The categorical labels were also numerically converted using sklearn's Labelencoder class.

## Neural Network Topology and Results Summary

The binary-crossentropy loss function was leveraged along with the rmsprop optimizer for this classification problem.


![model](https://user-images.githubusercontent.com/48378196/96961401-4be81500-1550-11eb-9cd2-4e0f682c3b56.png)

After 50 epochs the binary and validation classifiers reach 90% and 83% accuracy resepctively, in predicting post-operative surgery survival. 

![Post-op-Lung-Cancer-Surgery-Mortality-Predictor](https://user-images.githubusercontent.com/48378196/109084156-57945400-775b-11eb-81ff-4aacf9f6c2a6.png)


## License
[MIT](https://choosealicense.com/licenses/mit/) 

## References
[1] ZiÄ™ba, M., Tomczak, J. M., Lubicz, M., & ÅšwiÄ…tek, J. (2013). Boosted SVM for extracting rules from imbalanced data in application to prediction of the post-operative life expectancy in the lung cancer patients. Applied Soft Computing
