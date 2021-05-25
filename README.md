# data_science_projects

|Project description|Data used |Tasks |Conclusion |Libraries used |
|:-|:-|:-|:-|:-|
|Predicting the probability of bank customers' churn |Dataset with customers' data and their actions from the source: https://www.kaggle.com/barelydedicated/bank-customer-churn-modeling|<ul><li>Prepare data for analysis;</li><li>Look at the features distribution</li><li>Look at the models' results with the default parameters</li><li>Use several ways to balance the classes: balancing them in hyper-parameters, upsampling and downsampling</li><li>Test the best model and prepare a summary</li></ul>|Since the classes are not balanced, the optimal prediction model is the random forest model trained on a set with an increased positive class with parameters: number of estmators: 80, max depth: 10, criterion='gini'. We managed to get f1=0.61 on the test sample <ul>|<ul><li>Pandas</li><li>Numpy</li><li>Matplotlib.pyplot</li><li>sklearn</li></ul>|
