# RLBoost
Data quality or data evaluation is sometimes a task as important as collecting a large volume of data when it comes to generating accurate artificial intelligence models. In fact, being able to evaluate the data can lead to a larger database that is better suited to a particular problem because we have the ability to filter out data obtained automatically of dubious quality.

RLBoost is an algorithm that uses deep reinforcement learning strategies to evaluate a particular dataset and obtain a model capable of estimating the quality of any new data in order to improve the final predictive quality of a supervised learning model. This solution has the advantage that of being agnostic regarding the supervised model used and, through multi-attention strategies, takes into account the data in its context and not only individually.

![RLBoost](./doc/images/RLBoost.png)

# Installation
To install ```rlboost``` simply run:
```
pip install git+https://github.com/iiconocimiento/rlboost
```

# Standalone usage example
```python 
from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

from stable_baselines3 import PPO
from rlboost import RLBoostOutlierDetector

# Getting classification data
data = load_breast_cancer(as_frame=True)
df = data["data"]
df["target"] = data["target"]

# Building datasets
df_train, df_test = train_test_split(df, test_size=0.3, train_size=0.7)
df_train, df_valid = train_test_split(df_train, test_size=0.5, train_size=0.5)

X_train, y_train = df_train.drop(
    columns=["target"]).values, df_train["target"].values
X_valid, y_valid = df_valid.drop(
    columns=["target"]).values, df_valid["target"].values
X_test, y_test = df_test.drop(
    columns=["target"]).values, df_test["target"].values

# RLBoost fitting

# Estimator to refit iteratively (override score function as needed)
estimator = LogisticRegression()

rlboost = RLBoostOutlierDetector(X_valid, y_valid,
                                 agent_class=PPO,
                                 steps=10,
                                 estimator=estimator,
                                 agent_batch_size=64,
                                 use_vf_encoding=True,
                                 use_score_base=True,
                                 device="cpu")
rlboost.fit(X_train, y_train)

# RLBoost evaluating data
# (usually only train, but you can evaluate any data you want)
data_values_train = rlboost.get_data_values(X_train, y_train)
data_values_valid = rlboost.get_data_values(X_valid, y_valid)
data_values_test = rlboost.get_data_values(X_test, y_test)
print("Done")
```

# Explanation
RLBoost is a data evaluation algorithm based on deep reinforcement learning. This algorithm tries to maximize the difference between the score of an estimator trained on N data chosen from a dataset versus the score of the same estimator trained on the same data with prior filtering. 

This score is calculated with the default score method of the proposed estimator (usually accuracy for classification problems).

## Parameters
* **```agent_class: stable_baselines3.common.on_policy_algorithm.OnPolicyAlgorithm```** 
  * Reinforcement Learning algorithm to use. It has to be an ActorCritic one that accepts Boolean actions.
* **```steps: int```** 
  * Number of steps to run the algoritm. 1e4 is recommended.
* **```estimator: sklearn.base.BaseEstimator```** 
  * Estimator to wrap and use to evaluate the data
* **```agent_batch_size: int```** 
  * Number of samples usead in each iterations. Ussually 200 should be fine.
* **```use_vf_encoding: bool```** 
  * Whether to use a TransformerEncoder for the value function estimation. False for speedup, but true is recommended.
* **```use_score_base: bool```** 
  * Whether to use the baseline score for the value function estimation. True is recommended.
* **```device: str```** 
  * Device to train the model. You can choose from ```cpu```, ```cuda:0```, ```cuda:1```, etc. 
  
# Use Cases
- Data quality checker.
- Automatic data augmentation.

# Limitations
- Only classificaction problems (by now).
- Validation set has to be very precise.