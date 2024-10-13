#### Latent Factor Modeling with Expert Network for Multi-Behavior Recommendation

This is an implementation of MBLFE on Python 3

The Pytorch version is available in _requirements.txt_


#### Getting Started

First, run the data_process.py file in data/xxx to collate the data set.

you can run Tmall dataset with:

`python3 main.py --data_name tmall`

or

`./b_tmall.sh`

The optimal hyperparameters for the model are shown in the table below:

|  Dataset  |  num_experts  | ssl_tau | ssl_reg | reg_weight |  lr  |
|:---------:|:-------------:|:-------:|:-------:|:----------:|:----:|
|   Tmall   |      18       |   0.9   |   0.1   |    1e-3    | 1e-4 |
|  Taobao   |      10       |   0.9   |   0.1   |    1e-3    | 1e-3 |
|   Yelp    |      14       |   0.7   |   0.1   |    1e-3    | 3e-4 |