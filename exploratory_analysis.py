# * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# 
# Exploratory data analysis
#
#
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import missingno as msno
import seaborn as sns

cwd = os.getcwd()

datadir = f"{cwd}/data"
imgdir = f"{cwd}/img"

train = pd.read_csv(f"{datadir}/train.csv")
test = pd.read_csv(f"{datadir}/test.csv")

# import pdb; pdb.set_trace()

# number of zeros in budget column (812)
train_num_zero_budget = (train["budget"] == 0).astype(int).sum(axis=0)

# look at general info of our train data
print("\nData types")
train.info()

# separate categorical and numerical
train_cat = train.columns[train.dtypes == 'object']
# train_num = train.columns[train.dtypes != 'object'].drop(['id'], axis=1).fillna(0)
# train_numeric = train.select_dtypes(['number']).drop(['id'], axis=1).fillna(0)
train_num = train.columns[train.dtypes != 'object']
# print("\nNumeric variables ", len(train_numeric))
print("\nNum variables ", len(train_num))
print("\nCat variables ", len(train_cat))

# get some plots
sns.displot(train.revenue)

#msno.matrix(train.sample(500))
#plt.savefig(f"{imgdir}/missing_box_office_data1.png", bbox_inches='tight')
#plt.close()
