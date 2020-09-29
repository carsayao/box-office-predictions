# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# 
# Exploratory data analysis
#
# Sources:  https://github.com/DachshundSovereign/Kaggle-TMDB-Box-Office/
#
#
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import missingno as msno
import seaborn as sns
# import warnings   # ignore warnings
# warnings.filterwarnings("ignore")


cwd = os.getcwd()

datadir = f"{cwd}/data"
imgdir = f"{cwd}/img"

train = pd.read_csv(f"{datadir}/train.csv")
test = pd.read_csv(f"{datadir}/test.csv")

# import pdb; pdb.set_trace()

# # number of zeros in budget column (812)
# train_num_zero_budget = (train["budget"] == 0).sum(axis=0)
# 
# # look at general info of our train data
# print("\nData types")
# train.info()
# 
# # separate categorical and numerical
# train_cat_lab = train.columns[train.dtypes == 'object']
# train_cat = train.select_dtypes(['object'])
# train_num_lab = train.columns[train.dtypes != 'object']
# train_num = train.select_dtypes(['number']).fillna(0)
# # drop id for pair plot
# train_num_no_id = train_num.drop(['id'], axis=1)
# print("\nNumerical feature labels ", len(train_num_lab))
# print("Numerical features ", train_num.shape)
# print("\nCategorical feature labels ", len(train_cat_lab))
# print("Categorical features", train_cat.shape)
# print()
# 
# # get some plots
# 
# # univariate distributions
# sns.displot(train.revenue)
# plt.savefig("img/revenue.png")
# plt.close()
# sns.displot(np.log2(train.revenue+1))
# plt.savefig("img/revenue-log2.png")
# plt.close()
# 
# sns.displot(train_num.budget)
# plt.savefig("img/budget.png")
# plt.close()
# sns.displot(np.log2(train_num.budget+1))
# plt.savefig("img/budget-log2.png")
# plt.close()
# 
# sns.displot(train_num.runtime)
# plt.savefig("img/runtime.png")
# plt.close()
# 
# # pairplot
# sns.pairplot(train_num_no_id)
# plt.savefig("img/pairplot.png")
# plt.close()
# 
# # correlation matrix
# corr_mat = train_num_no_id.corr().round(2)
# sns.heatmap(corr_mat, annot=True, square=True, cmap="BuPu")
# plt.savefig("img/correlation_matrix.png")
# plt.close()
# 
# # release date plots
# 
# # process
# train['release_month'] = train.release_date.str.extract('(\S+)/\S+/\S+', expand=False).astype(np.int16)
# train['release_year'] = train.release_date.str.extract('\S+/\S+/(\S+)', expand=False).astype(np.int16)
# train['release_day'] = train.release_date.str.extract('\S+/(\S+)/\S+', expand=False).astype(np.int16)
# train.loc[(21 <= train.release_year) & (train.release_year <= 99), 'release_year'] += 1900
# train.loc[train.release_year < 21, 'release_year'] += 2000
# 
# # weekday
# train['release_date'] = pd.to_datetime(train.release_day.astype(str) + '-'
#                                        + train.release_month.astype(str) + '-'
#                                        + train.release_year.astype(str))
# train['release_weekday'] = train.release_date.dt.day_name().str.slice(0, 3)
# fig = sns.countplot(train.release_weekday, order=['Sun', 'Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat'])
# fig.set(ylabel="number of movies")
# plt.savefig("img/release_weekday.png")
# plt.close()
# 
# # day of month
# fig = sns.countplot(train.release_day)
# fig.set(ylabel="number of movies")
# # remove some ticks
# for index, label in enumerate(fig.xaxis.get_ticklabels()):
#     if index % 3 != 0:
#         label.set_visible(False)
# plt.savefig("img/release_day.png")
# plt.close()
# 
# # month
# fig = sns.countplot(train.release_month)
# fig.set(ylabel="number of movies")
# plt.savefig("img/release_month.png")
# plt.close()
# 
# # year
# fig = sns.countplot(train.release_year)
# fig.set(ylabel="number of movies")
# for index, label in enumerate(fig.xaxis.get_ticklabels()):
#     if index % 10 != 0:
#         label.set_visible(False)
# plt.savefig("img/release_year.png")
# plt.close()
# 
# # revenue by dates
# 
# # release_weekday
# rev_weekday = train.groupby('release_weekday')['revenue'].aggregate([np.mean])
# rev_weekday.reset_index(inplace=True)   # add index in place
# fig = sns.barplot(x='release_weekday', y='mean', data=rev_weekday, order=['Sun', 'Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat'])
# fig.set(ylabel="mean of revenue")
# plt.savefig("img/release_weekday_revenue.png")
# plt.close()
# # release_day
# rev_day = train.groupby('release_day')['revenue'].aggregate([np.mean])
# rev_day.reset_index(inplace=True)
# fig = sns.barplot(x='release_day', y='mean', data=rev_day)
# fig.set(ylabel='mean of revenue')
# for index, label in enumerate(fig.xaxis.get_ticklabels()):
#     if index % 3 != 0:
#         label.set_visible(False)
# plt.savefig("img/release_day_revenue.png")
# plt.close()
# # release_month
# rev_month = train.groupby('release_month')['revenue'].aggregate([np.mean])
# rev_month.reset_index(inplace=True)
# fig = sns.barplot(x='release_month', y='mean', data=rev_month)
# fig.set(ylabel='mean of revenue')
# plt.savefig("img/release_month_revenue.png")
# plt.close()
# # release_year
# rev_year = train.groupby('release_year')['revenue'].aggregate([np.mean])
# rev_year.reset_index(inplace=True)
# fig = sns.barplot(x='release_year', y='mean', data=rev_year)
# fig.set(ylabel='mean of revenue')
# for index, label in enumerate(fig.xaxis.get_ticklabels()):
#     if index % 10 != 0:
#         label.set_visible(False)
# plt.savefig("img/release_year_revenue.png")
# plt.close()

# genre
# convert categorical vars into dummy/indicator vars
train.loc[train.genres.isnull(), 'genres'] = "{}"
train['genres'] = train.genres.apply(lambda x: sorted([d['name'] for d in eval(x)])).apply(lambda x: ','.join(map(str, x)))
genres = train.genres.str.get_dummies(sep=',')
# plot movies by genre
movies_genre = pd.DataFrame(genres.sum(axis=0)).reset_index()   # number of movies with genre
movies_genre.columns = ['genres', 'movies'] # change labels
fig = sns.barplot(x='genres', y='movies', data=movies_genre)
fig.set(ylabel='number of movies')
fig.set_xticklabels(fig.get_xticklabels(), rotation=90)
plt.savefig("img/movies_by_genre.png", bbox_inches='tight')
plt.close()

# # missing values
# msno.matrix(train.sample(1000))
# plt.savefig(f"{imgdir}/missing_box_office_data1000.png", bbox_inches='tight')
# plt.close()

print()
