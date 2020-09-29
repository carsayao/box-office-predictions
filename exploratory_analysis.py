# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# 
# Exploratory data analysis
#
# Sources:
#   https://github.com/DachshundSovereign/Kaggle-TMDB-Box-Office/
#   https://www.kaggle.com/artgor/eda-feature-engineering-and-model-interpretation
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
train.to_pickle("data/train.pickle")

# import pdb; pdb.set_trace()

# number of zeros in budget column (812)
train_num_zero_budget = (train["budget"] == 0).sum(axis=0)
print("\nNumber of zeros in budget column: ", train_num_zero_budget)

# look at general info of our train data
print("\nData types")
train.info()

# separate categorical and numerical
train_cat_lab = train.columns[train.dtypes == 'object']
train_cat = train.select_dtypes(['object'])
train_num_lab = train.columns[train.dtypes != 'object']
train_num = train.select_dtypes(['number']).fillna(0)
# drop id for pair plot
train_num_no_id = train_num.drop(['id'], axis=1)
print("\nNumerical feature labels ", len(train_num_lab))
print("Numerical features ", train_num.shape)
print("\nCategorical feature labels ", len(train_cat_lab))
print("Categorical features", train_cat.shape)
print()

# get some plots

# univariate distributions
sns.displot(train.revenue)
plt.savefig("img/revenue.png")
plt.close()
sns.displot(np.log2(train.revenue+1))
plt.savefig("img/revenue-log2.png")
plt.close()

sns.displot(train_num.budget)
plt.savefig("img/budget.png")
plt.close()
sns.displot(np.log2(train_num.budget+1))
plt.savefig("img/budget-log2.png")
plt.close()

sns.displot(train_num.runtime)
plt.savefig("img/runtime.png")
plt.close()

# pairplot
sns.pairplot(train_num_no_id)
plt.savefig("img/pairplot.png")
plt.close()

# correlation matrix
corr_mat = train_num_no_id.corr().round(2)
sns.heatmap(corr_mat, annot=True, square=True, cmap="BuPu")
plt.savefig("img/correlation_matrix.png")
plt.close()

# release date plots

# process
train['release_month'] = (train.release_date.str
                          .extract('(\S+)/\S+/\S+', expand=False)
                          .astype(np.int16))
train['release_year']  = (train.release_date.str
                          .extract('\S+/\S+/(\S+)', expand=False)
                          .astype(np.int16))
train['release_day']   = (train.release_date.str
                          .extract('\S+/(\S+)/\S+', expand=False)
                          .astype(np.int16))
train.loc[(21 <= train.release_year) & (train.release_year <= 99),
           'release_year'] += 1900
train.loc[train.release_year < 21, 'release_year'] += 2000

# weekday
train['release_date'] = pd.to_datetime(train.release_day.astype(str) + '-'
                                       + train.release_month.astype(str) + '-'
                                       + train.release_year.astype(str))
train['release_weekday'] = train.release_date.dt.day_name().str.slice(0, 3)
fig = sns.countplot(train.release_weekday,
                    order=['Sun', 'Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat'])
fig.set(ylabel="number of movies")
plt.savefig("img/release_weekday.png")
plt.close()

# day of month
fig = sns.countplot(train.release_day)
fig.set(ylabel="number of movies")
# remove some ticks
for index, label in enumerate(fig.xaxis.get_ticklabels()):
    if index % 3 != 0:
        label.set_visible(False)
plt.savefig("img/release_day.png")
plt.close()

# month
fig = sns.countplot(train.release_month)
fig.set(ylabel="number of movies")
plt.savefig("img/release_month.png")
plt.close()

# year
fig = sns.countplot(train.release_year)
fig.set(ylabel="number of movies")
for index, label in enumerate(fig.xaxis.get_ticklabels()):
    if index % 10 != 0:
        label.set_visible(False)
plt.savefig("img/release_year.png")
plt.close()

# revenue by dates

# release_weekday
rev_weekday = train.groupby('release_weekday')['revenue'].aggregate([np.mean])
rev_weekday.reset_index(inplace=True)   # add index in place
fig = sns.barplot(x='release_weekday', y='mean', data=rev_weekday,
                  order=['Sun', 'Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat'])
fig.set(ylabel="mean of revenue")
plt.savefig("img/release_weekday_revenue.png")
plt.close()
# release_day
rev_day = train.groupby('release_day')['revenue'].aggregate([np.mean])
rev_day.reset_index(inplace=True)
fig = sns.barplot(x='release_day', y='mean', data=rev_day)
fig.set(ylabel='mean of revenue')
for index, label in enumerate(fig.xaxis.get_ticklabels()):
    if index % 3 != 0:
        label.set_visible(False)
plt.savefig("img/release_day_revenue.png")
plt.close()
# release_month
rev_month = train.groupby('release_month')['revenue'].aggregate([np.mean])
rev_month.reset_index(inplace=True)
fig = sns.barplot(x='release_month', y='mean', data=rev_month)
fig.set(ylabel='mean of revenue')
plt.savefig("img/release_month_revenue.png")
plt.close()
# release_year
rev_year = train.groupby('release_year')['revenue'].aggregate([np.mean])
rev_year.reset_index(inplace=True)
fig = sns.barplot(x='release_year', y='mean', data=rev_year)
fig.set(ylabel='mean of revenue')
for index, label in enumerate(fig.xaxis.get_ticklabels()):
    if index % 10 != 0:
        label.set_visible(False)
plt.savefig("img/release_year_revenue.png")
plt.close()


# genre

# convert categorical vars into dummy/indicator vars
print("\nNumber of null genres: ", len(train.loc[train.genres.isnull(), 'genres']))
train.loc[train.genres.isnull(), 'genres'] = "{}"   # turn NaN genres into "{}"
# turn genres into comma separated list
train['genres'] = (train.genres.apply(lambda x: sorted([d['name'] for d in eval(x)]))
                                       .apply(lambda x: ','.join(map(str, x))))
genres = train.genres.str.get_dummies(sep=',')  # encode

# plot movies by genre
movies_genre = pd.DataFrame(genres.sum(axis=0)).reset_index()   # num movies in genre
movies_genre.columns = ['genres', 'movies'] # change label names
fig = sns.barplot(x='genres', y='movies', data=movies_genre)
fig.set(ylabel='number of movies')
fig.set_xticklabels(fig.get_xticklabels(), rotation=90)
plt.savefig("img/movies_by_genre.png", bbox_inches='tight')
plt.close()

# median revenue for each genre
rev_genre = list()
for col in genres.columns:
    rev_genre.append([col, train.loc[genres[col] == 1, 'revenue'].median()])
rev_genre = pd.DataFrame(rev_genre, columns=['genres', 'revenue'])
fig = sns.barplot(x='genres', y='revenue', data = rev_genre)
fig.set_xticklabels(fig.get_xticklabels(), rotation=90)
plt.savefig("img/revenue_by_genre.png", bbox_inches='tight')
plt.close()


# production companies

# number of movies and median revenue for production company
print("Number of null production_companies: ",
      len(train.loc[train.production_companies.isnull(), 'production_companies']))
train.loc[train.production_companies.isnull(), 'production_companies'] = "{}"
train['production_companies'] = (train.production_companies
                                  .apply(lambda x: sorted([d['name'] for d in eval(x)]))
                                  .apply(lambda x: ','.join(map(str, x))))
companies = train.production_companies.str.get_dummies(sep=',') # encode

# number of movies for each production company
movies_companies = pd.DataFrame(companies.sum(axis=0)).reset_index()
movies_companies.columns = ['company', 'movies']
top_25_companies = (movies_companies
                    .sort_values(by='movies', ascending=False)
                    .reset_index().loc[0:25])
fig = sns.barplot(x='company', y='movies', data=top_25_companies)
fig.set(ylabel='number of movies', xlabel='top 25 companies')
fig.set_xticklabels(fig.get_xticklabels(), rotation=90)
#for index, label in enumerate(fig.xaxis.get_ticklabels()):
#    if index % 4 != 0:
#        label.set_visible(False)
plt.savefig("img/top25_movies_per_production_company.png", bbox_inches='tight')
plt.close()
 
# median revenue for each company
rev_companies = list()
for col in companies.columns:
    rev_companies.append([col, train.loc[companies[col]==1, 'revenue'].median()])
rev_companies = pd.DataFrame(rev_companies, columns=['company', 'revenue'])
top_25_companies = (rev_companies
                    .sort_values(by='revenue', ascending=False)
                    .reset_index()
                    .loc[0:25])
fig = sns.barplot(x='company', y='revenue', data=top_25_companies)
fig.set(xlabel='top 25 companies')
fig.set_xticklabels(fig.get_xticklabels(), rotation=90)
plt.savefig("img/top25_revenues_per_production_company.png", bbox_inches='tight')


# production countries

# number of movies and median revenue for production country
train.loc[train.production_countries.isnull(), 'production_countries'] = "{}"
train['production_countries'] = (train.production_countries
                                 .apply(lambda x: sorted([d['name'] for d in eval(x)]))
                                 .apply(lambda x: ','.join(map(str, x))))
countries = train.production_countries.str.get_dummies(sep=',')

# number movies for each country
movies_countries = pd.DataFrame(countries.sum(axis=0)).reset_index()
movies_countries.columns = ['countries', 'movies']
top_25_countries = (movies_countries
                    .sort_values(by='movies', ascending=False)
                    .reset_index().loc[0:25])
fig = sns.barplot(x='countries', y='movies', data=top_25_countries)
fig.set(ylabel='number of movies', xlabel='top 25 countries')
fig.set_xticklabels(fig.get_xticklabels(), rotation=90)
plt.savefig("img/top25_movies_per_country.png", bbox_inches='tight')
plt.close()

# median revenue for each country
rev_countries = list()
# for each country, append the median revenue to the list
for col in countries.columns:   # get medians of revenue by country
    rev_countries.append([col, train.loc[countries[col]==1, 'revenue'].median()])
# convert list to dataframe
rev_countries = pd.DataFrame(rev_countries, columns=['country', 'revenue'])
top_25_countries = (rev_countries
                    .sort_values(by='revenue', ascending=False)
                    .reset_index().loc[0:25])
fig = sns.barplot(x='country', y='revenue', data=top_25_countries)
fig.set_xticklabels(fig.get_xticklabels(), rotation=90)
fig.set(xlabel='top 25 countries')
plt.savefig("img/top25_revenue_per_country.png", bbox_inches='tight')
plt.close()

# number of movies and median revenue for spoken languages
train.loc[train.spoken_languages.isnull(), 'spoken_languages'] = "{}"
train['spoken_languages'] = (train.spoken_languages
                             .apply(lambda x: sorted([d['name'] for d in eval(x)]))
                             .apply(lambda x: ','.join(map(str, x))))
languages = train.spoken_languages.str.get_dummies(sep=',')

# number of movies for each spoken language
movies_lang = pd.DataFrame(languages.sum(axis=0)).reset_index()
movies_lang.columns = ['language', 'movies']
top_25_languages = (movies_lang
                    .sort_values(by='movies', ascending=False)
                    .reset_index().loc[0:25])
fig = sns.barplot(x='language', y='movies', data=top_25_languages)
fig.set(ylabel='number of movies', xlabel='top 25 languages')
fig.set_xticklabels(fig.get_xticklabels(), rotation=90)
plt.savefig("img/top25_movies_per_language.png", bbox_inches='tight')
plt.close()


# write pickles
train.to_pickle("data/train_processed.pickle")


# feature importance



# missing values
msno.matrix(train.sample(500))
plt.savefig(f"{imgdir}/missing_box_office_data500.png", bbox_inches='tight')
plt.close()

print()
