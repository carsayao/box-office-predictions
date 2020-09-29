import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

train = pd.read_pickle("data/train.pickle")
train_processed = pd.read_pickle("data/train_processed.pickle")

feature_importance = pd.DataFrame({'feature': train_processed.columns.drop(['revenue']),
                                   'importance': train_processed.feature_importances_})

fig = sns.barplot(x='feature', y='importance', data=feature_importance.sort_values(by='importance', ascending=False))
_ = fig.set_xticklabels(fig.get_xticklabels(), rotation=90)
for index, label in enumerate(fig.xaxis.get_ticklabels()):
    if index % 4 != 0:
        label.set_visible(False)
plt.savefig("img/feature_importance.png")
plt.close()

print("This is a break statement")
