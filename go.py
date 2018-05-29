import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns

import numpy as np

df = pd.read_csv('GO_terms.csv')
df = df.fillna(0)
# print(df.shape)
# print(df.head())
# print(df.columns)

names = df['Unnamed: 0']

df = df.drop(['Unnamed: 0'], axis=1)

# print(df.as_matrix())
go = np.asarray(df.as_matrix(),dtype=int)

print(go.shape)

products = np.sum(go[:,np.newaxis,:] * go[:,:,np.newaxis], axis=0)

fig, ax = plt.subplots(figsize=(20,20)) 
sns.heatmap(products, annot=True,  linewidths=.5, ax=ax)
# fig.add_subplot(ax)
fig.savefig('go.png')

fig, ax = plt.subplots(figsize=(20,20)) 
ax.hist(products.flatten(),bins=35)
# fig.add_subplot(ax)
fig.savefig('go_hist.png')


print(df.columns[30], df.columns[9])

print(np.sum(df['fim1']*df['arp3']))