# Import Required Packages
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCAcd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# Loading and examining the dataset
penguins_df = pd.read_csv("C:\\Users\\USERgit \\PycharmProjects\\pythonProject\\Datasets\\penguins.csv")
print(penguins_df.head())
print(penguins_df.columns)

print(penguins_df.shape,'original Dfshape')
penguins_df.drop(['species','island','year'],axis=1,inplace=True)
penguins_df.rename(columns={'bill_length_mm':'culmen_length_mm',
                            'bill_depth_mm':'culmen_depth_mm'}, inplace=True)
print(penguins_df.columns)
print(penguins_df.isna().sum())
penguins_clean =penguins_df.dropna()
print(penguins_clean.isna().sum())

penguins_clean['sex'].value_counts()#op has 3 unique values of sex
#incase this was an error
'''penguins_clean=penguins_clean[penguins_clean['sex'] != '.']
print(penguins_clean['sex'].value_counts())'''


#checking outliers
plt.boxplot(penguins_clean['culmen_length_mm'])#no outliers
plt.xlabel('culmen_length')
plt.clf()
plt.boxplot(penguins_clean['culmen_depth_mm'])#no outliers
plt.xlabel('culmen_depth')
plt.clf()
plt.boxplot(penguins_clean['flipper_length_mm'])#there are  outliers here
#plt.boxplot(penguins_clean['body_mass_g'])#no outliers
#plt.xlabel('body_mass')
#plt.clf()
#plt.clf()
#calculating percentile threshold for flipper_length_mm
'''seventy_fifth_fl=penguins_clean['flipper_length_mm'].quantile(0.75)
print('75th', seventy_fifth_fl)
twenty_fifth_fl=penguins_clean['flipper_length_mm'].quantile(0.25)
print('25th',twenty_fifth_fl)
IQR_fl=seventy_fifth_fl-twenty_fifth_fl
#threshold calculation
upper_fl=seventy_fifth_fl + (1.5 * IQR_fl)
lower_fl=twenty_fifth_fl - (1.5 * IQR_fl)
#finding values outside thresholds
fl_outliers= penguins_clean[(penguins_clean['flipper_length_mm'] < lower_fl) |(penguins_clean['flipper_length_mm'] > upper_fl)]
print(len(fl_outliers),'len') # there are 2 outliers
#dropping the rows with the outliers
print(penguins_clean.shape)
penguins_clean =penguins_clean[(penguins_clean['flipper_length_mm'] > lower_fl) & (penguins_clean['flipper_length_mm'] < upper_fl)] #this threshold is too tight
print(penguins_clean.shape)'''
# the outliers are > 4000 and below 0
print(penguins_clean[penguins_clean['flipper_length_mm'] > 4000])#row 9
print(penguins_clean[penguins_clean['flipper_length_mm'] < 0])#row 14
#we drop the two
#penguins_clean=penguins_clean.drop([9,14])
print('....................................................')

#confirming outliers are removed
print(penguins_clean[penguins_clean['flipper_length_mm'] > 4000])#row 9
print(penguins_clean[penguins_clean['flipper_length_mm'] < 0])#row 14
plt.boxplot(penguins_clean['flipper_length_mm'])
plt.xlabel('flipper_length')

#dummy variables and removing original categorical features
penguin_dummies=pd.get_dummies(penguins_clean['sex'], prefix='sex')
penguin_dummies=pd.concat([penguins_clean,penguin_dummies],axis=1)
penguin_dummies.drop("sex",axis=1,inplace=True)
#penguin_dummies.drop("sex_.",axis=1,inplace=True)
print(penguin_dummies.columns)
#instantiate standard scaling
scaler = StandardScaler()
print(penguin_dummies.describe())
dummies_scaled=scaler.fit_transform(penguin_dummies)
penguins_preprocessed = pd.DataFrame(data=dummies_scaled, columns=penguin_dummies.columns)
#print(penguins_preprocessed.head())
#print(penguins_preprocessed.columns)

#performing PCA
features = ['culmen_length_mm', 'culmen_depth_mm', 'flipper_length_mm',
       'body_mass_g','sex_FEMALE','sex_MALE',]
pca=PCA()
pca.fit(penguins_preprocessed)
#plot the ariance of PCA features
ex_variance_ratio=pca.explained_variance_ratio_
plt.bar(features,ex_variance_ratio) # we see 2 components above 0.1
plt.xticks(features)
plt.xlabel('pca features')
plt.ylabel('variance')
plt.title("Variance vs PCA features")
plt.clf()
#componenets with varaince ratio above 10%
n_components_boolean = ex_variance_ratio > 0.1
n_components=sum(n_components_boolean)
print(n_components)

#execute PCA using n_components
penguins_PCA=PCA(n_components=n_components)
penguins_PCA.fit(penguins_preprocessed)
penguins_PCA = penguins_PCA.transform(penguins_preprocessed)
print(penguins_PCA.shape)

#employing k-means
inertia=[]
#range_k_values = range(1,11)
for k in list(range(1,11)):
    kmeans=KMeans(n_clusters=k,random_state=42,n_init=10)
    kmeans.fit(penguins_PCA)
    inertia.append(kmeans.inertia_)
#plot the elbow
plt.plot(list(range(1,11)), inertia, marker='x')
plt.ylabel('inertias')
plt.xlabel('no of clusters k')
plt.title('Inertia vs number of clusters k')
#plt.xlim(1,11)
plt.clf()

'''import seaborn as sns
sns.lineplot(x=list(range(1,11)),y=inertia)'''

#plt.grid(True)
#plt.show()
#elbow starts at 4
n_clusters=4

#creating ew k cluster model
kmeans=KMeans(n_clusters=n_clusters,n_init=10, random_state=42)
kmeans.fit(penguins_PCA)
xs=penguins_PCA[:,0]
ys=penguins_PCA[:,1]
plt.scatter(xs,ys,c=kmeans.labels_)
plt.xlabel("1st pca component")
plt.ylabel("2nd pca component")
plt.title("2nd pca component vs 1st pca component")
plt.show()
#plt.legend()

#adding label column to penguins clean
penguins_clean.loc[:, 'label'] = kmeans.labels_
print(penguins_clean.head())
print(penguins_clean.columns)

stat_penguins=penguins_clean.groupby("label")[['culmen_length_mm', 'culmen_depth_mm', 'flipper_length_mm','body_mass_g','label']].mean()
print(stat_penguins)