# libaries
import pandas as pd

# import data
df = pd.read_csv(
    'data/BeerDataScienceProject.tar.bz2',
    compression='bz2',
    parse_dates=['review_time'],
    infer_datetime_format=True)
df.head()

# clean review_time
df['review_time'] = pd.to_datetime(df['review_time'], unit='s')
df.head()

##### Question 1 #####
# copy df
q1 = df.copy()
# group by brewer, calculate mean ABV
q1 = q1.groupby('beer_brewerId')['beer_ABV'].mean().reset_index()
# rename column
q1.rename(columns={'beer_ABV':'mean_ABV'}, inplace=True)
# get 3 largest
q1.nlargest(columns='mean_ABV', n=3)

##### Question 2 #####
# copy df
q2 = df.copy()
# create year column
q2['year'] = q2['review_time'].dt.year
# group by year, calculate avg review_overall
q2 = q2.groupby('year')['review_overall'].mean().to_frame('mean_overall').reset_index()
# get largest
q2.nlargest(columns='mean_overall', n=1)

##### Question 3 #####
# copy df
q3 = df.copy()

# initialize random forest
from sklearn.ensemble import RandomForestRegressor
model = RandomForestRegressor()

# features
X = q3[['review_taste', 'review_aroma', 'review_appearance', 'review_palette']]
# target
y = q3['review_overall']
# fit model
model.fit(X,y)
# feature and their importances
model_coef_df = pd.DataFrame({
    'feature':model.feature_names_in_
    ,'importances':model.feature_importances_
    })
model_coef_df
model_coef_df.nlargest(columns='importances', n=1)

##### Question 4 #####
q4 = df.copy()
k = q4['beer_style'].value_counts()
q4.groupby('beer_name')['review_taste', 'review_aroma']

##### Question 5 #####
# import libraries
from nltk.sentiment.vader import SentimentIntensityAnalyzer
sia = SentimentIntensityAnalyzer()
# copy df
q5 = df.copy()
# drop rows with missing review_text
q5.dropna(subset=['review_text'], inplace=True)

# get polarity scores
q5['polarity_compound'] = [sia.polarity_scores(text)['compound'] for text in q5['review_text']]

# top 10 favorites by written reviews based on polarity scores
q5.groupby('beer_style')['polarity_compound'].mean().to_frame('mean_polarity_compound').reset_index().nlargest(columns='mean_polarity_compound', n=10)
written_favs = q5.groupby('beer_style')['polarity_compound'].mean().to_frame('mean_polarity_compound').reset_index().nlargest(columns='mean_polarity_compound', n=10)['beer_style']

# top 10 favorite by overall review score
q5.groupby('beer_style')['review_overall'].mean().to_frame('mean_review_overall').reset_index().nlargest(columns='mean_review_overall', n=10)
review_score_favs = q5.groupby('beer_style')['review_overall'].mean().to_frame('mean_review_overall').reset_index().nlargest(columns='mean_review_overall', n=10)['beer_style']

# Beer Styles found in both
list(set(written_favs).intersection(set(review_score_favs)))
q5[['polarity_compound', 'review_overall']].corr()