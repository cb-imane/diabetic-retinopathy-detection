from ucimlrepo import fetch_ucirepo
import pandas as pd



diabetic_retinopathy_debrecen = fetch_ucirepo(id=329)



metadata = diabetic_retinopathy_debrecen.metadata
print(metadata)
data_url = metadata['data_url']
#https://archive.ics.uci.edu/static/public/329/data.csv"

df = pd.read_csv(data_url) 
df.to_csv('../data/data_retino.csv')
