import os
import util.benchmark_url
import requests
import zipfile

if __name__ == '__main__':

    DATA_FOLDER = 'data'

    # In[Make data folder if it does not exist]
    if not os.path.exists(DATA_FOLDER):
        os.makedirs(DATA_FOLDER)

    # In[Download dataset from www.nonlinearbenchmark.com]
    r = requests.get(util.benchmark_url.WH2009, allow_redirects=True)

    # In[Save zipped file]
    zipped_dataset_path = os.path.join('data', 'data.zip')
    with open(zipped_dataset_path, 'wb') as f:
        f.write(r.content)

    # In[Extract zipped file]
    with zipfile.ZipFile(zipped_dataset_path, 'r') as zip_ref:
        zip_ref.extractall('data')
