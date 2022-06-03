import os
import shutil
import tarfile
import requests
import threading
import numpy as np
import pandas as pd
from tqdm import tqdm
import multiprocessing
from itertools import count
from scipy.io import loadmat



### DRY

exs = os.path.exists

# negative checks for archives, folders, csv files
checks = {}

# update function
def check() -> None:
    global checks
    checks = {
    'it': not exs('imdb_crop.tar'),
    'wt': not exs('wiki_crop.tar'),
    'id': not exs('imdb_crop'),
    'wd': not exs('wiki_crop'),
    'ic': not exs('imdb.csv'),
    'wc': not exs('wiki.csv'),
}



### downloading the archives if necessary

check()

# download function
def download(url) -> None:
    response = requests.get(url)
    filename = url.split('/')[-1]
    with open(filename, 'wb') as tar:
        tar.write(response.content)


# dataset urls
imdb_url = 'https://data.vision.ee.ethz.ch/cvl/rrothe/imdb-wiki/static/imdb_crop.tar'
wiki_url = 'https://data.vision.ee.ethz.ch/cvl/rrothe/imdb-wiki/static/wiki_crop.tar'

# concurrent download
if checks['it'] and checks['wt'] and checks['id'] and checks['wd']:
    print('Downloading archives...')
    t1 = threading.Thread(target=download, args=(imdb_url,))
    t2 = threading.Thread(target=download, args=(wiki_url,))
    t1.start(), t2.start()
    t1.join(), t2.join()
    check()

# selective download
if checks['it'] and checks['id']:
    print('Downloading archive imdb...')
    download(imdb_url)

if checks['wt'] and checks['wd']:
    print('Downloading archive wiki...')
    download(wiki_url)




### extracting the archives if necessary

check()

# extract function
def untar(file) -> None:
    with tarfile.open(file) as tar:
        tar.extractall('./')

# parallelized extraction
if checks['id'] and checks['wd']:
    print('Extracting archives...')
    p1 = multiprocessing.Process(target=untar, args=('imdb_crop.tar',))
    p2 = multiprocessing.Process(target=untar, args=('wiki_crop.tar',))
    p1.start(), p2.start()
    p1.join(), p2.join()
    check()

# selective extraction
if checks['id']:
    print('Extracting archive imbd...')
    untar('imdb_crop.tar')

if checks['wd']:
    print('Extracting archive wiki...')
    untar('wiki_crop.tar')




### processing metadata if necessary

check()
if checks['ic'] and checks['wc']:
    
    # loading the metadata
    imdb = loadmat('imdb_crop/imdb.mat')['imdb']
    wiki = loadmat('wiki_crop/wiki.mat')['wiki']

    # extracting the metadata (female=0 & male=1)
    imdb_path =  ['imdb_crop/' + path[0] for path in imdb[0][0][2][0]]
    imdb_gender = imdb[0][0][3][0]
    imdb_face_score1 = imdb[0][0][6][0]
    imdb_face_score2 = imdb[0][0][7][0]

    wiki_path = ['wiki_crop/' + path[0] for path in wiki[0][0][2][0]]
    wiki_gender = wiki[0][0][3][0]
    wiki_face_score1 = wiki[0][0][6][0]
    wiki_face_score2 = wiki[0][0][7][0]

    # reorganizing the metadata
    cols = ['gender', 'path', 'face_score1', 'face_score2']
    imdb_df = pd.DataFrame(
        np.vstack((
            imdb_gender, imdb_path, imdb_face_score1, imdb_face_score2
        )).T
    )
    wiki_df = pd.DataFrame(
        np.vstack((
            wiki_gender, wiki_path, wiki_face_score1, wiki_face_score2
        )).T
    )

    imdb_df.columns = cols
    wiki_df.columns = cols

    # cleaning the metadata
    imdb_df = imdb_df[
            (imdb_df['face_score1'] != '-inf') & 
            (imdb_df['face_score2'] == 'nan')
        ].drop(
            ['face_score1', 'face_score2'],
            axis=1
        ).set_index(
            'path'
    )
    wiki_df = wiki_df[
            (wiki_df['face_score1'] != '-inf') & 
            (wiki_df['face_score2'] == 'nan')
        ].drop(
            ['face_score1', 'face_score2'],
            axis=1
        ).set_index(
            'path'
    )

    # backup the metadata
    imdb_df.to_csv('imdb.csv', index=False)
    wiki_df.to_csv('wiki.csv', index=False)

else:

    # loading the metadata
    imdb_df = pd.read_csv('imdb.csv', index_col=0)
    wiki_df = pd.read_csv('wiki.csv', index_col=0)



### cleaning and reorganizing the data

# set up
if not os.path.exists('imdb_wiki_clean'):
    os.mkdir('imdb_wiki_clean')
iterator = (count())

# cleaning the imdb data
print('Cleaning imdb data...')
for dir in tqdm(os.listdir('imdb_crop')):
    try:
        for file in os.listdir(os.path.join('imdb_crop', dir)):
            if file.endswith('.jpg'):
                path = os.path.join('imdb_crop', dir, file)
                try:
                    gender = str(imdb_df.loc[path][0])[0] + '-' + str(next(iterator)) + '.jpg'
                    shutil.move(path, os.path.join('imdb_wiki_clean', gender))
                except KeyError:
                    os.remove(path)
    except NotADirectoryError:
        pass # exclude the .mat file

# cleaning the wiki data
print('Cleaning wiki data...')
for dir in tqdm(os.listdir('wiki_crop')):
    try:
        for file in os.listdir(os.path.join('wiki_crop', dir)):
            if file.endswith('.jpg'):
                path = os.path.join('wiki_crop', dir, file)
                try:
                    gender = str(wiki_df.loc[path][0])[0] + '-' + str(next(iterator)) + '.jpg'
                    shutil.move(path, os.path.join('imdb_wiki_clean', gender))
                except KeyError:
                    os.remove(path)
    except NotADirectoryError:
        pass # exclude the .mat file

print(f'{len(os.listdir("imdb_wiki_clean")):,} images are ready for training! :D')


### cleaning up

check()
print('Deleting cached files...')
# if not checks['it']: os.remove('imdb_crop.tar')
# if not checks['wt']: os.remove('wiki_crop.tar')
if not checks['id']: shutil.rmtree('imdb_crop')
if not checks['wd']: shutil.rmtree('wiki_crop')
if not checks['ic']: os.remove('imdb.csv')
if not checks['wc']: os.remove('wiki.csv')
