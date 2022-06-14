import os
import kaggle


if not os.path.exists('bvc'):
    os.mkdir('bvc')
kaggle.api.authenticate()
dataset = 'ogechukwu/voice'
print('Downloading and extracting the dataset... ~ 2 min on my system')
kaggle.api.dataset_download_files(dataset, path='./bvc', unzip=True)

