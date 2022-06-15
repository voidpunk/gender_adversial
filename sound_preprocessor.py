import os
import kaggle


kaggle.api.authenticate()
# dataset = 'ogechukwu/voice'
dataset = 'tommyngx/fluent-speech-corpus'
print('Downloading and extracting the dataset... ~ 3 min on my system')
kaggle.api.dataset_download_files(dataset, path='./', unzip=True)
os.rename('fluent_speech_commands_dataset', 'fluent_speech')

