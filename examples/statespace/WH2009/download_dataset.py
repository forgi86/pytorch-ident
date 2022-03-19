import os
from google_drive_downloader import GoogleDriveDownloader as gdd


if __name__ == '__main__':

    DATA_FOLDER = 'data'

    # %% Make data folder if it does not exist
    if not os.path.exists(DATA_FOLDER):
        os.makedirs(DATA_FOLDER)

    # %% Download dataset from www.nonlinearbenchmark.com
    # https://drive.google.com/file/d/16ipySVfKfxkwqWmbO9Z19-VjDoC2S6hx/view?usp=sharing
    gdd.download_file_from_google_drive(file_id='16ipySVfKfxkwqWmbO9Z19-VjDoC2S6hx',
                                        dest_path='./data/data.zip',
                                        unzip=True)
