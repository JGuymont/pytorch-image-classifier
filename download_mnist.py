#!/usr/bin/env python3
"""
Created on Mon Jan 18 2018

@author: J. Guymont
"""

TRAIN_FILE_ID = '1XdxgtKIkc-C-r3E3ykZONTrZUHdEFDy7'
TRAIN_PATH = './data/jpg/trainSet.tar.gz'
TEST_FILE_ID = '1Z-tYr4yrl3wibWSlJyOV5z2cJJ3fo9ur'
TEST_PATH = './data/jpg/testSet.tar.gz'

DEST_PATH = './data/jpg/'

import tarfile
from google_drive_downloader import GoogleDriveDownloader as gdd

def untar(file_path, dest_path):
    """extract gzip folder"""
    tar = tarfile.open(file_path)
    tar.extractall(dest_path)
    tar.close()
    return None

if __name__ == "__main__":

    gdd.download_file_from_google_drive(file_id=TRAIN_FILE_ID,
                                        dest_path=TRAIN_PATH,
                                        unzip=False)
    gdd.download_file_from_google_drive(file_id=TEST_FILE_ID,
                                        dest_path=TEST_PATH,
                                        unzip=False)
    untar(TRAIN_PATH, DEST_PATH)
    untar(TEST_PATH, DEST_PATH)
