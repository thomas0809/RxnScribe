"""
this is used to down journals in PDF format using dois
"""

from helium import *
import time
import glob
import os
import zipfile


# this will prompt 6 windows for each query and then manually download
# queries: 'battery', 'catalysis', 'cells', 'city', 'solar', 'dssc', 'light', 'nano', 'neel', 'super alloy'
def download_pdf(query):
    for year in range(2016, 2022):
        path = 'https://www.sciencedirect.com/search?qs={}&show=100&years={}&lastSelectedFacet=accessTypes&articleTypes=FLA&accessTypes=openaccess'.format(
            query,
            year)
        start_chrome(path)
        time.sleep(5)


def unzip_single(name, path):
    for zip in path:
        file = zipfile.ZipFile(zip, "r")
        file.extractall('/Users/miao/Downloads/' + name)


def unzip_folders():
    for i in ['battery', 'catalysis', 'cells', 'city', 'solar', 'dssc', 'light', 'nano', 'neel', 'super alloy']:
        folder = r'/Users/miao/Downloads/{}/*.zip'.format(i)
        path = (glob.glob(folder))
        unzip_single(i, path)


def create_folder():
    for i in ['battery', 'catalysis', 'cells', 'city', 'solar', 'dssc', 'light', 'nano', 'neel', 'super alloy']:
        os.mkdir(r'/Users/miao/Downloads/{}'.format(i))

# create_folder()
# download_pdf('catalysis')
# unzip_folders()
