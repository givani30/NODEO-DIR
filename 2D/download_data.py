import os
from nilearn import datasets
#Download the OASIS-1 dataset and save it in the Data folder

def download_dataset(data_dir,n_subjects=None):
    '''Download the OASIS-1 dataset and save it in the Data folder	
    Parameters
    ----------
    data_dir: str
        Path to the Data folder
    n_subjects: int, optional
        Number of subjects to download. If None, download all subjects
    '''
    print("Downloading dataset...")
    data=datasets.fetch_oasis_vbm(data_dir=data_dir,n_subjects=None,legacy_format=False,dartel_version=False)
    print("Dataset downloaded.")
if __name__ == '__main__':
    n_subjects = None # 1 for testing, None for all
    data_dir = os.path.join(os.getcwd(), "Data") # Path to the Data folder
    download_dataset(data_dir,n_subjects=n_subjects)