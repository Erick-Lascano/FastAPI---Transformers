import kagglehub

def download_dataset(dataset_name: str) -> str:
    """
    Descarga dataset desde Kaggle y retorna el path local.
    """
    path = kagglehub.dataset_download(dataset_name)
    return path