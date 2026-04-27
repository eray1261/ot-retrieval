"""
Code adapted from libero repository at https://github.com/Lifelong-Robot-Learning/LIBERO/blob/master/
"""
import os
from tqdm.auto import tqdm
from urllib import request
import time
import zipfile

LIBERO_DOWNLOAD_LINK = "https://utexas.box.com/shared/static/cv73j8zschq8auh9npzt876fdc1akvmk.zip"

class DownloadProgressBar(tqdm):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def update_to(self, b=1, bsize=1, tsize=None):
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)


def url_is_alive(url):
    """
    From https://gist.github.com/dehowell/884204.
    """
    test_request = request.Request(url)

    try:
        request.urlopen(test_request)
        return True
    except request.HTTPError:
        return False

def main():
    # NOTE Override this if you want to download to a different directory
    download_dir = os.path.join(os.path.dirname(__file__), 'LIBERO')

    os.makedirs(download_dir, exist_ok=True)
    print(f"Datasets downloaded to {download_dir}")
    print(f"Downloading LIBERO 100 datasets")
    
    assert url_is_alive(LIBERO_DOWNLOAD_LINK), "@download_url got unreachable url: {}".format(LIBERO_DOWNLOAD_LINK)
    time.sleep(0.5)

    # infer filename from url link
    fname = LIBERO_DOWNLOAD_LINK.split("/")[-1]
    file_to_write = os.path.join(download_dir, fname)

    with DownloadProgressBar(
        unit="B", unit_scale=True, miniters=1, desc=fname
    ) as t:
        request.urlretrieve(
            LIBERO_DOWNLOAD_LINK, filename=file_to_write, reporthook=t.update_to
        )

    with zipfile.ZipFile(file_to_write, "r") as archive:
        archive.extractall(path=download_dir)
    if os.path.isfile(file_to_write):
        os.remove(file_to_write)
  


if __name__ == "__main__":
    main()