import gdown
import zipfile
import os
from pathlib import Path

class DataDownloader:
    def __init__(self, url, download_path, extract_to):
        self.url = url
        self.download_path = Path(download_path)
        self.extract_to = Path(extract_to)

    def download_file_from_google_drive(self):
        """Download a file from Google Drive using gdown"""
        print(f"Downloading {self.url} to {self.download_path}...")
        gdown.download(self.url, str(self.download_path), quiet=False)

    def unzip_file(self):
        """Unzip a file to the destination directory"""
        print(f"Unzipping {self.download_path} to {self.extract_to}...")
        with zipfile.ZipFile(self.download_path, 'r') as zip_ref:
            zip_ref.extractall(self.extract_to)
        print("Unzipping completed.")

    def cleanup(self):
        """Remove the zip file after extraction"""
        if self.download_path.exists():
            os.remove(self.download_path)
            print(f"Removed zip file: {self.download_path}")

    def download_and_extract(self):
        """Download the file and then extract it"""
        self.download_file_from_google_drive()
        self.unzip_file()
        self.cleanup()

if __name__ == "__main__":
    # The direct download link
    url = "https://drive.google.com/uc?id=1tQZFfpEcazxvrUuoqnbYfKiHY51X0ody"
    downloader = DataDownloader(url, "Dataset.zip", "Dataset")
    downloader.download_and_extract()

