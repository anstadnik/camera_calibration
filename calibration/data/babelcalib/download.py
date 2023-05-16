from glob import glob
import shutil
from plotly.express import data
import requests
from bs4 import BeautifulSoup, Tag
from tqdm.auto import tqdm
import zipfile
import os


files = {
    "OV.zip": "https://drive.google.com/uc?export=download&id=1KqJqyqmRsBVr-9PnXCV8CuFbasOUobaF",
    "Kalibr.zip": "https://drive.google.com/uc?export=download&id=1KuR_JhC7cB0Ybzxq6Hr6dtn1j3IMjifn",
    "UZH.zip": "https://drive.google.com/uc?export=download&id=1KuR_JhC7cB0Ybzxq6Hr6dtn1j3IMjifn",
    "OCamCalib.zip": "https://drive.google.com/uc?export=download&id=1tbPfs_c5-scSbqWXoyuLfaadSj1ZK65T",
}


def download_file(url, destination):
    response = requests.get(url)
    response.raise_for_status()

    soup = BeautifulSoup(response.text, "html.parser")
    form_tag = soup.find("form", {"id": "download-form"})
    assert isinstance(form_tag, Tag)
    download_link = form_tag["action"]
    assert isinstance(download_link, str)

    with (
        requests.get(download_link, stream=True) as download_response,
        open(destination, "wb") as file,
    ):
        download_response.raise_for_status()

        total_size = int(download_response.headers.get("content-length", 0))
        block_size = 8192  # 8 KB
        with tqdm(total=total_size, unit="B", unit_scale=True, leave=False) as pbar:
            with open(destination, "wb") as file:
                for chunk in download_response.iter_content(block_size):
                    if chunk:
                        file.write(chunk)
                        pbar.update(len(chunk))


def assure_babelcalib_downloaded(data_dir: str):
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)

    if not all(
        os.path.isdir(os.path.join(data_dir, ds.removesuffix(".zip"))) for ds in files
    ):
        for file_name, file_url in files.items():
            file_path = os.path.join(data_dir, file_name)
            if not os.path.exists(file_path):
                # download_file_from_google_drive(file_url.split("/")[5], file_path)
                download_file(file_url, file_path)
                print(f"Downloaded {file_name} to {data_dir}")

            if not os.path.isdir(file_path.removesuffix(".zip")):
                with zipfile.ZipFile(file_path, "r") as zip_ref:
                    if file_name.removesuffix(".zip") in zip_ref.namelist():
                        extract_path = data_dir
                    else:
                        extract_path = os.path.join(
                            data_dir, file_name.removesuffix(".zip")
                        )
                        os.makedirs(extract_path)
                    zip_ref.extractall(extract_path)
                    print(f"Extracted {file_name} to {extract_path}")

        files_to_remove = glob(os.path.join(data_dir, "**", "__MACOSX"), recursive=True)
        files_to_remove += glob(
            os.path.join(data_dir, "**", ".DS_Store"), recursive=True
        )
        files_to_remove += glob(os.path.join(data_dir, "**", "Icon"), recursive=True)
        for path in files_to_remove:
            if os.path.isdir(path):
                shutil.rmtree(path)
            else:
                os.remove(path)


if __name__ == "__main__":
    assure_babelcalib_downloaded("data/BabelCalib")
