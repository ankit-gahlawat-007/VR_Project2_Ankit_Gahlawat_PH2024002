import requests
import gzip
import zipfile
import os
import shutil

"""
Example usage:

from downloader import download_and_extract

# Example usage:
zip_url = "https://huggingface.co/kushaaagr/Vilt-finetuned-for-VQA/resolve/main/vilt-finetuned-vqa-15.zip"
output_directory = "."
extracted_path = download_and_extract(zip_url, output_directory)

"""

def download_and_extract(url, output_dir="."):
    """
    Downloads a file from a given URL and extracts its contents
    if it's a gzip or zip file, to the specified output directory
    (defaults to the current directory). For other file types,
    it simply downloads the file.

    Args:
        url (str): The URL of the file to download.
        output_dir (str, optional): The directory to extract the contents to
                                     (for gzip/zip) or save the file to (for others).
                                     Defaults to the current directory.
    """
    os.makedirs(output_dir, exist_ok=True)
    try:
        print(f"Downloading file from: {url}")
        response = requests.get(url, stream=True)
        response.raise_for_status()  # Raise an exception for bad status codes

        filename = url.split("/")[-1]
        filepath = os.path.join(output_dir, filename)
        print(f"Saving downloaded file to: {filepath}")
        with open(filepath, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)

        print(f"Successfully downloaded: {filename}")

        if filename.endswith(".gz"):
            output_filename = filename[:-3]  # Remove the .gz extension
            output_filepath = os.path.join(output_dir, output_filename)
            print(f"Extracting gzip file to: {output_filepath}")
            with gzip.open(filepath, 'rb') as f_in:
                with open(output_filepath, 'wb') as f_out:
                    shutil.copyfileobj(f_in, f_out)
            print(f"Successfully extracted to: {output_filepath}")
            os.remove(filepath)
            print(f"Removed the downloaded gzip file: {filepath}")
        elif filename.endswith(".zip"):
            print(f"Unzipping file to: {output_dir}")
            with zipfile.ZipFile(filepath, 'r') as zip_ref:
                zip_ref.extractall(output_dir)
            print(f"Successfully unzipped to: {output_dir}")
            os.remove(filepath)
            print(f"Removed the downloaded zip file: {filepath}")
        else:
            print(f"Downloaded file is not a gzip or zip file. Saved to: {filepath}")

    except requests.exceptions.RequestException as e:
        print(f"Error during download: {e}")
    except gzip.BadGzipFile:
        print(f"Error: The downloaded file is not a valid gzip file.")
    except zipfile.BadZipFile:
        print(f"Error: The downloaded file is not a valid zip file.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

if __name__ == "__main__":
    # Set the working directory to the directory of the script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)
    print("Current Working Directory:", os.getcwd())
    # Example for a zip file
    zip_url = "https://huggingface.co/kushaaagr/Vilt-finetuned-for-VQA/resolve/main/vilt-finetuned-vqa-15.zip"
    # zip_output_dir = "finetuned-vilt"
    zip_output_dir = "."
    download_and_extract(zip_url, zip_output_dir)
    print("\nProcessed zip file.")

    print("Script finished.")