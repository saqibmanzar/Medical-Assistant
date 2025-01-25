import requests
import json
import os
import time

# Directory to save the JSON files
SAVE_DIR = "pubchem_data"
os.makedirs(SAVE_DIR, exist_ok=True)

# Retry parameters
MAX_RETRIES = 5
RETRY_DELAY = 5  # seconds

# Function to download data with retry logic
def download_pubchem_data(start_id, end_id):
    for compound_id in range(start_id, end_id + 1):
        url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug_view/data/compound/{compound_id}/JSON/"
        
        attempt = 0
        while attempt < MAX_RETRIES:
            try:
                print(f"Downloading data for compound ID: {compound_id} (Attempt {attempt + 1})")
                response = requests.get(url)
                response.raise_for_status()  # Raise an HTTPError for bad responses

                # Save the JSON data to a file
                file_path = os.path.join(SAVE_DIR, f"compound_{compound_id}.json")
                with open(file_path, "w") as file:
                    json.dump(response.json(), file, indent=4)

                print(f"Data for compound ID {compound_id} saved to {file_path}\n")
                break  # Exit retry loop if successful
            except requests.exceptions.HTTPError as http_err:
                print(f"HTTP error occurred for compound ID {compound_id}: {http_err}\n")
                break  # Stop retrying if it's an HTTP error
            except requests.exceptions.RequestException as req_err:
                print(f"Request error occurred for compound ID {compound_id}: {req_err}\n")
                attempt += 1
                if attempt < MAX_RETRIES:
                    print(f"Retrying in {RETRY_DELAY} seconds...\n")
                    time.sleep(RETRY_DELAY)  # Wait before retrying
            except Exception as err:
                print(f"An unexpected error occurred for compound ID {compound_id}: {err}\n")
                break  # Stop retrying on unexpected errors

# Specify the range of compound IDs you want to download
start_id = 7929
end_id = 100000000  # Adjust this range as needed

download_pubchem_data(start_id, end_id)
