import requests
import csv
import io
import time
import pandas as pd
from typing import List
import pandas as pd
import numpy as np
import re
from typing import List
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder

# ------------------------------
# Configuration and Constants
# ------------------------------
API_HEADERS = {
    'accept': 'text/csv',
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/107.0.0.0 Safari/537.36'
}

API_PARAMS = {
    'format': 'csv',
    'fields': 'NCT Number|Study Title|Study URL|Acronym|Study Status|Brief Summary|Study Results|Conditions|Interventions|Primary Outcome Measures|Secondary Outcome Measures|Other Outcome Measures|Sponsor|Collaborators|Sex|Age|Phases|Enrollment|Funder Type|Study Type|Study Design|Start Date|Primary Completion Date|Completion Date|First Posted|Results First Posted|Last Update Posted|Locations|Study Documents'
}

# ------------------------------
# Core Data Fetching Function
# ------------------------------
def fetch_trials_data(nct_ids: List[str], output_filename: str, delay: float = 1.0):
    """
    Fetch metadata from ClinicalTrials.gov for a list of NCT IDs and write to CSV.
    
    Args:
        nct_ids (List[str]): List of NCT IDs to fetch.
        output_filename (str): Output CSV path.
        delay (float): Delay between requests to be polite to the API.
    """
    with open(output_filename, 'w', newline='', encoding='utf-8') as csv_file:
        csv_writer = csv.writer(csv_file)
        header_written = False

        for nct_id in nct_ids:
            url = f"https://clinicaltrials.gov/api/v2/studies/{nct_id}"
            print(f"Fetching data for {nct_id}...")

            try:
                response = requests.get(url, params=API_PARAMS, headers=API_HEADERS, timeout=30)
                response.raise_for_status()
                text_data = response.text

                if not text_data.strip():
                    print(f"  Warning: Empty response for {nct_id}. Skipping.")
                    continue

                string_file = io.StringIO(text_data)
                csv_reader = csv.reader(string_file)
                header = next(csv_reader)

                if not header_written:
                    csv_writer.writerow(header)
                    header_written = True

                for data_row in csv_reader:
                    csv_writer.writerow(data_row)

                print(f"  Success! Wrote data for {nct_id}.")

            except requests.exceptions.HTTPError as e:
                print(f"  HTTP Error for {nct_id}: {e}")
            except requests.exceptions.RequestException as e:
                print(f"  Network Error for {nct_id}: {e}")

            time.sleep(delay)

    print(f"\n✅ Processing complete. Data saved to '{output_filename}'.")


# ------------------------------
# Utility Functions
# ------------------------------
def load_nct_ids_from_csv(filepath: str, id_column: str = 'Trial_ID') -> List[str]:
    """
    Load NCT IDs from a CSV file.

    Args:
        filepath (str): Path to CSV file.
        id_column (str): Name of the column containing NCT IDs.
    
    Returns:
        List[str]: A list of NCT IDs
    """
    df = pd.read_csv(filepath)
    df.columns = df.columns.str.strip().str.lower().str.replace(' ', '_').str.replace(r'[^\w_]', '', regex=True)
    return df[id_column.lower()].dropna().astype(str).tolist()


if __name__ == '__main__':
    # Example usage:
    # train_ids = load_nct_ids_from_csv('Train.csv')
    # fetch_trials_data(train_ids, 'historical_trials_data.csv')
    # test_ids = load_nct_ids_from_csv('Test.csv')
    # fetch_trials_data(test_ids, 'active_trials_data.csv')
    pass