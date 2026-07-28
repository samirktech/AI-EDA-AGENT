
import os
import pandas as pd


def read_uploaded_file(file_path):
    # Get the file extension and convert to lowercase (e.g., '.csv', '.xlsx')
    _, file_extension = os.path.splitext(file_path)
    file_extension = file_extension.lower()

    # Dictionary mapping extensions to their respective pandas read functions
    try:
        if file_extension == ".csv":
            # You can add parameters like sep=',' if needed
            df = pd.read_csv(file_path)

        elif file_extension in [".xls", ".xlsx"]:
            df = pd.read_excel(file_path)

        elif file_extension == ".json":
            df = pd.read_json(file_path)

        elif file_extension == ".parquet":
            df = pd.read_parquet(file_path)

        elif file_extension in [".txt", ".tsv"]:
            # Assuming tab-separated for .txt/.tsv, adjust sep as needed
            df = pd.read_csv(file_path, sep="\t")

        else:
            raise ValueError(f"Unsupported file extension: {file_extension}")

        return df

    except Exception as e:
        print(f"Error reading file: {e}")
        return None


# --- Example Usage ---
# file_path = 'path/to/your/uploaded_file.csv'
# df = read_uploaded_file(file_path)
# print(df.head())
