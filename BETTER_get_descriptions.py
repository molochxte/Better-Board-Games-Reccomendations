import requests
from bs4 import BeautifulSoup
import re
import json

def extract_description_from_json(data):
    """Recursively search dict/list for a 'description' field."""
    if isinstance(data, dict):
        if "description" in data and isinstance(data["description"], str):
            return data["description"]
        for v in data.values():
            found = extract_description_from_json(v)
            if found:
                return found
    elif isinstance(data, list):
        for v in data:
            found = extract_description_from_json(v)
            if found:
                return found
    return None

def get_long_description(url):
    try:
        response = requests.get(url, timeout=10)
        soup = BeautifulSoup(response.content, "html.parser")

        script_tag = soup.find("script", string=re.compile(r"GEEK\.geekitemPreload"))
        if script_tag:
            match = re.search(
                r"GEEK\.geekitemPreload\s*=\s*(\{.*?\});",
                script_tag.string,
                re.DOTALL,
            )
            if match:
                json_text = match.group(1).rstrip(";")
                data = json.loads(json_text)

                desc_html = extract_description_from_json(data)
                if desc_html:
                    desc_soup = BeautifulSoup(desc_html, "html.parser")
                    return desc_soup.get_text(separator=" ", strip=True)

        return None
    except Exception as e:
        return f"ERROR: {e}"


def save_output_to_text_file(output):
    # open output_descriptions.txt and append the output dictionary as a new line
    with open('output_descriptions.txt', 'a') as f:
        f.write(json.dumps(output) + '\n')
        f.flush()
        
def extract_descriptions_from_txt():
        # Example usage: read cleaned URLs from file and get descriptions
    with open('cleaned_broken_urls.txt', 'r') as f:
        urls = [line.strip() for line in f.readlines() if line.strip()]

    if urls:
        descriptions = {}
        for url in urls:
            description = get_long_description(url)
            descriptions[url] = description
            print(f"URL: {url}\nDescription: {description}\n")

            # save to output_descriptions.txt
            save_output_to_text_file(description)

def extract_descriptions_from_csv_in_batches(batch_size,start_index=0, csv_filename='bgg_dataset_15k.csv'):
    import pandas as pd
    df = pd.read_csv(csv_filename)
    total_urls = len(df)
    for i in range(start_index, total_urls, batch_size):
        batch_df = df.iloc[i:i+batch_size]
        for index, row in batch_df.iterrows():
            url = row['URL']
            description = get_long_description(url)
            df.at[index, 'DESCRIPTION'] = description
            print(f"Processed {index+1}/{total_urls}: {url}")

        # Save progress after each batch
        save_progress(batch_df, filename=f'boardgamegeek_with_descriptions_broken_part_{i//batch_size + 1}.csv')

def save_progress(df, filename='boardgamegeek_with_descriptions_broken.csv'):
    df.to_csv(filename, index=False)
    print(f"Progress saved to {filename}")

if __name__ == "__main__":
    # Uncomment one of the following lines to run the desired function
    # extract_descriptions_from_txt()
    extract_descriptions_from_csv_in_batches(batch_size=10, start_index=0, csv_filename='bgg_dataset_15k.csv')