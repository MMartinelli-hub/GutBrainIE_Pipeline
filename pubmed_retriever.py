# pubmed_retriever.py
"""
This file defines a class that handles the interactions with PubMed
"""

import utils

import pandas as pd
from Bio import Entrez

import os


class PubMedRetriever:
    def __init__(self, email):
        """
        Initialize the PubMedRetriever with the user's email.

        Parameters:
        email (str): User's email for NCBI API.
        """
        self.email = email
        Entrez.email = email # Set the email globally for Entrez


    def fetch_abstract(self, pmid_list):
        """
        Fetches abstracts from PubMed given a list of PMIDs in text format.

        Parameters:
        pmid_list (list of str): List of PubMed IDs.

        Returns:
        str: Concatenated articles information (including title, author, journal, doi, and abstract) as a single string.
        """
        # Convert the list of PMIDs into a comma-separated string
        id_list = ",".join(pmid_list)

        # Fetch the data from PubMed
        handle = Entrez.efetch(db="pubmed", id=id_list, rettype="abstract", retmode="text")
        abstracts = handle.read()
        handle.close()

        return abstracts


    def fetch_article_details(self, pmid_list):
        """
        Fetches article details from PubMed given a list of PMIDs in XML format.

        Parameters:
        pmid_list (list of str): List of PubMed IDs.

        Returns:
        list of dict: List containing article details for each PMID.
        """
        # Convert the list of PMIDs into a comma-separated string
        id_list = ",".join(pmid_list)

        # Fetch the data from PubMed in XML format
        handle = Entrez.efetch(db="pubmed", id=id_list, rettype="xml", retmode="xml")
        records = Entrez.read(handle)
        handle.close()

        articles = []

        # Iterate over the records and extract relevant fields
        for record in records['PubmedArticle']:
            article_info = {}

            # Access the MedlineCitation and Article sections
            medline_citation = record.get('MedlineCitation', {})
            article = medline_citation.get('Article', {})

            # Get PMID
            pmid = medline_citation.get('PMID', None)
            if pmid:
                article_info['pmid'] = str(pmid)

            # Extract the article title
            article_info['title'] = article.get('ArticleTitle', 'No title available')

            # Extract the abstract text
            abstract_text_list = article.get('Abstract', {}).get('AbstractText', [])
            if abstract_text_list:
                # Handle cases where AbstractText is a list of paragraphs
                if isinstance(abstract_text_list, list):
                    abstract_text = ''
                    for item in abstract_text_list:
                        if isinstance(item, str):
                            abstract_text += item + ' '
                        elif isinstance(item, dict):
                            # For cases where the abstract is structured with labels
                            abstract_text += item.get('text', '') + ' '
                    article_info['abstract'] = abstract_text.strip()
                else:
                    article_info['abstract'] = str(abstract_text_list)
            else:
                article_info['abstract'] = 'No abstract available'

            # Extract the journal title
            journal = article.get('Journal', {}).get('Title', 'No journal available')
            article_info['journal'] = journal

            # Extract the publication year
            pub_date = article.get('Journal', {}).get('JournalIssue', {}).get('PubDate', {})
            year = pub_date.get('Year', None)

            # If 'Year' is not available, try 'MedlineDate' (e.g., '2012 Nov-Dec')
            if not year:
                medline_date = pub_date.get('MedlineDate', None)
                if medline_date:
                    # Extract the year from the 'MedlineDate' string
                    year = medline_date[:4]  # Assumes the year is the first four characters

            # If still not found, try 'ArticleDate'
            if not year:
                article_date_list = article.get('ArticleDate', [])
                if article_date_list:
                    year = article_date_list[0].get('Year', None)

            # If still not found, set as 'Unknown'
            if not year:
                year = 'Unknown'

            article_info['year'] = year

            # Extract the authors
            authors_list = article.get('AuthorList', [])
            authors = []
            for author in authors_list:
                last_name = author.get('LastName', '')
                fore_name = author.get('ForeName', '')
                full_name = f"{fore_name} {last_name}".strip()
                if full_name:
                    authors.append(full_name)
            article_info['authors'] = authors

            # Append the article information to the list
            articles.append(article_info)

        return articles
        

    def process_csv(self, input_csv, exclude_csv=None):
        """
        Reads PMIDs from a CSV file and fetches article details.

        Parameters:
        input_csv (str): Path to the CSV file with a column 'pmid'.
        exclude_csv (str): Path to the CSV file with a column 'pmid', defining the articles to exclude (optional)

        Returns:
        pd.DataFrame: A DataFrame containing the article details along with 'ref' and 'pmid'.
        """
        # Read the CSV file into a DataFrame
        df = pd.read_csv(input_csv)
        df['pmid'] = df['pmid'].astype(str)  # Ensure PMIDs are strings

        # If a CSV file for exclusion is defined
        if exclude_csv:
            # Read the exclude CSV file and get PMIDs to exclude
            exclude_df = pd.read_csv(exclude_csv)
            exclude_pmids = exclude_df['pmid'].astype(str).tolist()

            # Filter out rows where 'pmid' is in exclude_pmids
            df = df[~df['pmid'].isin(exclude_pmids)]

            # Get the list of PMIDs from the filtered DataFrame
            pmid_list = df['pmid'].tolist()

        # Get the list of PMIDs
        pmid_list = df['pmid'].tolist()

        # Fetch article details for PMIDs
        articles = self.fetch_article_details(pmid_list)

        # Convert the list of article dictionaries to a DataFrame
        articles_df = pd.DataFrame(articles)

        # Merge the original DataFrame with the articles DataFrame on 'pmid'
        result_df = pd.merge(df, articles_df, on='pmid', how='left')

        # Remove duplicates 
        result_df = result_df.drop_duplicates(subset=['pmid'])

        return result_df
    

    def write_articles_separated_pubtator(self, articles_df, output_dir='retrieved_articles'):
        """
        Writes each article's details into a separate text file in PubTator format, named '*pmid*.txt'.

        Parameters:
        articles_df (pd.DataFrame): DataFrame containing article details.
        output_dir (str): Path to the directory where the article files will be saved.
        """
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        # Iterate over each article in the DataFrame
        for index, row in articles_df.iterrows():
            pmid = str(row['pmid'])

            # Open a file named 'pmid.txt' for writing
            file_path = os.path.join(output_dir, f'{pmid}.txt')
            with open(file_path, 'w', encoding='utf-8') as f:
                # Write the title
                title = row.get('title', 'No title available').replace('\n', ' ').strip()
                f.write(f"{pmid}|t|{title}\n")

                # Write the authors
                authors = row.get('authors', [])
                authors_str = '; '.join(authors) if authors else 'No authors available'
                f.write(f"{pmid}|w|{authors_str}\n")

                # Write the journal
                journal = row.get('journal', 'No journal available').replace('\n', ' ').strip()
                f.write(f"{pmid}|j|{journal}\n")

                # Write the pubblication year
                year = row.get('year', 'No year available').replace('\n', ' ').strip()
                f.write(f"{pmid}|y|{year}\n")

                # Write the abstract
                abstract = row.get('abstract', 'No abstract available').replace('\n', ' ').strip()
                f.write(f"{pmid}|a|{abstract}\n")

        print(f"Articles have been written to '{file_path}'.")
 

    def write_articles_together_pubtator(self, articles_df, output_dir='retrieved_articles', output_file_name='retrieved_articles.txt'):
        """
        Writes each article's details into a unique text file in PubTator format.

        Parameters:
        articles_df (pd.DataFrame): DataFrame containing article details.
        output_dir (str): Path to the directory where the file will be saved.
        output_file_name (str): Name of the output file.
        """
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        # Open a file for writing
        file_path = os.path.join(output_dir, output_file_name)
        with open(file_path, 'w', encoding='utf-8') as f:
            # Iterate over each article in the DataFrame
            for index, row in articles_df.iterrows():
                # Get the PMID of the article
                pmid = str(row['pmid'])

                # Write the title
                title = row.get('title', 'No title available').replace('\n', ' ').strip()
                f.write(f"{pmid}|t|{title}\n")

                # Write the authors
                authors = row.get('authors', [])
                authors_str = '; '.join(authors) if authors else 'No authors available'
                f.write(f"{pmid}|w|{authors_str}\n")

                # Write the journal
                journal = row.get('journal', 'No journal available').replace('\n', ' ').strip()
                f.write(f"{pmid}|j|{journal}\n")

                # Write the pubblication year
                year = row.get('year', 'No year available').replace('\n', ' ').strip()
                f.write(f"{pmid}|y|{year}\n")

                # Write the abstract
                abstract = row.get('abstract', 'No abstract available').replace('\n', ' ').strip()
                f.write(f"{pmid}|a|{abstract}\n")

                f.write(f"\n\n")

        print(f"Articles have been written to '{file_path}'.")
 

    def write_articles_to_csv(self, articles_df, output_dir='retrieved_articles', output_file_name='retrieved_articles.csv'):
        """
        Writes all articles' details into a single CSV file.

        Parameters:
        articles_df (pd.DataFrame): DataFrame containing article details.
        output_dir (str): Path to the directory where the CSV file will be saved.
        output_file_name (str): Name of the output CSV file.
        """
        # Ensure the 'authors' field is a string
        articles_df['authors'] = articles_df['authors'].apply(
            lambda authors: '; '.join(authors) if isinstance(authors, list) else authors
        )

        # Replace missing values with default messages
        articles_df['title'] = articles_df['title'].fillna('No title available')
        articles_df['authors'] = articles_df['authors'].fillna('No authors available')
        articles_df['journal'] = articles_df['journal'].fillna('No journal available')
        articles_df['year'] = articles_df['year'].fillna('No year available')
        articles_df['abstract'] = articles_df['abstract'].fillna('No abstract available')

        # Select and order the desired columns
        output_df = articles_df[['pmid', 'title', 'authors', 'journal', 'year', 'abstract']]

        # Write the DataFrame to a CSV file
        file_path = os.path.join(output_dir, output_file_name)
        output_df.to_csv(file_path, index=False, encoding='utf-8')

        print(f"Articles have been written to '{file_path}'.")


# Example usage
if __name__ == '__main__':
    # Load the email from the config file
    config = utils.load_config('data_retrieval/config.yaml')
    email = config['email']

    # Flags to decide what to test
    test_fetch_text_format = config['test_fetch_text_format']
    test_fetch_xml_format = config['test_fetch_xml_format']
    test_fetch_from_csv = config['test_fetch_from_csv']

    # Initialize retriever
    retriever = PubMedRetriever(email)

    # Define path for CSV file
    input_csv = config['path_to_PMID_list']  # Load CSV file path from config file
    exclude_csv = config['path_to_PMID_exclusion_list'] 

    # Example PMIDs
    pmid_list = ['23064760', '22612585']

    # Fetch and print the abstracts
    if test_fetch_text_format:
        print("Fetching abstracts in text format:")
        abstracts = retriever.fetch_abstract(pmid_list)
        print(abstracts)
        print("-" * 80)

    # Fetch and print the article details
    if test_fetch_xml_format:
        print("Fetching article details:")
        articles = retriever.fetch_article_details(pmid_list)
        for article in articles:
            print(f"PMID: {article['pmid']}")
            print(f"Title: {article['title']}")
            print(f"Journal: {article['journal']}")
            print(f"Year: {article['year']}")
            print("Authors:")
            for author in article['authors']:
                print(f"  {author}")
            print(f"Abstract: {article['abstract']}\n")
            print("-" * 80)

    # Process PMIDs from a CSV file
    # The CSV file should have columns 'ref' and 'pmid'
    if test_fetch_from_csv:
        result_df = retriever.process_csv(input_csv, exclude_csv)
        print("Processed DataFrame from CSV:")
        print(result_df.head())

        print('\n\n')
        for idx, row in result_df.iterrows():
            print(f'idx: {idx} \t\t row: {row}')

        output_dir = config['retrieved_articles_output_dir']
        retriever.write_articles_separated_pubtator(result_df, output_dir)
        retriever.write_articles_together_pubtator(result_df, output_dir)
        retriever.write_articles_to_csv(result_df, output_dir)
