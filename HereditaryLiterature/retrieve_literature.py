import pandas as pd
from Bio import Entrez
import os

class PubMedRetriever:
    def __init__(self, email, ncbi_api_key):
        self.email = email
        self.ncbi_api_key = ncbi_api_key
        Entrez.email = email
        Entrez.api_key = ncbi_api_key

    def fetch_article_details(self, pmid_list):
        id_list = ",".join(pmid_list)
        handle = Entrez.efetch(db="pubmed", id=id_list, rettype="xml", retmode="xml")
        records = Entrez.read(handle)
        handle.close()

        articles = []
        for record in records['PubmedArticle']:
            article_info = {}

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
            abstract_text = ''.join(abstract_text_list) if abstract_text_list else 'No abstract available'
            article_info['abstract'] = abstract_text.strip()

            # Extract the journal title
            article_info['journal'] = article.get('Journal', {}).get('Title', 'No journal available')

            # Extract the publication year
            pub_date = article.get('Journal', {}).get('JournalIssue', {}).get('PubDate', {})
            year = pub_date.get('Year', None)
            if not year:
                medline_date = pub_date.get('MedlineDate', None)
                year = medline_date[:4] if medline_date else 'Unknown'
            article_info['year'] = year

            # Extract the authors
            authors_list = article.get('AuthorList', [])
            authors = [f"{author.get('ForeName', '')} {author.get('LastName', '')}".strip() for author in authors_list]
            article_info['authors'] = '; '.join(authors) if authors else 'No authors available'

            articles.append(article_info)

        return articles

    def update_csv_with_pubmed_data(self, input_csv):
        df = pd.read_csv(input_csv)
        df['pmid'] = df['pmid'].astype(str)
        pmid_list = df['pmid'].tolist()

        articles = self.fetch_article_details(pmid_list)
        articles_df = pd.DataFrame(articles)

        result_df = pd.merge(df, articles_df, on='pmid', how='left')
        result_df.to_csv(input_csv, index=False, encoding='utf-8')

def list_csv_files(folder_path):
    return [f for f in os.listdir(folder_path) if f.endswith('.csv')]

if __name__ == '__main__':
    email = "marco.martinelli.4@phd.unipd.it"
    ncbi_api_key = "0a4de2bce06c8db5ccf9f92d03d5daab2e0a"
    
    retriever = PubMedRetriever(email, ncbi_api_key)
    
    folder_path = "HereditaryLiterature/old_literature/parkinson"
    csv_files = list_csv_files(folder_path)

    for csv_file in csv_files:
        input_csv = os.path.join(folder_path, csv_file)
        retriever.update_csv_with_pubmed_data(input_csv)

    #file_path = "HereditaryLiterature/old_literature/hereditary_old_merged.csv"
    #retriever.update_csv_with_pubmed_data(file_path)