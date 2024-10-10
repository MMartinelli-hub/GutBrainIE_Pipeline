import pubmed_retriever
import preprocess
import gliner_interface
import grapher_interface
import postprocess
import metatron_adapter
import utils

# Load the configuration file in a local variable
config = utils.load_config('config/config.yaml')

# Load the email from the configuration file to query NCBI API
email = config['email']

# Load the NCBI API Key from the configuration file
ncbi_api_key = config['ncbi_api_key']

# Initialize retriever
retriever = pubmed_retriever.PubMedRetriever(email, ncbi_api_key)

# Retrieve the articles associated to the PMIDs specified in 'pmid_list.csv'
retrieved_articles_df = retriever.process_csv(config['pmid_list_path'])

store_retrieved_articles_path = config['store_retrieved_articles_path']
store_retrieved_articles_file_name = config['store_retrieved_articles_file_name']
retriever.write_articles_together_pubtator(retrieved_articles_df, output_dir=store_retrieved_articles_path, output_file_name=f'{store_retrieved_articles_file_name}.txt')
retriever.write_articles_to_csv(retrieved_articles_df, output_dir=store_retrieved_articles_path, output_file_name=f'{store_retrieved_articles_file_name}.csv')

# Perform pre-processing on retrieved articles (NOT IMPLEMENTED)
preprocess.preprocess_for_gliner(None)

# Initialize GLiNER predictor for predicting entities (NER)
gliner = gliner_interface.GLiNERInterface('config/gliner_config.yaml')

# Run GLiNER pipeline
gliner.run_pipeline()

# Initialize GraphER predictor for predicting relations (RE)
grapher = grapher_interface.GraphERInterface('config/grapher_config.yaml')

# Run GraphER pipeline
grapher.run_pipeline()

# Perform post-processing on predicted entities and relations (NOT IMPLEMENTED)
postprocess.postprocess_annotations(None)

# Upload annotations to MetaTron (NOT IMPLEMENTED)
metatron_adapter.upload_to_metatron(None, None)