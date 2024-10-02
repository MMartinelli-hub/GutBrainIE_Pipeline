import pubmed_retriever
import preprocess
import gliner_interface
import postprocess
import utils

# Load the configuration file in a local variable
config = utils.load_config('config/config.yaml')

# Load the email from the configuration file to query PubMed API
email = config['email']

# Initialize retriever
retriever = pubmed_retriever.PubMedRetriever(email)

# Retrieve the articles associated to the PMIDs specified in 'pmid_list.csv'
retrieved_articles_df = retriever.process_csv(config['pmid_list_path'])

retriever.write_articles_together_pubtator(retrieved_articles_df, config['store_retrieved_articles_path'])
retriever.write_articles_to_csv(retrieved_articles_df, config['store_retrieved_articles_path'])

# Perform pre-processing on retrieved articles (NOT IMPLEMENTED)
preprocess.preprocess_for_gliner(None)

# Initialize GLiNER predictor
gliner = gliner_interface.GLiNERInterface('config/gliner_config.yaml')

# Perform post-processing on retrieved articles and predicted annotations (NOT IMPLEMENTED)
postprocess.postprocess_annotations(None)

# Run GLiNER pipeline
gliner.run_pipeline()