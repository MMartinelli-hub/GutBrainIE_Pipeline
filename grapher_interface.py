from EnriCo.model import EnriCo

import torch
import re
import os
import logging
import utils
import json

class GraphERInterface:
    def __init__(self, config_file='config/grapher_config.yaml'):
        """
        Initialize the GraphER interface with the pretrained model
        """
        # Load configuration
        self.config = utils.load_config(config_file)

        # Parse GraphER runtime parameters
        self.model_name = self.config['model_name']
        self.threshold = self.config['threshold']
        self.one_by_one_prediction = self.config['one_by_one_prediction']

        # Parse input and output paths
        self.corpus_file_path = self.config['corpus_file_path']
        self.output_directory = self.config['output_directory']
        self.load_from_json = self.config['load_from_json']

        # Create the output directory if it doesn't exist
        if not os.path.exists(self.output_directory):
            os.makedirs(self.output_directory)
        
        # Set up logging
        self.setup_logging()

        # Initialize GLiNER model
        self.initialize_grapher()

        # Initialize other variables
        self.articles = {}
        self.relation_types = list(set(self.config['relation_types'].values())) # Ensure unique labels
        self.predictions = {}
        self.processing_approach = ''
        self.overall_metrics = {}

        # Define if corpus has ground_truth
        self.has_ground_truth = self.config['has_ground_truth']


    def setup_logging(self):
        """
        Sets up logging configuration.
        """
        #pipeline_output_path = os.path.join(self.config['output_directory'], 'grapher_pipeline.log')
        logs_directory = 'logs'
        # Create the output directory if it doesn't exist
        if not os.path.exists(logs_directory):
            os.makedirs(logs_directory)
        
        pipeline_output_path = os.path.join(logs_directory, 'grapher_pipeline.log')

        # Open the log file (create it if it doesn't exist)
        with open(pipeline_output_path, 'a') as f:
            f.write('-'*100)
            f.write('\n\n')

        logging.basicConfig(level=logging.INFO,
                            format='%(asctime)s - %(levelname)s - %(message)s',
                            handlers=[
                                logging.StreamHandler(),
                                logging.FileHandler(pipeline_output_path)
                            ])
        self.logger = logging.getLogger(__name__)


    def initialize_grapher(self):
        """
        Initializes the GLiNER model, using GPU if available.
        """
        self.logger.info("Initializing GraphER model...")
        self.logger.info(f'model_name: {self.model_name}')
        # Use GPU if available
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        # Load the pretrained model for GraphER
        self.model = EnriCo.from_pretrained(self.model_name).to(device)
        # Set the model in evaluation mode
        self.model = self.model.eval()
        self.device = device
        self.logger.info(f"GLiNER model loaded on {self.device}")


    def load_corpus(self):
        """
        Parses the corpus file into a dictionary of articles.
        """
        self.logger.info("Loading and parsing the corpus...")
 
        # Dictionary to hold documents and their annotations, if the flag is set to True in the config file
        # PMID -> {'title': ..., 'author': ..., 'journal': ..., 'year': ..., 'abstract': ..., 'annotations': [...]}
        articles = {}
        
        current_pmid = None

        with open(self.corpus_file_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue # Skip empty line
                if '|t|' in line or '|w|' in line or '|j|' in line or '|y|' in line or '|a|' in line:
                    pmid, field, content = line.split('|', 2)
                    if current_pmid != pmid: # Create a new dictionary entry for each article
                        current_pmid = pmid
                        if self.has_ground_truth:
                            articles[current_pmid] = {'title': '', 'author': '', 'journal': '', 'year': '', 'abstract': '', 'ground_truth': []}
                        else:    
                            articles[current_pmid] = {'title': '', 'author': '', 'journal': '', 'year': '', 'abstract': ''}   
                    if '|t|' in line:
                        articles[current_pmid]['title'] += content
                    elif '|w|' in line:
                        articles[current_pmid]['author'] += content
                    elif '|j|' in line:
                        articles[current_pmid]['journal'] += content
                    elif '|y|' in line:
                        articles[current_pmid]['year'] += content    
                    elif '|a|' in line:
                        articles[current_pmid]['abstract'] += content
                elif self.has_ground_truth:
                    # If available, load ground truth
                    parts = line.split('\t')
                    # Ground truth structure:
                    # {pmid}\t{start_index}\t{end_index}\t{text_span}\t{label} => len: 5
                    if len(parts) == 5:
                        pmid, start_idx, end_idx, text_span, label = parts
                        if label in self.labels:
                            annotation = {
                                'start': int(start_idx),
                                'end': int(end_idx),
                                'text': text_span,
                                'label': label
                            }
                            articles[pmid]['ground_truth'].append(annotation)
                    """
                    parts = line.split(' '|'\t')
                    start_idx = parts[1]
                    end_idx = parts[2]
                    label = parts[-1]
                    text = ''
                    if parts[-2] == 'dietary' or parts[-2] == 'anatomical':
                        label = f'{parts[-2]} {parts[-1]}'
                        for i in range(3, len(parts)-2):
                            text += f'{parts[i]} '
                    else:
                        for i in range(3, len(parts)-1):
                            text += f'{parts[i]} '
                    """
                    
        self.articles = articles

        if self.has_ground_truth:
            print('-'*120)
            for pmid in self.articles:
                title = self.articles[pmid]['title']
                print(f'{pmid}|t|{title}')
                #abstract = self.articles[pmid]['abstract']
                #print(f'{pmid}|a|{abstract}')
                print(f'{pmid}|a|ABSTRACT')

                for ann in self.articles[pmid]['ground_truth']:
                    start_idx = ann['start']
                    end_idx = ann['end']
                    text = ann['text']
                    label = ann['label']
                    print(f'{pmid}\t{start_idx}\t{end_idx}\t{text}\t{label}')
                
                print('')
            print('-'*120)

    
    def load_corpus_from_json(self):
        """
        Parses the corpus from a JSON file into a dictionary variable
        """
        # Load the corpus from JSON to the self.articles dictionary variable
        try:
            self.articles = utils.load_articles_from_json(self.corpus_file_path)
        except(FileNotFoundError, ValueError) as e:
            # TODO: Implement appropriate error handling
            raise ImportError(f'An error occurred while importing JSON file {self.corpus_file_path}: {e}')
        
        # For each entry in self.articles, add the field to store the predicted relations
        for pmid in self.articles:
            self.articles[pmid]['pred_relations'] = []

        #utils.print_structure(self.articles)


    def perform_re(self):
        """
        Performs RE on the documents using GraphER.
        """
        self.logger.info("Performing RE on documents with GraphER...")
        
        # Dictionary to hold predicted relations
        # PMID -> {{'head_start_idx': ..., 'head_end_idx': ..., 'head_text_span': ..., tail_start_idx': ..., 'tail_end_idx': ..., 'tail_text_span': ..., 'relation_type': ..., 'score': ...}, ...}
        predictions = {} 

        def tokenize_text(text):
            """
            Tokenize the text based on whitespaces, separators, and symbols
            """
            return re.findall(r'\w+(?:[-_]\w+)*|\S', text)

        for pmid in self.articles:
            title = self.articles[pmid]['title']
            abstract = self.articles[pmid]['abstract']
            text = f'{title} {abstract}'

            # Remove line breaks from the text and tokenize
            tokens = tokenize_text(text.replace('\n', ''))

            # Prepare the input for the model
            input_x = {'tokenized_text': tokens, 'spans': [], 'relations': []}

            # Call the model's collate function
            x = self.model.collate_fn([input_x], self.relation_types)

            # Predict using the model
            out = self.model.predict(x, self.threshold, output_confidence=True)

            # Process the output to extract relations and their confidence
            relations = []
            for el in out[0]:
                relation = {}
                (s_h, e_h), (s_t, e_t), rtype, conf = el
                relation['head_start_idx'] = s_h
                relation['head_end_idx'] = e_h
                relation['head_text_span'] = ' '.join(tokens[s_h:e_h+1])
                relation['tail_start_idx'] = s_t
                relation['tail_end_idx'] = e_t
                relation['tail_text_span'] = ' '.join(tokens[s_t:e_t+1])
                relation['relation_type'] = rtype
                relation['score'] = conf
                relations.append(relation)
            
            #predictions[pmid] = relations
            self.articles[pmid]['pred_relations'] = relations

        #self.predictions = predictions
        return predictions
    

    def write_predictions_pubtator(self):
        """
        Writes all predicted annotations to a single PubTator-formatted file named 'predictions_pubtator.txt'.
        Includes GLiNER parameters if 'include_configuration_in_output' flag is set to True in the configuration file.
        """
        predictions_file_path = os.path.join(self.output_directory, "predictions_pubtator.txt")

        with open(predictions_file_path, 'w', encoding='utf-8') as pred_file:
            # Write predicted annotations in PubTator format
            for pmid in self.articles:
                title = self.articles[pmid]['title']
                abstract = self.articles[pmid]['abstract']
                pred_annotations = self.articles[pmid]['pred_entities']
                pred_relations = self.articles[pmid]['pred_relations']

                # Write title and abstract
                pred_file.write(f"{pmid}|t|{title}\n")
                pred_file.write(f"{pmid}|a|{abstract}\n")

                # Write predicted annotations
                for ann in pred_annotations:
                    start = ann['start_idx']
                    end = ann['end_idx']
                    tag = ann['tag']
                    text = ann['text_span']
                    label = ann['entity_label']
                    score = ann['score']

                    # Write in PubTator format: PMID \t start_idx \t end_idx \t text_span \t label \t score
                    pred_file.write(f"{pmid}\t{start}\t{end}\t{tag}\t{text}\t{label}\t{score}\n")

                # Write predicted relations
                for relation in pred_relations:
                    head_start_idx = relation['head_start_idx'] 
                    head_end_idx = relation['head_end_idx']
                    head_text_span = relation['head_text_span']
                    tail_start_idx = relation['tail_start_idx']
                    tail_end_idx = relation['tail_end_idx']
                    tail_text_span = relation['tail_text_span']
                    relation_type = relation['relation_type']
                    score = relation['score']

                    # Write in PubTator format: PMID \t head_start_idx \t head_end_idx \t head_text_span \t tail_start_idx \t tail_end_idx \t tail_text_span \t relation_label \t score
                    pred_file.write(f'{pmid}\t{head_start_idx}\t{head_end_idx}\t{head_text_span}\t{tail_start_idx}\t{tail_end_idx}\t{tail_text_span}\t{relation_type}\t{score}')

                pred_file.write("\n")  # Separate documents by a newline

        #print(f"\nAll predictions have been successfully written to {predictions_file_path}\n")
        self.logger.info(f"All predictions have been successfully written to {predictions_file_path}")


    def write_predictions_pubtator_per_pmid(self):
        """
        Writes predictions for each article in a separated file in PubTator format named '<pmid>.txt'
        """
        # Iterate through the articles
        for pmid in self.articles:
            # Define output path
            predictions_file_path = os.path.join(self.output_directory, f'{pmid}.txt')

            with open(predictions_file_path, 'w', encoding='utf-8') as pred_file: # Create and open file for writing
                title = self.articles[pmid]['title']
                abstract = self.articles[pmid]['abstract']
                pred_annotations = self.articles[pmid]['pred_entities']
                pred_relations = self.articles[pmid]['pred_relations']

                # Write title and abstract
                pred_file.write(f"{pmid}|t|{title}\n")
                pred_file.write(f"{pmid}|a|{abstract}\n")

                # Write predicted annotations
                for ann in pred_annotations:
                    start = ann['start_idx']
                    end = ann['end_idx']
                    tag = ann['tag']
                    text = ann['text_span']
                    label = ann['entity_label']
                    score = ann['score']

                    # Write in PubTator format: PMID \t start_idx \t end_idx \t text_span \t label \t score
                    pred_file.write(f"{pmid}\t{start}\t{end}\t{tag}\t{text}\t{label}\t{score}\n")

                # Write predicted relations
                for relation in pred_relations:
                    head_start_idx = relation['head_start_idx'] 
                    head_end_idx = relation['head_end_idx']
                    head_text_span = relation['head_text_span']
                    tail_start_idx = relation['tail_start_idx']
                    tail_end_idx = relation['tail_end_idx']
                    tail_text_span = relation['tail_text_span']
                    relation_type = relation['relation_type']
                    score = relation['score']

                    # Write in PubTator format: PMID \t head_start_idx \t head_end_idx \t head_text_span \t tail_start_idx \t tail_end_idx \t tail_text_span \t relation_label \t score
                    pred_file.write(f'{pmid}\t{head_start_idx}\t{head_end_idx}\t{head_text_span}\t{tail_start_idx}\t{tail_end_idx}\t{tail_text_span}\t{relation_type}\t{score}')

        self.logger.info(f"Predictions for each article have been successfully written separated to {self.output_directory}")


    def store_to_json(self):
        """
        Saves articles and predictions to a JSON file in the format:
        articles = {
            "title": (str)
            "abstract": (str)
            "pred_entities": [
                {
                    "start_idx":    (int)
                    "end_idx":      (int)
                    "tag":          ('t'|'a')
                    "text_span":    (str)
                    "entity_label": (str)
                    "score":        (float)
                }
                {
                    ...
                }
            ]
            "pred_relations": [
                {
                    "head_start_idx":   (int)
                    "head_end_idx":     (int)
                    "head_text_span":   (str)
                    "tail_start_idx":   (int)
                    "tail_end_idx":     (int)
                    "tail_text_span":   (str)
                    "relation_label":   (str)
                    "score":            (float)
                }
                {
                    ...
                }
            ]
        }
        """
        # Convert any non-serializable data if necessary
        def default_serializer(obj):
            if isinstance(obj, set):
                return list(obj)
            # Add other types if needed
            raise TypeError(f'Type {type(obj)} not serializable')

        file_path = os.path.join(self.output_directory, 'grapher_predictions.json')

        with open((file_path), 'w', encoding='utf-8') as f:
            json.dump(self.articles, f, ensure_ascii=False, indent=4, default=default_serializer)

        self.logger.info(f"Predictions have been exported in JSON format to {file_path}")


    def save_results(self):
        """
        Saves the annotations and predictions to files.
        """
        self.logger.info("Saving results to files...")
        self.write_predictions_pubtator()
        self.write_predictions_pubtator_per_pmid()
        self.store_to_json()


    def run_pipeline(self):
        """
        Runs the entire RE pipeline
        """
        if self.load_from_json:
            self.load_corpus_from_json()
        else:
            self.load_corpus()

        self.perform_re()
        #self.compute_metrics()
        self.save_results()
        self.logger.info('Pipeline execution completed.')
        
        """
        print('\n')
        for pmid in self.articles:
            predictions = self.articles[pmid]['pred_relations']
            print('-'*120)
            print(f'Predicted relations for doc {pmid}')
            for relation in predictions:
                head_start_idx = relation['head_start_idx']
                head_end_idx = relation['head_end_idx']
                head_text_span = relation['head_text_span']
                tail_start_idx = relation['tail_start_idx']
                tail_end_idx = relation['tail_end_idx']
                tail_text_span = relation['tail_text_span']
                relation_type = relation['relation_type']
                score = relation['score']
                print(f'\t HEAD: ({head_start_idx},{head_end_idx}) - {head_text_span} \t REL: {relation_type} \t TAIL: ({tail_start_idx},{tail_end_idx}) - {tail_text_span} \t CONF: {score}')
        """
        

# Entry point
if __name__ == '__main__':
    pipeline = GraphERInterface()
    pipeline.run_pipeline()