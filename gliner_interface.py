# gliner_interface.py

import os
import argparse
import logging
import numpy as np
import pandas as pd
from gliner import GLiNER
from concurrent.futures import ThreadPoolExecutor, as_completed
import torch
import utils


class GLiNERInterface:
    def __init__(self, config_file='config/gliner_config.yaml'):
        """
        Initialize the GLiNERInterface with configurations from a YAML file.
        Command-line arguments can override these configurations
        """
        # Load configuration
        self.config = utils.load_config(config_file)

        # Parse GLiNER runtime parameters
        self.model_name = self.config['model_name']
        self.threshold = self.config['threshold']        
        self.processing = self.config['processing']        
        self.flat_ner = self.config['flat_ner']        
        self.multi_label = self.config['multi_label']
        
        # Parse input and output paths
        self.corpus_file_path = self.config['corpus_file_path']
        self.output_directory = self.config['output_directory']

        # Create the output directory if it doesn't exist
        if not os.path.exists(self.output_directory):
            os.makedirs(self.output_directory)
        
        # Set up logging
        self.setup_logging()

        # Initialize GLiNER model
        self.initialize_gliner()
        # Initialize other variables
        self.articles = {}
        self.labels = list(set(self.config['labels'].values())) # Ensure unique labels
        self.predictions = {}
        self.processing_approach = ''
        self.overall_metrics = None


    def setup_logging(self):
        """
        Sets up logging configuration.
        """
        #pipeline_output_path = os.path.join(self.config['output_directory'], 'pipeline.log')
        pipeline_output_path = 'pipeline.log'

        # Create the log file and, if already existing, make it empty
        with open(pipeline_output_path, 'w+') as f:
            f.write('')

        logging.basicConfig(level=logging.INFO,
                            format='%(asctime)s - %(levelname)s - %(message)s',
                            handlers=[
                                logging.StreamHandler(),
                                logging.FileHandler(pipeline_output_path)
                            ])
        self.logger = logging.getLogger(__name__)


    def initialize_gliner(self):
        """
        Initializes the GLiNER model, using GPU if available.
        """
        self.logger.info("Initializing GLiNER model...")
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model = GLiNER.from_pretrained(self.model_name).to(device)
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
                    if current_pmid != pmid:
                        current_pmid = pmid
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
                
        self.articles = articles


    def perform_ner(self):
        """
        Performs NER on the documents using GLiNER.
        """
        self.logger.info("Performing NER on documents with GLiNER...")
        # Choose the processing approach based on configuration
        if self.processing == 'sentence':
            self.predictions = self.process_documents_sentence_level()
            self.processing_approach = "sentence_by_sentence"
        elif self.processing == 'label':
            self.predictions = self.process_documents_label_by_label()
            self.processing_approach = "label_by_label"
        else:
            self.predictions = self.process_documents()
            self.processing_approach = "document_level"


    def process_documents(self):
        """
        Perform NER on the titles and abstracts at the document level.
        """
        self.logger.info("Processing articles at the document level...")

        # Parse GLiNER parameters to local variables
        threshold = self.threshold
        flat_ner = self.flat_ner
        multi_label = self.multi_label
        
        # Dictionary to hold predicted annotations
        # PMID -> {{'start_idx': ..., 'end_idx': ..., 'text_span': ..., 'label': ..., 'score': ...}, ...}
        predictions = {} 

        for pmid, content in self.articles.items():
            title = content['title']
            abstract = content['abstract']

            # Predict entities 
            title_entities = self.model.predict_entities(title, self.labels, threshold=threshold, flat_ner=flat_ner, multi_label=multi_label)
            abstract_entities = self.model.predict_entities(abstract, self.labels, threshold=threshold, flat_ner=flat_ner, multi_label=multi_label)

            # Adjust indices for annotations in the abstract
            for entity in abstract_entities:
                entity['start'] += len(title)
                entity['end'] += len(title)

            # Remove duplicates from predicted entities
            unique_entities = []
            seen_entities = set()

            # Remove duplicates from title entities and add tag field with value 't'
            for entity in title_entities:
                key = (entity['start'], entity['end'], entity['text'], entity['label'], entity['score'])
                if key not in seen_entities:
                    entity['tag'] = 't'
                    unique_entities.append(entity)
                    seen_entities.add(key)

            # Remove duplicates from abstract entities and add tag field with value 'a'
            for entity in abstract_entities:
                key = (entity['start'], entity['end'], entity['text'], entity['label'], entity['score'])
                if key not in seen_entities:
                    entity['tag'] = 'a'
                    unique_entities.append(entity)
                    seen_entities.add(key)

            predictions[pmid] = unique_entities
            
        return predictions


    def process_documents_sentence_level(self):
        """
        Perform NER on the titles and abstracts at the sentence level, i.e., using the '.' symbol as delimiter.
        """
        self.logger.info("Processing articles at the sentence level...")

        # Parse GLiNER parameters from configuration
        threshold = self.threshold
        flat_ner = self.flat_ner
        multi_label = self.multi_label
        
        # Dictionary to hold predicted annotations
        # PMID -> {{'start_idx': ..., 'end_idx': ..., 'text_span': ..., 'label': ..., 'score': ...}, ...}
        predictions = {} 

        for pmid, content in self.articles.items():
            title = content['title']
            abstract = content['abstract']

            # Predict title entities 
            title_entities = self.model.predict_entities(title, self.labels, threshold=threshold, flat_ner=flat_ner, multi_label=multi_label)
            
            # Split abstract into sentences
            separator = '. '
            sentences = abstract.split(separator)
            sentence_offsets = []
            offset = len(title)
            separator_length = len(separator)
            
            # Define the sentences offsets
            for sentence in sentences:
                sentence_offsets.append(offset)
                offset += len(sentence) + separator_length  # Adjust for separator length

            # Define the variable to hold predicted abstract entities
            abstract_entities = []

            # Predict abstract entities sentence by sentence
            for idx, sentence in enumerate(sentences):
                sentence_entities = self.model.predict_entities(sentence, self.labels, threshold=threshold, flat_ner=flat_ner, multi_label=multi_label)
                # Adjust entity positions to the original text
                for entity in sentence_entities:
                    adjusted_entity = {
                        'start': entity['start'] + sentence_offsets[idx],
                        'end': entity['end'] + sentence_offsets[idx],
                        'text': entity['text'],
                        'label': entity['label'],
                        'score': entity['score']
                    }
                    abstract_entities.append(adjusted_entity)

            # Remove duplicates from predicted entities
            unique_entities = []
            seen_entities = set()

            # Remove duplicates from title entities and add tag field with value 't'
            for entity in title_entities:
                key = (entity['start'], entity['end'], entity['text'], entity['label'], entity['score'])
                if key not in seen_entities:
                    entity['tag'] = 't'
                    unique_entities.append(entity)
                    seen_entities.add(key)

            # Remove duplicates from abstract entities and add tag field with value 'a'
            for entity in abstract_entities:
                key = (entity['start'], entity['end'], entity['text'], entity['label'], entity['score'])
                if key not in seen_entities:
                    entity['tag'] = 'a'
                    unique_entities.append(entity)
                    seen_entities.add(key)

                """
                This part showcase an example of labels aggregation
                """
                if False:
                    if entity['label'] in ['Gene', 'Gene Product']:
                        entity['label'] = 'GeneorGeneProduct'
                    elif entity['label'] in ['Disease', 'Phenotypic Feature']:
                        entity['label'] = 'DiseaseOrPhenotypicFeature'
                    elif entity['label'] in ['Chemical', 'Chemical Entity']:
                        entity['label'] = 'ChemicalEntity'
                    elif entity['label'] in ['Organism', 'Organism Taxonomy']:
                        entity['label'] = 'OrganismTaxon'

            predictions[pmid] = unique_entities
        
        """
        This part showcase an example of labels aggregation
        """
        if False:
            self.labels = ['GeneorGeneProduct', 'DiseaseOrPhenotypicFeature', 'ChemicalEntity', 'OrganismTaxon']
 
        return predictions


    def process_documents_label_by_label(self):
        """
        Perform NER on the titles and abstracts at the sentence level, i.e., using the '.' symbol as delimiter, predicting one label at a time.
        """
        self.logger.info("Processing articles at the sentence level, one label at a time...")

        # Parse GLiNER parameters from configuration
        threshold = self.threshold
        flat_ner = self.flat_ner
        multi_label = self.multi_label
        
        # Dictionary to hold predicted annotations
        # PMID -> {{'start_idx': ..., 'end_idx': ..., 'text_span': ..., 'label': ..., 'score': ...}, ...}
        predictions = {} 

        for pmid, content in self.articles.items():
            title = content['title']
            abstract = content['abstract']

            # Predict title entities 
            title_entities = self.model.predict_entities(title, self.labels, threshold=threshold, flat_ner=flat_ner, multi_label=multi_label)
            
            # Split abstract into sentences
            separator = '. '
            sentences = abstract.split(separator)
            sentence_offsets = []
            offset = len(title)
            separator_length = len(separator)
            for sentence in sentences:
                sentence_offsets.append(offset)
                offset += len(sentence) + separator_length  # Adjust for separator length

            # Define the variable to hold predicted abstract entities
            abstract_entities = []

            # Predict abstract entities sentence by sentence, one label at a time
            for idx, sentence in enumerate(sentences):
                for label in self.labels:
                    # Predict entities for the current label
                    sentence_entities = self.model.predict_entities(sentence, [label], threshold=threshold, flat_ner=flat_ner, multi_label=multi_label)
                    # Adjust entity positions to the original text
                    for entity in sentence_entities:
                        adjusted_entity = {
                            'start': entity['start'] + sentence_offsets[idx],
                            'end': entity['end'] + sentence_offsets[idx],
                            'text': entity['text'],
                            'label': entity['label'],
                            'score': entity['score']
                        }
                        abstract_entities.append(adjusted_entity)

            # Remove duplicates from predicted entities
            unique_entities = []
            seen_entities = set()

            # Remove duplicates from title entities and add tag field with value 't'
            for entity in title_entities:
                key = (entity['start'], entity['end'], entity['text'], entity['label'], entity['score'])
                if key not in seen_entities:
                    entity['tag'] = 't'
                    unique_entities.append(entity)
                    seen_entities.add(key)

            # Remove duplicates from abstract entities and add tag field with value 'a'
            for entity in abstract_entities:
                key = (entity['start'], entity['end'], entity['text'], entity['label'], entity['score'])
                if key not in seen_entities:
                    entity['tag'] = 'a'
                    unique_entities.append(entity)
                    seen_entities.add(key)
                    
        return predictions


    def compute_metrics(self):
        """
        Computes evaluation metrics for the predictions.
        """
        self.logger.info("Computing evaluation metrics...")

        # Define variables to store predictions counts
        total_predictions = 0  
        predictions_per_label = dict.fromkeys(self.labels, 0)

        # Iterate through the predictions for each article and update the counts
        for pmid in self.articles:
            for ann in self.predictions[pmid]:
                total_predictions += 1
                predictions_per_label[ann['label']] += 1


        print(f'Num processed articles:\t {len(self.articles)}')
        print(f'Num predicted entities:\t {total_predictions}')
        print(f'Avg predicted entities per article:\t {round(len(self.articles)/total_predictions, 3)}')
        for label in sorted(predictions_per_label, key=predictions_per_label.get):
            print(f'Num of {label.upper()} entities predicted:\t {predictions_per_label[label]}')

        self.logger.info(f'Num processed articles:\t {len(self.articles)}')
        self.logger.info(f'Num predicted entities:\t {total_predictions}')
        self.logger.info(f'Avg predicted entities per article:\t {round(len(self.articles)/total_predictions, 3)}')
        for label in sorted(predictions_per_label, key=predictions_per_label.get):
            self.logger.info(f'Num of {label.upper()} entities predicted:\t {predictions_per_label[label]}')

        # Implement metric computation logic
        # Store the result in self.overall_metrics
        

    def write_predictions_pubtator(self):
        """
        Writes all predicted annotations to a single PubTator-formatted file named 'predictions_pubtator.txt'.
        Includes GLiNER parameters if 'include_configuration_in_output' flag is set to True in the configuration file.
        """
        predictions_file_path = os.path.join(self.output_directory, "predictions_pubtator.txt")

        with open(predictions_file_path, 'w', encoding='utf-8') as pred_file:
            if self.config['include_configuration_in_output']:
                # Write GLiNER parameters at the top
                pred_file.write("## GLiNER Parameters ##\n")
                pred_file.write(f"Model Used: {self.model_name}\n")
                pred_file.write(f"Threshold: {self.threshold}\n")
                pred_file.write(f"Flat NER: {self.flat_ner}\n")
                pred_file.write(f"Labels: {', '.join(self.labels)}\n")
                pred_file.write(f"Processing Approach: {self.processing}\n")
                # Write separators
                pred_file.write("-"*100 + "\n")

            # Write predicted annotations in PubTator format
            for pmid in self.articles:
                title = self.articles[pmid]['title']
                abstract = self.articles[pmid]['abstract']
                pred_annotations = self.predictions.get(pmid, []) # using 'get' in place of '[]' cause no entities might be predicted for a certain article

                # Write title and abstract
                pred_file.write(f"{pmid}|t|{title}\n")
                pred_file.write(f"{pmid}|a|{abstract}\n")

                # Write predicted annotations
                for ann in pred_annotations:
                    start = ann['start']
                    end = ann['end']
                    text = ann['text']
                    label = ann['label']
                    score = ann['score']

                    # Write in PubTator format: PMID \t start_idx \t end_idx \t text_span \t label \t score
                    # Here, CUI is replaced with the semantic type name
                    pred_file.write(f"{pmid}\t{start}\t{end}\t{text}\t{text}\t{label}\t{score}\n")

                pred_file.write("\n")  # Separate documents by a newline

        print(f"\nAll predictions have been successfully written to {predictions_file_path}\n")
        self.logger.info(f"All predictions have been successfully written to {predictions_file_path}")


    def save_results(self):
        """
        Saves the annotations and predictions to files.
        """
        
        self.logger.info("Saving results to files...")
        self.write_predictions_pubtator()
        """
        for pmid in self.articles:
            title = self.articles[pmid]['title']
            author = self.articles[pmid]['author']
            journal = self.articles[pmid]['journal']
            year = self.articles[pmid]['year']
            abstract = self.articles[pmid]['abstract']

            print(f'{pmid}|t|{title}')
            print(f'{pmid}|w|{author}')
            print(f'{pmid}|j|{journal}')
            print(f'{pmid}|y|{year}')
            print(f'{pmid}|a|{abstract}')

            pred_annotations = self.predictions[pmid]
            for ann in pred_annotations:
                start = ann['start']
                end = ann['end']
                text_span = ann['text']
                label = ann['label']
                score = ann['score']

                print(f'{start}\t{end}\t{text_span}\t{label}\t{score}')
        """        
        # Implement logic to save annotations per PMID


    def run_pipeline(self):
        """
        Runs the entire NER pipeline.
        """
        self.load_corpus()
        self.perform_ner()
        self.compute_metrics()
        self.save_results()
        self.logger.info("Pipeline execution completed.")

    # Additional methods for parsing, filtering, NER processing, etc.

# Entry point
if __name__ == '__main__':
    pipeline = GLiNERInterface()
    pipeline.run_pipeline()




#def parse_arguments(self):
#        """
#        Parses command-line arguments and overrides configurations.
#        """
#        parser = argparse.ArgumentParser(description="GLiNER-based NER Pipeline", add_help=True)
#
#        parser.add_argument('-c', '--config', type=str, default='gliner_config.yaml',
#                            help='Path to the configuration YAML file (default: gliner_config.yaml)')
#        parser.add_argument('-t', '--threshold', type=float,
#                            help='Threshold value for GLiNER predictions')
#        parser.add_argument('-proc', '--processing', type=str, choices=['sentence', 'label', 'document'],
#                            help='Level of processing: sentence, label, document')
#        parser.add_argument('-f', '--flat', action='store_true',
#                            help='Enable flat NER (no overlapping entities)')
#        parser.add_argument('-ml', '--multi_label', action='store_true',
#                            help='Enable multi-label extraction')
#        parser.add_argument('-minc', '--min_count', type=int,
#                            help='Minimum number of annotated entities per document')
#        parser.add_argument('-topx', '--top_x', type=int,
#                            help='Number of top annotated documents to retain')
#        parser.add_argument('-filter', '--filtering_method', type=str, choices=['iqr', 'mad'],
#                            help='Filtering method to apply: iqr or mad')
#        parser.add_argument('-cfp', '--corpus_file_path', type=str,
#                            help='Path to the corpus file')
#        parser.add_argument('-sfp', '--srdef_file_path', type=str,
#                            help='Path to the SRDEF file')
#        parser.add_argument('-od', '--output_directory', type=str,
#                            help='Directory to store output files')
#        
#        args = parser.parse_args()
#
#        # Override configurations with command-line arguments if provided
#        arg_dict = vars(args)
#        for key, value in arg_dict.items():
#            if value is not None and key in self.config:
#                self.config[key] = value


"""
BioRED details
GeneOrGeneProduct|4430
DiseaseOrPhenotypicFeature|3646
ChemicalEntity|2853
OrganismTaxon|1429
SequenceVariant|890
CellLine|103
"""