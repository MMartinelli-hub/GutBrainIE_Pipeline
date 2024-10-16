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

import json


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
        self.entity_labels = list(set(self.config['entity_labels'].values())) # Ensure unique labels
        self.entity_labels = [l.lower() for l in self.entity_labels] # Make labels lower-cased (needed for NuZero)
        self.processing_approach = ''
        self.overall_metrics = {}

        # Define if corpus has ground_truth
        self.has_ground_truth = self.config['has_ground_truth']


    def setup_logging(self):
        """
        Sets up logging configuration.
        """        
        logs_directory = 'logs'
        # Create the output directory if it doesn't exist
        if not os.path.exists(logs_directory):
            os.makedirs(logs_directory)

        pipeline_output_path = os.path.join(logs_directory, 'gliner_pipeline.log')

        # Open the log file (create it if it doesn't exist)
        with open(pipeline_output_path, 'a') as f:
            f.write('-'*100)
            f.write('\n\n')

        # Initialize logger
        logging.basicConfig(level=logging.INFO,
                            format='%(asctime)s - %(levelname)s - %(message)s',
                            handlers=[
                                logging.StreamHandler(),
                                logging.FileHandler(pipeline_output_path)
                            ])

        # Assign logger to class variable
        self.logger = logging.getLogger(__name__)


    def initialize_gliner(self):
        """
        Initializes the GLiNER model, using GPU if available.
        """
        self.logger.info("Initializing GLiNER model...")
        self.logger.info(f'model_name: {self.model_name}, threshold: {self.threshold}, processing: {self.processing}, flat_ner: {self.flat_ner}, multi_label: {self.multi_label}')
        
        # Use GPU if available
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        # Load the pretrained model for GLiNER
        self.model = GLiNER.from_pretrained(self.model_name).to(device)

        self.device = device
        self.logger.info(f"GLiNER model loaded on {self.device}")


    def load_corpus(self):
        """
        Parses the corpus file into a dictionary of articles.
        """
        self.logger.info("Loading and parsing the corpus...")
 
        # Dictionary to hold documents and their ground truth, if the flag is set to True in the config file
        # PMID -> {'title': ..., 'author': ..., 'journal': ..., 'year': ..., 'abstract': ..., 'ground_truth': [...], 'pred_entities': []}
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
                            articles[current_pmid] = {'title': '', 'author': '', 'journal': '', 'year': '', 'abstract': '', 'ground_truth': [], 'pred_entities': []}
                        else:    
                            articles[current_pmid] = {'title': '', 'author': '', 'journal': '', 'year': '', 'abstract': '', 'pred_entities': []}   
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
                        if label in self.entity_labels:
                            gt_entity = {
                                'start_idx': int(start_idx),
                                'end_idx': int(end_idx),
                                'text_span': text_span,
                                'entity_label': label
                            }
                            articles[pmid]['ground_truth'].append(gt_entity)
                    
        self.articles = articles


    def perform_ner(self):
        """
        Performs NER on the documents using GLiNER.
        """
        self.logger.info(f"Performing NER on documents with GLiNER using {self.model_name} as model...")
        # Choose the processing approach based on configuration
        if self.processing == 'sentence':
            self.process_documents_sentence_level()
            self.processing_approach = "sentence_by_sentence"
        elif self.processing == 'label':
            self.process_documents_label_by_label()
            self.processing_approach = "label_by_label"
        else:
            self.process_documents()
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
        
        # Dictionary to hold predicted entities
        # PMID -> {{'start_idx': ..., 'end_idx': ..., 'text_span': ..., 'entity_label': ..., 'score': ...}, ...}
        predictions = {} 

        for pmid, content in self.articles.items():
            title = content['title']
            abstract = content['abstract']

            # Predict entities 
            title_entities = self.model.predict_entities(title, self.entity_labels, threshold=threshold, flat_ner=flat_ner, multi_label=multi_label)
            abstract_entities = self.model.predict_entities(abstract, self.entity_labels, threshold=threshold, flat_ner=flat_ner, multi_label=multi_label)

            if(self.model_name == "numind/NuNerZero"):
                # If the model is NuNerZero, perform entities merging
                def merge_entities(text, entities):
                    if not entities:
                        return []
                    merged = []
                    current = entities[0]
                    for next_entity in entities[1:]:
                        if next_entity['label'] == current['label'] and (next_entity['start'] == current['end'] + 1 or next_entity['start'] == current['end']):
                            current['text'] = text[current['start']: next_entity['end']].strip()
                            current['end'] = next_entity['end']
                        else:
                            merged.append(current)
                            current = next_entity
                    # Append the last entity
                    merged.append(current)
                    return merged
                
                title_entities = merge_entities(title, title_entities)
                abstract_entities = merge_entities(abstract, abstract_entities)

            # Adjust indices for predicted entities in the abstract
            for entity in abstract_entities:
                entity['start'] += len(title) + 1
                entity['end'] += len(title) + 1

            # Remove duplicates from predicted entities
            unique_entities = []
            seen_entities = set()

            # Remove duplicates from title entities and add tag field with value 't'
            for entity in title_entities:
                key = (entity['start'], entity['end'], entity['text'], entity['label'], entity['score'])
                if key not in seen_entities:
                    tmp_entity = {
                        'start_idx': entity['start'],
                        'end_idx': entity['end'],
                        'tag': 't',
                        'text_span': entity['text'],
                        'entity_label': entity['label'],
                        'score': entity['score']
                    }
                    unique_entities.append(tmp_entity)
                    seen_entities.add(key)

            # Remove duplicates from abstract entities and add tag field with value 'a'
            for entity in abstract_entities:
                key = (entity['start'], entity['end'], entity['text'], entity['label'], entity['score'])
                if key not in seen_entities:
                    tmp_entity = {
                        'start_idx': entity['start'],
                        'end_idx': entity['end'],
                        'tag': 'a',
                        'text_span': entity['text'],
                        'entity_label': entity['label'],
                        'score': entity['score']
                    }
                    unique_entities.append(tmp_entity)
                    seen_entities.add(key)

            predictions[pmid] = unique_entities
            self.articles[pmid]['pred_entities'] = unique_entities
            
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
        
        # Dictionary to hold predicted entities
        # PMID -> {{'start_idx': ..., 'end_idx': ..., 'text_span': ..., 'entity_label': ..., 'score': ...}, ...}
        predictions = {} 

        for pmid, content in self.articles.items():
            title = content['title']
            abstract = content['abstract']

            # Predict title entities 
            title_entities = self.model.predict_entities(title, self.entity_labels, threshold=threshold, flat_ner=flat_ner, multi_label=multi_label)
            
            # Split abstract into sentences
            separator = '. '
            sentences = abstract.split(separator)
            sentence_offsets = []
            offset = len(title) + 1
            if pmid == 562: # TODO: Remove that hardcoded shit and do something clever
                offset += 1
            separator_length = len(separator)
            
            # Define the sentences offsets
            for sentence in sentences:
                sentence_offsets.append(offset)
                offset += len(sentence) + separator_length  # Adjust for separator length

            # Define the variable to hold predicted abstract entities
            abstract_entities = []

            # Predict abstract entities sentence by sentence
            for idx, sentence in enumerate(sentences):
                sentence_entities = self.model.predict_entities(sentence, self.entity_labels, threshold=threshold, flat_ner=flat_ner, multi_label=multi_label)
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
                    tmp_entity = {
                        'start_idx': entity['start'],
                        'end_idx': entity['end'],
                        'tag': 't',
                        'text_span': entity['text'],
                        'entity_label': entity['label'],
                        'score': entity['score']
                    }
                    unique_entities.append(tmp_entity)
                    seen_entities.add(key)

            # Remove duplicates from abstract entities and add tag field with value 'a'
            for entity in abstract_entities:
                key = (entity['start'], entity['end'], entity['text'], entity['label'], entity['score'])
                if key not in seen_entities:
                    tmp_entity = {
                        'start_idx': entity['start'],
                        'end_idx': entity['end'],
                        'tag': 'a',
                        'text_span': entity['text'],
                        'entity_label': entity['label'],
                        'score': entity['score']
                    }
                    unique_entities.append(tmp_entity)
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

            self.articles[pmid]['pred_entities'] = unique_entities
            predictions[pmid] = unique_entities
        
        """
        This part showcase an example of labels aggregation
        """
        if False:
            self.entity_labels = ['GeneorGeneProduct', 'DiseaseOrPhenotypicFeature', 'ChemicalEntity', 'OrganismTaxon']
 
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
        
        # Dictionary to hold predicted entities
        # PMID -> {{'start_idx': ..., 'end_idx': ..., 'text_span': ..., 'entity_label': ..., 'score': ...}, ...}
        predictions = {} 

        for pmid, content in self.articles.items():
            title = content['title']
            abstract = content['abstract']

            # Predict title entities 
            title_entities = self.model.predict_entities(title, self.entity_labels, threshold=threshold, flat_ner=flat_ner, multi_label=multi_label)
            
            # Split abstract into sentences
            separator = '. '
            sentences = abstract.split(separator)
            sentence_offsets = []
            offset = len(title) + 1
            separator_length = len(separator)
            for sentence in sentences:
                sentence_offsets.append(offset)
                offset += len(sentence) + separator_length  # Adjust for separator length

            # Define the variable to hold predicted abstract entities
            abstract_entities = []

            # Predict abstract entities sentence by sentence, one label at a time
            for idx, sentence in enumerate(sentences):
                for label in self.entity_labels:
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
                    tmp_entity = {
                        'start_idx': entity['start'],
                        'end_idx': entity['end'],
                        'tag': 't',
                        'text_span': entity['text'],
                        'entity_label': entity['label'],
                        'score': entity['score']
                    }
                    unique_entities.append(tmp_entity)
                    seen_entities.add(key)

            # Remove duplicates from abstract entities and add tag field with value 'a'
            for entity in abstract_entities:
                key = (entity['start'], entity['end'], entity['text'], entity['label'], entity['score'])
                if key not in seen_entities:
                    tmp_entity = {
                        'start_idx': entity['start'],
                        'end_idx': entity['end'],
                        'tag': 'a',
                        'text_span': entity['text'],
                        'entity_label': entity['label'],
                        'score': entity['score']
                    }
                    unique_entities.append(tmp_entity)
                    seen_entities.add(key)

            self.articles[pmid]['pred_entities'] = unique_entities
            predictions[pmid] = unique_entities
                    
        return predictions


    def compute_metrics(self):
        """
        Computes evaluation metrics for the predictions.
        """
        self.logger.info("Computing evaluation metrics...")

        # Define variables to store predictions counts
        total_predictions = 0  
        predictions_per_label = dict.fromkeys(self.entity_labels, 0)

        # Iterate through the predictions for each article and update the counts
        for pmid in self.articles:
            for ent in self.articles[pmid]['pred_entities']:
                total_predictions += 1
                predictions_per_label[ent['entity_label']] += 1

        # Store the results in self.overall_metrics
        self.overall_metrics['num_articles'] = len(self.articles)
        self.overall_metrics['num_predicted_entities'] = total_predictions
        self.overall_metrics['avg_predicted_entities_per_article'] = round(total_predictions/len(self.articles), 3)
        self.overall_metrics['predictions_per_label'] = {}
        for label in sorted(predictions_per_label, key=predictions_per_label.get):
            self.overall_metrics['predictions_per_label'][label] = predictions_per_label[label]

        # If ground truth is available, compute prediction effectiveness metrics
        if self.has_ground_truth:
            # Define variable to hold all ground truth and predicted entities 
            gt_entities = []
            pred_entities = []
            gt_entities_per_label = dict.fromkeys(self.entity_labels, 0)
            for pmid in self.articles:
                gt_entities.extend(self.articles[pmid]['ground_truth'])
                pred_entities.extend(self.articles[pmid]['pred_entities'])
                for ent in self.articles[pmid]['ground_truth']:
                    gt_entities_per_label[ent['entity_label']] += 1
            
            self.overall_metrics['gt_entities_per_label'] = {}
            for label in sorted(predictions_per_label, key=predictions_per_label.get):
                self.overall_metrics['gt_entities_per_label'][label] = gt_entities_per_label[label]

            # Parse the ground truth entities in a set variable
            gt_set = set((ent['start_idx'], ent['end_idx'], ent['entity_label']) for ent in gt_entities)
            
            # Parse the predicted entities in a set variable
            pred_set = set((ent['start_idx'], ent['end_idx'], ent['entity_label']) for ent in pred_entities)

            # Count ground truth and predicted entities
            self.overall_metrics['num_gt_entities'] = len(gt_set)
            self.overall_metrics['num_pred_entities'] = len(pred_set)

            # Compute positives and negatives
            self.overall_metrics['true_positives'] = len(gt_set & pred_set) 
            self.overall_metrics['false_positives'] = len(pred_set - gt_set)
            self.overall_metrics['false_negatives'] = len(gt_set - pred_set)

            # Compute precision, recall, and F1-score
            self.overall_metrics['precision'] = self.overall_metrics['true_positives'] / (self.overall_metrics['true_positives'] + self.overall_metrics['false_positives'] + 1e-10) # add 1e-10 to avoid division by zero
            self.overall_metrics['recall'] = self.overall_metrics['true_positives'] / (self.overall_metrics['true_positives'] + self.overall_metrics['false_negatives'] + 1e-10)
            self.overall_metrics['f1_score'] = 2 * self.overall_metrics['precision'] * self.overall_metrics['recall'] / (self.overall_metrics['precision'] + self.overall_metrics['recall'] + 1e-10)

        # Print the results in output using the logger
        for key in self.overall_metrics:
            self.logger.info(f'{key}: {self.overall_metrics[key]}')        


    def write_predictions_pubtator(self, output_file_name = "predictions_pubtator.txt"):
        """
        Writes all predicted entities to a single PubTator-formatted file named 'predictions_pubtator.txt'.
        Includes GLiNER parameters if 'include_configuration_in_output' flag is set to True in the configuration file.
        """
        predictions_file_path = os.path.join(self.output_directory, output_file_name)

        with open(predictions_file_path, 'w', encoding='utf-8') as pred_file:
            if self.config['include_configuration_in_output']:
                # Write GLiNER parameters at the top
                pred_file.write("## GLiNER Parameters ##\n")
                pred_file.write(f"Model Used: {self.model_name}\n")
                pred_file.write(f"Threshold: {self.threshold}\n")
                pred_file.write(f"Flat NER: {self.flat_ner}\n")
                pred_file.write(f"Labels: {', '.join(self.entity_labels)}\n")
                pred_file.write(f"Processing Approach: {self.processing}\n")
                # Write separators
                pred_file.write("-"*100 + "\n")

            # Write predicted entities in PubTator format
            for pmid in self.articles:
                title = self.articles[pmid]['title']
                abstract = self.articles[pmid]['abstract']
                pred_entities = self.articles[pmid]['pred_entities']

                # Write title and abstract
                pred_file.write(f"{pmid}|t|{title}\n")
                pred_file.write(f"{pmid}|a|{abstract}\n")

                # Write predicted entities
                for ent in pred_entities:
                    start = ent['start_idx']
                    end = ent['end_idx']
                    tag = ent['tag']
                    text = ent['text_span']
                    entity_label = ent['entity_label']
                    score = ent['score']

                    # Write in PubTator format: PMID \t start_idx \t end_idx \t text_span \t label \t score
                    pred_file.write(f"{pmid}\t{start}\t{end}\t{tag}\t{text}\t{entity_label}\t{score}\n")

                pred_file.write("\n")  # Separate documents by a newline

        self.logger.info(f"All predictions have been successfully written to '/{predictions_file_path}'")


    def write_predictions_pubtator_per_pmid(self):
        """
        Writes predictions for each article in a separated file in PubTator format named '<pmid>_entities.txt'
        """
        # Iterate through the articles
        for pmid in self.articles:
            # Define output path
            predictions_file_path = os.path.join(self.output_directory, f'{pmid}_entities.txt')

            with open(predictions_file_path, 'w', encoding='utf-8') as pred_file: # Create and open file for writing
                title = self.articles[pmid]['title'] # Retrieve title
                abstract = self.articles[pmid]['abstract'] # Retrieve abstract
                pred_entities = self.articles[pmid]['pred_entities'] # Retrieve predicted entities

                # Write title and abstract
                pred_file.write(f"{pmid}|t|{title}\n")
                pred_file.write(f"{pmid}|a|{abstract}\n")

                # Retrieve predicted entities
                for ent in pred_entities:
                    start = ent['start_idx']
                    end = ent['end_idx']
                    tag = ent['tag']
                    text = ent['text_span']
                    entity_label = ent['entity_label']
                    score = ent['score']

                    # Write predicted entities in PubTator format: PMID \t start_idx \t end_idx \t text_span \t label \t score
                    pred_file.write(f"{pmid}\t{start}\t{end}\t{tag}\t{text}\t{entity_label}\t{score}\n")

        self.logger.info(f"Predictions for each article have been successfully written separated to '/{self.output_directory}'")


    def write_comparison_gt_predictions_pubtator(self, output_file_name = "gt_pred_comparison.txt"):
        """
        Writes the comparison between ground truth and predicted entities in a file in PubTator format
        The caller to this function should check before if self.has_ground_truth is True
        """
        file_path = os.path.join(self.output_directory, output_file_name)

        def align_entities(gt_entities, pred_entities):
            """
            Aligns ground truth and predicted entities based on overlap

            Parameters:
            gt_entities (list of tuples): the ground truth entities for a certain article
            pred_entities (list of tuples): the predicted entities for a certain article 

            Returns:
            list of tuples: a list of tuples (gt_ent, pred_ent)
            """
            aligned_pairs = []
            matched_gt = set()
            matched_pred = set()

            def spans_overlap(start1, end1, start2, end2):
                """
                Checks if two text spans overlap
                """
                return max(start1, start2) < min(end1, end2)

            # Find overlapping entities
            for i, gt_ent in enumerate(gt_entities):
                match_found = False
                for j, pred_ent in enumerate(pred_entities):
                    if j in matched_pred:
                        continue
                    if spans_overlap(gt_ent['start_idx'], gt_ent['end_idx'], pred_ent['start_idx'], pred_ent['end_idx']):
                        aligned_pairs.append((gt_ent, pred_ent))
                        matched_gt.add(i)
                        matched_pred.add(j)
                        break

            # Collect unmatched ground truth entities 
            unmatched_gt = [(gt_entities[i], None) for i in range(len(gt_entities)) if i not in matched_gt]

            # Collect unmatched predicted entities
            unmatched_pred = [(None, pred_entities[j]) for j in range(len(pred_entities)) if j not in matched_pred]

            # Combine all pairs 
            all_pairs = aligned_pairs + unmatched_gt + unmatched_pred

            # Sort all pairs based on the start index of the entities
            def get_start_index(pair):
                starts = []
                if pair[0]:
                    starts.append(pair[0]['start_idx'])
                if pair[1]:
                    starts.append(pair[1]['start_idx'])
                return min(starts) if starts else 0 # Should not happen

            all_pairs.sort(key=get_start_index)

            return all_pairs
        
        with open(file_path, 'w', encoding='utf-8') as f:
            for pmid in self.articles:
                title = self.articles[pmid]['title']
                abstract = self.articles[pmid]['abstract']

                gt_entities = self.articles[pmid]['ground_truth']
                pred_entities = self.articles[pmid]['pred_entities']

                aligned_tuples = align_entities(gt_entities, pred_entities)

                # Write title and abstract
                f.write(f'{pmid}|t|{title}\n')
                f.write(f'{pmid}|a|{abstract}\n')  

                # Write entities
                for gt_ent, pred_ent in aligned_tuples:
                    if gt_ent and pred_ent:
                        # Scenario 1: Overlapping entities
                        # Ground truth entity
                        gt_line = "\t".join([
                            "grtr",
                            str(gt_ent['start_idx']),
                            str(gt_ent['end_idx']),
                            gt_ent['text_span'],
                            gt_ent['entity_label'],
                        ])
                        # Predicted entity
                        pred_line = "\t".join([
                            "pred",
                            str(pred_ent['start_idx']),
                            str(pred_ent['end_idx']),
                            pred_ent['text_span'],
                            pred_ent['entity_label'],
                            str(pred_ent['score'])
                        ])
                    elif gt_ent and not pred_ent:
                        # Scenario 2: Ground truth entity not predicted
                        gt_line = "\t".join([
                            "grtr",
                            str(gt_ent['start_idx']),
                            str(gt_ent['end_idx']),
                            gt_ent['text_span'],
                            gt_ent['entity_label'],
                        ])
                        pred_line = "\t".join([
                            "pred",
                            "——",
                            "——",
                            "——",
                            "——",
                            "——"
                        ])
                    elif not gt_ent and pred_ent:
                        # Scenario 3: Predicted entity not in ground truth
                        gt_line = "\t".join([
                            "grtr",
                            "——",
                            "——",
                            "——",
                            "——",
                            "——"
                        ])
                        pred_line = "\t".join([
                            "pred",
                            str(pred_ent['start_idx']),
                            str(pred_ent['end_idx']),
                            pred_ent['text_span'],
                            pred_ent['entity_label'],
                            str(pred_ent['score'])
                        ])
                    else:
                        # Should not occur
                        continue

                    # Write the lines
                    f.write(f"{gt_line}\n")
                    f.write(f"{pred_line}\n\n") # Add an empty line between entities

                f.write('\n\n') # Add two empty lines when switching articles 


    def store_to_json(self, output_file_name='gliner_predictions.json'):
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
        }
        """
        # Convert any non-serializable data if necessary
        def default_serializer(obj):
            if isinstance(obj, set):
                return list(obj)
            # Add other types if needed
            raise TypeError(f'Type {type(obj)} not serializable')

        output_file_path = os.path.join(self.output_directory, output_file_name)
        with open(output_file_path, 'w', encoding='utf-8') as f:
            json.dump(self.articles, f, ensure_ascii=False, indent=4, default=default_serializer)

        self.logger.info(f"Predictions have been exported in JSON format to '/{output_file_path}'")


    def save_results(self):
        """
        Saves the ground truth and predicted entities to files.
        """
        self.logger.info("Saving results to files...")
        self.write_predictions_pubtator(output_file_name = self.config['pubtator_aggregated_predictions_file_name'])
        self.write_predictions_pubtator_per_pmid()
        self.store_to_json(output_file_name = self.config['json_aggregated_predictions_file_name'])
        if self.has_ground_truth:
            self.write_comparison_gt_predictions_pubtator(output_file_name = self.config['pubtator_comparison_ground_truth_predicted_file_name'])


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
#                            help='Minimum number of ground truth entities per document')
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