# helper.py
"""
This file contains helper functions and classes
"""

import yaml
import json
import os

def print_structure(data, indent=0):
    """
    Recursively prints all fields and subfields of the given data.

    Parameters:
    data (dict): The dictionary struccture to print.
    indent (int): The extra number of spaces to use for indentation (default: 1).
    """
    if isinstance(data, dict):
        for key, value in data.items():
            print("  " * indent + f"Key: {key}")
            print_structure(value, indent + 1)
    elif isinstance(data, list):
        for i, item in enumerate(data):
            print("  " * indent + f"Item {i}:")
            print_structure(item, indent + 1)
    else:
        print("  " * indent + f"Value: {data}")


def load_config(config_path='config/config.yaml'):
    """
    Load the YAML configuration file passed as argument.

    Parameters:
    config_path (str): The path to the YAML configuration file to load.

    Returns:
    dict: A dictionary containing the loaded configuration parameters. 
    """
    with open(config_path, 'r') as file:
        return yaml.safe_load(file)
    
def load_articles_from_json(filename):
    """
    Loads articles from a JSON file into a dictionary variable.
    Includes error handling for file I/O and JSON decoding
    
    Parameters:
    filename (str): The path to the JSON file containing the articles.

    Returns:
    dict: A dictionary containing the articles loaded from the JSON file.
    """
    if not os.path.exists(filename):
        raise FileNotFoundError(f'The file {filename} does not exist')

    try:
        with open(filename, 'r', encoding='utf-8') as f:
            articles = json.load(f)
    except json.JSONDecodeError as e:
        raise ValueError(f'Error decoding JSON from file {filename}: {e}')
    
    return articles

import json

def remove_duplicate_relations(input_json_path, output_json_path):
    # Load JSON file into a dictionary
    with open(input_json_path, 'r', encoding="utf-8") as f:
        data = json.load(f)

    total_initial_relations = 0
    total_duplicate_relations = 0
    total_removed_relations = 0
    total_final_relations = 0
    num_documents_processed = 0

    for pmid, doc in data.items():
        num_documents_processed += 1
        relations = doc.get('pred_relations', [])
        total_initial_relations += len(relations)

        unique_relations = []
        seen_relations = set()
        duplicate_count = 0

        for relation in relations:
            # Create a key to identify unique relations
            head_key = (relation['head_start_idx'], relation['head_end_idx'], relation['tail_start_idx'], relation['tail_end_idx'])
            tail_key = (relation['tail_start_idx'], relation['tail_end_idx'], relation['head_start_idx'], relation['head_end_idx'])

            if head_key in seen_relations or tail_key in seen_relations:
                duplicate_count += 1
            else:
                seen_relations.add(head_key)
                unique_relations.append(relation)

        total_duplicate_relations += duplicate_count
        total_removed_relations += duplicate_count
        total_final_relations += len(unique_relations)

        # Update document with unique relations
        doc['pred_relations'] = unique_relations

        if False:
            # Print document statistics
            print(f"$ Document {pmid} processed")
            print(f"$ Num of initial relations: {len(relations)}")
            print(f"$ Num of duplicate relations: {duplicate_count}")
            print(f"$ Num of removed relations: {duplicate_count}")
            print(f"$ Num of final relations: {len(unique_relations)}")

    # Save the updated data to output JSON file
    with open(output_json_path, 'w', encoding="utf-8") as f:
        json.dump(data, f, indent=4)

    # Calculate average number of relations per document
    avg_relations_per_document = total_final_relations / num_documents_processed if num_documents_processed > 0 else 0

    # Print overall statistics
    print(f"$ Num of documents processed: {num_documents_processed}")
    print(f"$ Total number of initial relations: {total_initial_relations}")
    print(f"$ Total number of duplicate relations: {total_duplicate_relations}")
    print(f"$ Total number of removed relations: {total_removed_relations}")
    print(f"$ Total number of final relations: {total_final_relations}")
    print(f"$ Average number of relations per document: {avg_relations_per_document:.2f}")

def remove_unknown_relations(input_json_path, output_json_path):
    # Load JSON file into a dictionary
    with open(input_json_path, 'r', encoding="utf-8") as f:
        data = json.load(f)

    total_initial_relations = 0
    total_unknown_relations = 0
    total_final_relations = 0
    num_documents_processed = 0

    for pmid, doc in data.items():
        num_documents_processed += 1
        relations = doc.get('pred_relations', [])
        total_initial_relations += len(relations)

        filtered_relations = [
            relation for relation in relations
            if relation['head_entity_label'] != 'unknown' and relation['tail_entity_label'] != 'unknown'
        ]

        num_removed_relations = len(relations) - len(filtered_relations)
        total_unknown_relations += num_removed_relations
        total_final_relations += len(filtered_relations)

        # Update document with filtered relations
        doc['pred_relations'] = filtered_relations

    # Save the updated data to output JSON file
    with open(output_json_path, 'w', encoding="utf-8") as f:
        json.dump(data, f, indent=4)

    # Calculate average number of relations per document
    avg_relations_per_document = total_final_relations / num_documents_processed if num_documents_processed > 0 else 0

    # Print overall statistics
    print(f"$ Num of documents processed: {num_documents_processed}")
    print(f"$ Total number of initial relations: {total_initial_relations}")
    print(f"$ Total number of removed relations (relations with 'unknown' head or tail): {total_unknown_relations}")
    print(f"$ Total number of final relations: {total_final_relations}")
    print(f"$ Average number of relations per document: {avg_relations_per_document:.2f}")

def remove_text_span_duplicates(input_json_path, output_json_path):
    # Load JSON file into a dictionary
    with open(input_json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    total_initial_relations = 0
    total_removed_relations = 0
    total_final_relations = 0
    num_documents_processed = 0

    for pmid, doc in data.items():
        num_documents_processed += 1
        relations = doc.get('pred_relations', [])
        total_initial_relations += len(relations)

        seen_relations = set()
        filtered_relations = []

        for relation in relations:
            head_text_span = relation['head_text_span'].lower()
            tail_text_span = relation['tail_text_span'].lower()
            #relation_key = (head_text_span, tail_text_span, relation['relation_label'])
            relation_key1 = (head_text_span, tail_text_span)
            relation_key2 = (tail_text_span, head_text_span)

            if relation_key1 not in seen_relations and relation_key2 not in seen_relations:
                seen_relations.add(relation_key1)
                filtered_relations.append(relation)
            else:
                total_removed_relations += 1

        total_final_relations += len(filtered_relations)

        # Update document with filtered relations
        doc['pred_relations'] = filtered_relations

    # Save the updated data to output JSON file
    with open(output_json_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=4)

    # Calculate average number of relations per document
    avg_relations_per_document = total_final_relations / num_documents_processed if num_documents_processed > 0 else 0

    # Print overall statistics
    print(f"$ Num of documents processed: {num_documents_processed}")
    print(f"$ Total number of initial relations: {total_initial_relations}")
    print(f"$ Total number of removed relations (text span duplicates): {total_removed_relations}")
    print(f"$ Total number of final relations: {total_final_relations}")
    print(f"$ Average number of relations per document: {avg_relations_per_document:.2f}")

def spans_overlap(start1, end1, start2, end2):
    """
    Checks if two text spans overlap
    """
    return max(start1, start2) < min(end1, end2)

def find_overlapping_entities(input_json_path, output_txt_path):
    # Load JSON file into a dictionary
    with open(input_json_path, 'r') as f:
        data = json.load(f)

    with open(output_txt_path, 'w') as f:
        for pmid, doc in data.items():
            entities = doc.get('pred_entities', [])
            overlapping_groups = []
            seen = set()

            # Find overlapping entities
            for i, entity1 in enumerate(entities):
                if i in seen:
                    continue
                overlapping = [entity1]
                for j, entity2 in enumerate(entities[i + 1:], start=i + 1):
                    if j in seen:
                        continue
                    if spans_overlap(entity1['start_idx'], entity1['end_idx'], entity2['start_idx'], entity2['end_idx']):
                        overlapping.append(entity2)
                        seen.add(j)
                if len(overlapping) > 1:
                    overlapping_groups.append(overlapping)
                    seen.add(i)

            # Write to output file
            if overlapping_groups:
                f.write(f"PMID: {pmid}\n")
                f.write(f"Title: {doc.get('title', '')}\n")
                f.write(f"Abstract: {doc.get('abstract', '')}\n")
                for idx, group in enumerate(overlapping_groups, start=1):
                    f.write(f"Overlapping {idx}:\n")
                    for entity in group:
                        f.write(f"{entity['start_idx']}\t{entity['end_idx']}\t{entity['text_span']}\t{entity['entity_label']}\n")
                f.write("\n")

def parse_annotation_results(input_json_path, metadata_json_path, output_json_path):
    # Load JSON file into a dictionary
    with open(input_json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    with open(metadata_json_path, 'r', encoding='utf-8') as f:
        metadata = json.load(f)

    result = {}

    # Process tags and relationships
    tags = data.get('tags', [])
    relationships = data.get('relationships', [])

    # Process entities (tags)
    for tag in tags:
        pmid = tag['document_id']
        if pmid not in result:
            if pmid in metadata:
                meta_info = metadata[pmid]
            else:
                meta_info = None #TODO: implement PubMed fetch here

            result[pmid] = {
                #'annotator': None,
                'title': meta_info.get('title', ''),
                'author': meta_info.get('author', ''),
                'journal': meta_info.get('journal', ''),
                'year': meta_info.get('year', ''),
                'abstract': meta_info.get('abstract', ''),
                'entities': [],
                'relations': [],
                'binary_tag_based_relations': [],
                'ternary_tag_based': [],
                'ternary_mention_based': []
            }      

        result[pmid]['entities'].append({
            'annotator': tag['username'],
            'start': tag['start'],
            'end': tag['stop'],
            'mention_location': tag['mention_location'],
            'mention_text': tag['mention_text'],
            'mention_label': tag['tag']
        })

    # Process relationships
    for relation in relationships:
        pmid = relation['document_id']
        if pmid not in result:
            if pmid in metadata:
                meta_info = metadata[pmid]
            else:
                meta_info = None #TODO: implement PubMed fetch here
                
            result[pmid] = {
                'annotator': None,
                'title': meta_info.get('title', ''),
                'author': meta_info.get('author', ''),
                'journal': meta_info.get('journal', ''),
                'year': meta_info.get('year', ''),
                'abstract': meta_info.get('abstract', ''),
                'entities': [],
                'relations': [],
                'binary_tag_based_relations': [],
                'ternary_tag_based': [],
                'ternary_mention_based': []
            }

        result[pmid]['relations'].append({
            'annotator': relation['username'],
            'subject_start': relation['subject_start'],
            'subject_end': relation['subject_stop'],
            'subject_mention_location': relation['subject_mention_location'],
            'subject_mention_text': relation['subject_mention_text'],
            'subject_mention_label': relation['subject_tags'][0]['tag'] if relation['subject_tags'] else None,
            'predicate': relation['predicate_tags'][0]['concept_name'] if relation['predicate_tags'] else None,
            'object_start': relation['object_start'],
            'object_end': relation['object_stop'],
            'object_mention_location': relation['object_mention_location'],
            'object_mention_text': relation['object_mention_text'],
            'object_mention_label': relation['object_tags'][0]['tag'] if relation['object_tags'] else None
        }) 

    # Save the parsed data to output JSON file
    with open(output_json_path, 'w', encoding='utf-8') as f:
        json.dump(result, f, indent=4)

# Define the legal relations from the provided table
LEGAL_RELATIONS = {
    ("anatomical location", "human"): ["located in"],
    ("anatomical location", "animal"): ["located in"],
    ("bacteria", "microbiome"): ["part of"],
    ("bacteria", "disease"): ["influence"],
    ("disease", "bacteria"): ["change abundance"],
    ("disease", "human"): ["affect"],
    ("disease", "animal"): ["affect"],
    ("disease", "microbiome"): ["change abundance"],
    ("drug", "disease"): ["change effect"],
    ("chemical", "disease"): ["change effect"],
    ("dietary supplement", "disease"): ["change effect"],
    ("drug", "microbiome"): ["impact"],
    ("chemical", "microbiome"): ["impact"],
    ("dietary supplement", "microbiome"): ["impact"],
    ("human", "assay"): ["used by"],
    ("animal", "assay"): ["used by"],
    ("metabolite", "microbiome"): ["produced by"],
    ("metabolite", "anatomical location"): ["located in"],
    ("microbiome", "assay"): ["used by"],
    ("microbiome", "human"): ["located in"],
    ("microbiome", "animal"): ["located in"],
    ("microbiome", "gene"): ["change expression"],
    ("microbiome", "disease"): ["is linked to"],
    ("microbiome", "microbiome"): ["compared to"],
    ("neurotransmitter", "microbiome"): ["related to"],
    ("neurotransmitter", "anatomical location"): ["located in"]
}

def extract_relations(input_json_path, output_json_path):
    # Load parsed JSON file into a dictionary
    with open(input_json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    output_data = {}

    for document_id, content in data.items():
        relations = content.get('relations', [])
        binary_untagged_count = {}
        binary_tagged_count = {}
        ternary_tagged_count = {}

        for relation in relations:
            subject_tag = relation['subject_tag'].lower() if relation['subject_tag'] else None
            object_tag = relation['object_tag'].lower() if relation['object_tag'] else None
            predicate = relation['predicate'].lower() if relation['predicate'] else None
            subject_mention_text = relation['subject_mention_text']
            object_mention_text = relation['object_mention_text']

            if subject_tag and object_tag:
                # Check if the relation is legal
                subject_object_pair = (subject_tag, object_tag)
                #subject_object_pair_reversed = (object_tag, subject_tag)
                #valid_predicates = LEGAL_RELATIONS.get(subject_object_pair, []) + LEGAL_RELATIONS.get(subject_object_pair_reversed, [])
                valid_predicates = LEGAL_RELATIONS.get(subject_object_pair, [])

                if predicate in valid_predicates:
                    # Create binary untagged and tagged relations
                    binary_untagged = (subject_tag, object_tag)
                    binary_tagged = (subject_tag, predicate, object_tag)
                    ternary_tagged = (subject_mention_text, predicate, object_mention_text)

                    # Count occurrences
                    if binary_untagged in binary_untagged_count:
                        binary_untagged_count[binary_untagged] += 1
                    else:
                        binary_untagged_count[binary_untagged] = 1

                    if binary_tagged in binary_tagged_count:
                        binary_tagged_count[binary_tagged] += 1
                    else:
                        binary_tagged_count[binary_tagged] = 1

                    if ternary_tagged in ternary_tagged_count:
                        ternary_tagged_count[ternary_tagged] += 1
                    else:
                        ternary_tagged_count[ternary_tagged] = 1

        # Prepare output for the current document
        output_data[document_id] = {
            "binary_untagged": [list(relation) + [count] for relation, count in binary_untagged_count.items()],
            "binary_tagged": [list(relation) + [count] for relation, count in binary_tagged_count.items()],
            "ternary_tagged": [list(relation) + [count] for relation, count in ternary_tagged_count.items()]
        }

    # Save the output data to JSON file
    with open(output_json_path, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=4)


if __name__ == "__main__":
    if False:
        input_json_path = 'predicted_relations/new_predicted_relations'
        output_json_path = 'predicted_relations/new_predicted_relations_no_duplicates'
        #output_json_path = 'predicted_relations/new_predicted_relations_no_duplicates_no_unknown_relations_single_text_spans.json'
        print(f'input json = {input_json_path}')
        print(f'output json = {output_json_path}')

        print('-'*100)
        print('## REMOVE DUPLICATE RELATIONS ##')
        remove_duplicate_relations(f'{input_json_path}.json', f'{output_json_path}.json')
        print('-'*100)
        
        print('## REMOVE UNKNOWN RELATIONS ##')
        remove_unknown_relations(f'{output_json_path}.json', f'{output_json_path}_no_unknown_relations.json')
        print('-'*100)

        print('## REMOVE RELATIONS WITH DUPLICATE TEXT SPANS ##')
        remove_text_span_duplicates(f'{output_json_path}_no_unknown_relations.json', f'{output_json_path}_no_unknown_relations_single_text_spans.json')
        print('-'*100)

    if False:
        input_json_path = 'predicted_relations/new_predicted_relations_no_duplicates_no_unknown_relations.json'
        output_txt_path = 'predicted_entities/overlapping_entities.txt'
        find_overlapping_entities(input_json_path, output_txt_path)

    if True:
        annotation_output_json_path = 'parsed_annotations.json'
        parse_annotation_results('retrieved_articles/pubmed_31952911.json', 'predicted_relations/old_predicted_relations.json', 'retrieved_articles/parsed_annotations.json')

    if False:
        extract_relations('retrieved_articles/parsed_annotations.json', 'retrieved_articles/parsed_relations.json')

"""
RegEx to match entire line having certain keywords
^.*(word1|word2|word3).*\n?
"""