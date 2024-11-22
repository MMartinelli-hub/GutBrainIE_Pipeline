import json
import re

# Define the set of legal entities
LEGAL_ENTITIES = {
    "anatomical location",
    "animal",
    "assay",
    "bacteria",
    "chemical",
    "dietary supplement",
    "disease",
    "drug",
    "gene",
    "human",
    "metabolite",
    "microbiome",
    "neurotransmitter"
}

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

# Function to parse the MetaTron annotations to Ground Truth format
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
        pmid = tag['document_id'].strip('pubmed_')
        mention_label = tag['tag'].lower()
        if mention_label not in LEGAL_ENTITIES:
            continue

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
                'ternary_tag_based_relations': [],
                'ternary_mention_based_relations': []
            }      

        result[pmid]['entities'].append({
            'annotator': tag['username'],
            'start': tag['start'],
            'end': tag['stop'],
            'mention_location': tag['mention_location'],
            'mention_text': tag['mention_text'],
            'mention_label': mention_label
        })

    # Process relationships
    for relation in relationships:
        pmid = relation['document_id'].strip('pubmed_')
        subject_tag = relation['subject_tags'][0]['tag'].lower() if relation['subject_tags'] else None
        object_tag = relation['object_tags'][0]['tag'].lower() if relation['object_tags'] else None

        if subject_tag not in LEGAL_ENTITIES or object_tag not in LEGAL_ENTITIES:
            continue

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
                'ternary_tag_based_relations': [],
                'ternary_mention_based_relations': []
            }

        result[pmid]['relations'].append({
            'annotator': relation['username'],
            'subject_start': relation['subject_start'],
            'subject_end': relation['subject_stop'],
            'subject_mention_location': relation['subject_mention_location'],
            'subject_mention_text': relation['subject_mention_text'],
            'subject_mention_label': subject_tag,
            'predicate': relation['predicate_tags'][0]['concept_name'] if relation['predicate_tags'] else None,
            'object_start': relation['object_start'],
            'object_end': relation['object_stop'],
            'object_mention_location': relation['object_mention_location'],
            'object_mention_text': relation['object_mention_text'],
            'object_mention_label': object_tag
        }) 

    # Extract binary and ternary relations
    for pmid, content in result.items():
        relations = content.get('relations', [])
        binary_tagged_count = {}
        ternary_tagged_count = {}
        ternary_mention_based_count = {}

        for relation in relations:
            subject_tag = relation['subject_mention_label'].lower() if relation['subject_mention_label'] else None
            object_tag = relation['object_mention_label'].lower() if relation['object_mention_label'] else None
            predicate = relation['predicate'].lower() if relation['predicate'] else None
            subject_mention_text = relation['subject_mention_text']
            object_mention_text = relation['object_mention_text']
            subject_start = relation['subject_start']
            subject_end = relation['subject_end']
            object_start = relation['object_start']
            object_end = relation['object_end']
            subject_mention_location = relation['subject_mention_location']
            object_mention_location = relation['object_mention_location']

            if subject_tag and object_tag:
                # Check if the relation is legal
                subject_object_pair = (subject_tag, object_tag)
                valid_predicates = LEGAL_RELATIONS.get(subject_object_pair, [])

                if predicate in valid_predicates:
                    binary_tagged = (subject_tag, object_tag)
                    ternary_tagged = (subject_tag, predicate, object_tag)
                    ternary_mention_based = {
                        "subject_start": subject_start,
                        "subject_end": subject_end,
                        "subject_mention_text": subject_mention_text,
                        "subject_mention_label": subject_tag,
                        "subject_mention_location": subject_mention_location,
                        "predicate": predicate,
                        "object_start": object_start,
                        "object_end": object_end,
                        "object_mention_text": object_mention_text,
                        "object_mention_label": object_tag,
                        "object_mention_location": object_mention_location
                    }

                    # Count occurrences
                    if binary_tagged in binary_tagged_count:
                        binary_tagged_count[binary_tagged] += 1
                    else:
                        binary_tagged_count[binary_tagged] = 1

                    if ternary_tagged in ternary_tagged_count:
                        ternary_tagged_count[ternary_tagged] += 1
                    else:
                        ternary_tagged_count[ternary_tagged] = 1

                    ternary_mention_based_key = tuple(ternary_mention_based.items())
                    if ternary_mention_based_key in ternary_mention_based_count:
                        ternary_mention_based_count[ternary_mention_based_key] += 1
                    else:
                        ternary_mention_based_count[ternary_mention_based_key] = 1

        # Prepare binary and ternary relation outputs
        content["binary_tag_based_relations"] = [
            {
                "subject_mention_label": relation[0],
                "object_mention_label": relation[1],
                "count": count
            }
            for relation, count in binary_tagged_count.items()
        ]

        content["ternary_tag_based_relations"] = [
            {
                "subject_mention_label": relation[0],
                "predicate": relation[1],
                "object_mention_label": relation[2],
                "count": count
            }
            for relation, count in ternary_tagged_count.items()
        ]

        content["ternary_mention_based_relations"] = [
            dict(ternary) | {"count": count}
            for ternary, count in ternary_mention_based_count.items()
        ]

    # Save the output data to JSON file
    with open(output_json_path, 'w', encoding='utf-8') as f:
        json.dump(result, f, indent=4)

# Function to parse the MetaTron annotations to Ground Truth format
def parse_and_sort_annotation_results(input_json_path, metadata_json_path, output_json_path):
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
        pmid = tag['document_id'].strip('pubmed_')
        mention_label = tag['tag'].lower()
        if mention_label not in LEGAL_ENTITIES:
            continue

        if pmid not in result:
            if pmid in metadata:
                meta_info = metadata[pmid]
            else:
                meta_info = None  # TODO: implement PubMed fetch here

            result[pmid] = {
                # 'annotator': None,
                'title': meta_info.get('title', ''),
                'author': meta_info.get('author', ''),
                'journal': meta_info.get('journal', ''),
                'year': meta_info.get('year', ''),
                'abstract': meta_info.get('abstract', ''),
                'entities': [],
                'relations': [],
                'binary_tag_based_relations': [],
                'ternary_tag_based_relations': [],
                'ternary_mention_based_relations': []
            }

        result[pmid]['entities'].append({
            'annotator': tag['username'],
            'start': tag['start'],
            'end': tag['stop'],
            'mention_location': tag['mention_location'],
            'mention_text': tag['mention_text'],
            'mention_label': mention_label
        })

    # *** Modification: Separate and sort entities by mention_location and start index ***
    for pmid, doc_data in result.items():
        title_entities = []
        abstract_entities = []
        other_entities = []

        # Separate entities based on their mention_location
        for entity in doc_data['entities']:
            mention_location = entity.get('mention_location', '')
            if 'title' in mention_location:
                title_entities.append(entity)
            elif 'abstract' in mention_location:
                abstract_entities.append(entity)
            else:
                other_entities.append(entity)

        # Sort entities within each group by 'start' index
        title_entities_sorted = sorted(title_entities, key=lambda x: x['start'])
        abstract_entities_sorted = sorted(abstract_entities, key=lambda x: x['start'])
        other_entities_sorted = sorted(other_entities, key=lambda x: x['start'])

        # Concatenate the lists: title entities first, then abstract entities, then others
        doc_data['entities'] = title_entities_sorted + abstract_entities_sorted + other_entities_sorted

    # Process relationships
    for relation in relationships:
        pmid = relation['document_id'].strip('pubmed_')
        subject_tag = relation['subject_tags'][0]['tag'].lower() if relation['subject_tags'] else None
        object_tag = relation['object_tags'][0]['tag'].lower() if relation['object_tags'] else None

        if subject_tag not in LEGAL_ENTITIES or object_tag not in LEGAL_ENTITIES:
            continue

        if pmid not in result:
            if pmid in metadata:
                meta_info = metadata[pmid]
            else:
                meta_info = None  # TODO: implement PubMed fetch here

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
                'ternary_tag_based_relations': [],
                'ternary_mention_based_relations': []
            }

        result[pmid]['relations'].append({
            'annotator': relation['username'],
            'subject_start': relation['subject_start'],
            'subject_end': relation['subject_stop'],
            'subject_mention_location': relation['subject_mention_location'],
            'subject_mention_text': relation['subject_mention_text'],
            'subject_mention_label': subject_tag,
            'predicate': relation['predicate_tags'][0]['concept_name'] if relation['predicate_tags'] else None,
            'object_start': relation['object_start'],
            'object_end': relation['object_stop'],
            'object_mention_location': relation['object_mention_location'],
            'object_mention_text': relation['object_mention_text'],
            'object_mention_label': object_tag
        })

    # Extract binary and ternary relations
    for pmid, content in result.items():
        relations = content.get('relations', [])
        binary_tagged_count = {}
        ternary_tagged_count = {}
        ternary_mention_based_count = {}

        for relation in relations:
            subject_tag = relation['subject_mention_label'].lower() if relation['subject_mention_label'] else None
            object_tag = relation['object_mention_label'].lower() if relation['object_mention_label'] else None
            predicate = relation['predicate'].lower() if relation['predicate'] else None
            subject_mention_text = relation['subject_mention_text']
            object_mention_text = relation['object_mention_text']
            subject_start = relation['subject_start']
            subject_end = relation['subject_end']
            object_start = relation['object_start']
            object_end = relation['object_end']
            subject_mention_location = relation['subject_mention_location']
            object_mention_location = relation['object_mention_location']

            if subject_tag and object_tag:
                # Check if the relation is legal
                subject_object_pair = (subject_tag, object_tag)
                valid_predicates = LEGAL_RELATIONS.get(subject_object_pair, [])

                if predicate in valid_predicates:
                    binary_tagged = (subject_tag, object_tag)
                    ternary_tagged = (subject_tag, predicate, object_tag)
                    ternary_mention_based = {
                        "subject_start": subject_start,
                        "subject_end": subject_end,
                        "subject_mention_text": subject_mention_text,
                        "subject_mention_label": subject_tag,
                        "subject_mention_location": subject_mention_location,
                        "predicate": predicate,
                        "object_start": object_start,
                        "object_end": object_end,
                        "object_mention_text": object_mention_text,
                        "object_mention_label": object_tag,
                        "object_mention_location": object_mention_location
                    }

                    # Count occurrences
                    ternary_mention_based_key = tuple(ternary_mention_based.items())
                    binary_tagged_count[binary_tagged] = binary_tagged_count.get(binary_tagged, 0) + 1
                    ternary_tagged_count[ternary_tagged] = ternary_tagged_count.get(ternary_tagged, 0) + 1
                    ternary_mention_based_count[ternary_mention_based_key] = ternary_mention_based_count.get(ternary_mention_based_key, 0) + 1

        # Prepare binary and ternary relation outputs
        content["binary_tag_based_relations"] = [
            {
                "subject_mention_label": relation[0],
                "object_mention_label": relation[1],
                "count": count
            }
            for relation, count in binary_tagged_count.items()
        ]

        content["ternary_tag_based_relations"] = [
            {
                "subject_mention_label": relation[0],
                "predicate": relation[1],
                "object_mention_label": relation[2],
                "count": count
            }
            for relation, count in ternary_tagged_count.items()
        ]

        content["ternary_mention_based_relations"] = [
            dict(ternary) | {"count": count}
            for ternary, count in ternary_mention_based_count.items()
        ]

    # Save the output data to JSON file
    with open(output_json_path, 'w', encoding='utf-8') as f:
        json.dump(result, f, indent=4)

def tokenize_text_with_positions(text):
    # Split text into tokens, preserving punctuation (except for hyphens and underscores)
    tokens = []
    token_spans = []  # list of (start_char_index, end_char_index) for each token
    pattern = re.compile(r"\w+|[.,!?;:\'\"()\[\]{}<>]|[\s]+|\S")
    for match in pattern.finditer(text):
        token = match.group()
        if token.isspace():
            continue  # Skip whitespace tokens
        start_pos = match.start()
        if re.match(r"\w+-\w+", token) or re.match(r"\w+_\w+", token):
            # Keep hyphenated or underscored words intact
            tokens.append(token)
            token_spans.append((start_pos, match.end()))
        else:
            # Split contractions (e.g., "don't" -> "don", "'", "t")
            contraction_match = re.match(r"(\w+)(')(\w+)", token)
            if contraction_match:
                groups = contraction_match.groups()
                for group in groups:
                    end_pos = start_pos + len(group)
                    tokens.append(group)
                    token_spans.append((start_pos, end_pos))
                    start_pos = end_pos
            else:
                tokens.append(token)
                token_spans.append((start_pos, match.end()))
    return tokens, token_spans

def process_ground_truth(ground_truth_path, output_path):
    # Read the ground truth data
    with open(ground_truth_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    output_data = []

    for doc_id, doc_data in data.items():
        overall_tokenized_text = []
        overall_ner = []
        token_offset = 0

        fields = ["title", "abstract"]

        for field in fields:
            text = doc_data.get(field, "")
            tokens, token_spans = tokenize_text_with_positions(text)

            # Collect entities for this field
            field_entities = []
            for entity in doc_data.get("entities", []):
                mention_location = entity.get("mention_location", "")
                if mention_location == field + "_value":
                    field_entities.append(entity)

            # Map entities from character indices to token indices
            for entity in field_entities:
                entity_start_char = entity["start"]
                entity_end_char = entity["end"] + 1  # Adjusting end index to be exclusive
                entity_label = entity["mention_label"]

                entity_start_token_index = None
                entity_end_token_index = None

                for i, (token_start_char, token_end_char) in enumerate(token_spans):
                    if token_end_char <= entity_start_char:
                        continue  # Token is before the entity
                    if token_start_char >= entity_end_char:
                        break  # Token is after the entity
                    # Token overlaps with entity
                    if entity_start_token_index is None:
                        entity_start_token_index = i
                    entity_end_token_index = i  # Update to the last overlapping token

                if entity_start_token_index is not None and entity_end_token_index is not None:
                    overall_ner.append([
                        entity_start_token_index + token_offset,
                        entity_end_token_index + token_offset,
                        entity_label.lower()
                    ])
                else:
                    print(f"Warning: Could not find tokens for entity in doc {doc_id}, field {field}")

            # Append tokens to the overall tokenized text
            overall_tokenized_text.extend(tokens)
            token_offset += len(tokens)

        # Sort the word positions by the start index
        overall_ner.sort(key=lambda x: x[0])

        # Create the output dictionary for this document
        output_doc = {
            "tokenized_text": overall_tokenized_text,
            "ner": overall_ner
        }

        output_data.append(output_doc)

    # Write the output data to a JSON file
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=4)

# Function to evaluate NER against the ground truth
def evaluate_ner(ground_truth_json_path, predictions_json_path):
    # Load ground truth and predictions JSON files into dictionaries
    with open(ground_truth_json_path, 'r', encoding='utf-8') as f:
        ground_truth_data = json.load(f)

    with open(predictions_json_path, 'r', encoding='utf-8') as f:
        predictions_data = json.load(f)
    
    gt_entities = []
    pred_entities = []

    # Process each document separately
    for pmid in predictions_data:
        gt_doc_entities = ground_truth_data.get(pmid, {}).get('entities', [])
        pred_doc_entities = predictions_data.get(pmid, {}).get('pred_entities', [])

        # Add document-specific information to distinguish entities from different documents
        gt_entities.extend((pmid, ent['start'], ent['end'], ent['mention_label']) for ent in gt_doc_entities)
        pred_entities.extend((pmid, ent['start_idx'], ent['end_idx'], ent['entity_label']) for ent in pred_doc_entities)
    
    gt_set = set(gt_entities)
    pred_set = set(pred_entities)

    num_gt_entities = len(gt_entities)
    num_pred_entities = len(pred_entities)

    true_positives = len(gt_set & pred_set)
    false_positives = len(pred_set - gt_set)
    false_negatives = len(gt_set - pred_set)

    precision = true_positives / (true_positives + false_positives + 1e-10)
    recall = true_positives / (true_positives + false_negatives + 1e-10)
    f1_score = (2 * precision * recall) / (precision + recall + 1e-10)

    print('num_gt_entities: ', num_gt_entities)
    print('num_pred_entities: ', num_pred_entities)
    print('true_positives: ', true_positives)
    print('false_positives: ', false_positives)
    print('false_negatives: ', false_negatives)
    print('precision: ', precision)
    print('recall: ', recall)
    print('f1_score: ', f1_score)

if __name__ == "__main__":
    input_json_path = 'annotations/marco_all_annotations.json'
    metadata_json_path = 'predicted_relations/old_predicted_relations.json'
    ground_truth_json_path = 'annotations/ground_truth.json'
    finetune_data_json_path = 'annotations/finetune_data.json'

    parse_and_sort_annotation_results(input_json_path, metadata_json_path, ground_truth_json_path)
    process_ground_truth(ground_truth_json_path, finetune_data_json_path)


    #evaluate_ner(output_json_path, 'predicted_entities/predicted_entities.json')