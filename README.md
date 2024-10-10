[![-----------------------------------------------------](https://raw.githubusercontent.com/andreasbm/readme/master/assets/lines/colored.png)](#gutbrainie_pipeline)

# GutBrainIE Pipeline

## Project Description

The **GutBrainIE Pipeline** is a comprehensive tool designed to facilitate the retrieval, processing, and analysis of scientific literature related to the gut-brain axis. It integrates multiple components, such as Data Retrieval from PubMed, Preprocessing, Named Entity Recognition (NER), Relation Extraction (RE), and Visualization, to provide insights into the interactions between the gut and brain. The main goal of this pipeline is to assist researchers in the biomedical field to extract meaningful knowledge from vast amounts of literature by providing reliable distantly-supervised annotations for entities and relations in biomedical texts. 

## Table of Contents

1. [Features](#features)
2. [Installation](#installation)
3. [Usage](#usage)
4. [Project Structure](#project-structure)
5. [Configuration Parameters](#configuration-parameters)
6. [Examples](#examples)
7. [License](#license)

## Features

- **PubMed Retrieval**: Retrieve articles from PubMed using the `pubmed_retriever.py` script, based on predefined lists of PubMed IDs ([PMIDs](https://en.wikipedia.org/wiki/PubMed#PubMed_identifier)).
- **Preprocessing and Postprocessing**: Clean and process text data to facilitate NER and RE.
- **Annotation and Relation Extraction**: Use [GLiNER](https://github.com/urchade/GLiNER) and [GraphER](https://github.com/urchade/GraphER) modules to predict relevant annotations and relations from the retrieved text.
- **Configuration Management**: Configurable pipeline with YAML files for easy adjustments to different settings.
- **Logging**: Comprehensive logging of the pipeline to track progresses and identify issues.

## Installation

To set up the GutBrainIE Pipeline locally, follow these steps:

1. **Clone the Repository**:

   ```sh
   git clone https://github.com/MMartinelli-hub/GutBrainIE_Pipeline.git
   cd GutBrainIE_Pipeline
   ```

2. **Create a Virtual Environment** (optional but recommended):

   ```sh
   python3 -m venv venv
   source venv/bin/activate  # On Windows use: venv\Scripts\activate
   ```

3. **Install Dependencies**:

   ```sh
   pip install -r requirements.txt
   ```

4. **Set Up Configuration**:
   Edit the configuration files in the `config` directory as needed.

## Usage

To run the pipeline, use the following command:

```sh
python main.py
```

### Main Components

- **PubMed Retriever**: Retrieves articles based on IDs listed in `pmid_lists/`. The IDs must be contained in a CSV file located in the `pmid_lists/`, having a single column named `pmid` (refer to the samples already located in the folder).
  ```sh
  python pubmed_retriever.py
  ```
- **Gliner Interface**: Performs NER using GLiNER. The configuration file is named `gliner_config.yaml` and is located in the folder `config/`.
  ```sh
  python gliner_interface.py
  ```
- **Grapher Interface**: Performs RE using GraphER. The configuration file is named `grapher_config.yaml` and is located in the folder `config/`.
  ```sh
  python grapher_interface.py
  ```

The pipeline is modular, meaning you can run individual components as needed or execute `main.py` to run the complete workflow.
If you want to run individual components, check inside the component of interest to understand what it's being done.
Note that running individual components is intended just for debugging and testing purposes.

## Project Structure

```
GutBrainIE_Pipeline/
├── .git/                          # Git-related files (version control)
├── config/
│   ├── config.yaml                # Config for PubMed Retriever component
│   ├── gliner_config.yaml         # Config for Gliner component
│   └── grapher_config.yaml        # Config for Grapher component
├── logs/
│   ├── gliner_pipeline.log        # Log file for Gliner component
│   └── grapher_pipeline.log       # Log file for Grapher component
├── pmid_lists/
│   ├── pmid_list.csv              # Full list of PubMed IDs
│   └── pmid_list_short.csv        # Short list of PubMed IDs 
├── predicted_annotations/         # Output folder for Gliner predictions
├── predicted_relations/           # Output folder for Grapher predictions
├── retrieved_articles/            # Output folder for retrieved PubMed articles
├── main.py                        # Entry point for running the pipeline
├── gliner_interface.py            # Gliner module interface
├── grapher_interface.py           # Grapher module interface
├── preprocess.py                  # Preprocessing code (NOT IMPLEMENTED)
├── postprocess.py                 # Postprocessing code (NOT IMPLEMENTED)
├── pubmed_retriever.py            # Retrieves articles from PubMed
└── utils.py                       # Utility functions
```

## Configuration Parameters

The configuration files located in the `config/` directory allow users to customize the pipeline's behavior. Below are the details of the parameters that can be set in each configuration file:

### `config.yaml`
- **email**: The email used for accessing PubMed data.
- **pmid_list_path**: Path to the file where the list of PMIDs is located.
- **store_retrieved_articles_path**: Path to the directory where the retrieved articles will be saved.

### `gliner_config.yaml`
- **model_name**: The name of the pretrained NER model to load (from [HuggingFace](https://huggingface.co/)).
- **threshold**: The threshold value (float) to be used by the model when predicting entities. It indicates the minimum confidence level for a prediction to be kept.
- **processing**: The processing type to be used by the model. It is possible to decide between:
    - document: Process each document in a single batch.
    - sentence: Process each document considering each sentence as a separated batch. A sentence is considered delimited by the dot ('.') symbol.
    - label: Same as sentence, but predicts a single entity label at a time.
- **flat_ner**: The boolean value defining whether to perform Flat NER, i.e., no overlapping entities.
- **multi_label**: The boolean value defining whether to perform multi-label extraction, i.e., multiple labels for the same entity.
- **corpus_file_path**: The path to the file (in PubTator format) containing the documents to be processed for NER.
- **has_ground_truth**: The boolean value defining if the corpus contains also the ground truth (in PubTator format). If set to True, evaluation metrics (precision, recall, F1-score) will be computed, and a file comparing ground truth and predicted entities will be stored.
- **output_directory**: The path to the directory where to store the predicted entities. It will be stored a separate file in PubTator format for each processed documents, named <pmid>.txt, and two files, one in PubTator format and one in JSON format, containing all the processed documents and predicted entities.
- **include_configuration_in_output**: The boolean value defining whether to store the configuration parameters at the top of the output files.
- **entity_labels**: The dictionary (i.e., pairs of key-value) defining the entity labels to be predicted by the model. The model will consider as entity labels the values populating the dictionary. The keys are not used.

### `grapher_config.yaml`
- **model_name**: The name of the pretrained RE model to load.
- **threshold**: The threshold value (float) to be used by the model when predicting relations. It indicates the minimum confidence level for a prediction to be kept.
- **one_by_one_prediction**: The boolean value defining whether to extract one relation type at a time.
- **corpus_file_path**: The path to the file (in PubTator or JSON format) containing the documents to be processed for RE.
- **load_from_json**: The boolean value defining if the corpus to be loaded is in JSON format.
- **output_directory**: The path to the directory where to store the predicted relations. It will be stored a separate file in PubTator format for each processed documents, named <pmid>.txt, and two files, one in PubTator format and one in JSON format, containing all the processed documents and predicted relations. If the loaded corpus is the JSON produced in output by the NER module, the output will be comprehensive of both predicted entities and relations.
- **include_configuration_in_output**: The boolean value defining whether to store the configuration parameters at the top of the output files.
- **has_ground_truth**: The boolean value defining if the corpus contains also the ground truth (in PubTator format). If set to True, evaluation metrics (precision, recall, F1-score) will be computed, and a file comparing ground truth and predicted entities will be stored (NOT IMPLEMENTED FOR RE).
- **relation_labels**: The dictionary (i.e., pairs of key-value) defining the relation labels to be predicted by the model. The model will consider as relation labels the values populating the dictionary. The keys are not used.

## Examples

Here are a few example use cases:

1. **Retrieve Articles**: Run `pubmed_retriever.py` to fetch PubMed articles listed in `pmid_list.csv`. The retrieved articles are saved in `retrieved_articles/`.
2. **Predict Annotations**: Use `gliner_interface.py` to run annotation predictions and save results in `predicted_annotations/`.
3. **Extract Relations**: Use `grapher_interface.py` to predict relations between concepts and save the results in `predicted_relations/`.

<!--- ## Contributing

Contributions are welcome! Please fork the repository, create a feature branch, and submit a pull request. --->

## License

This project is licensed under the [Apache 2.0 License](https://www.apache.org/licenses/LICENSE-2.0.txt). 
<!--- See the `LICENSE` file for details. --->
