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
6. [Output Structure](#output-structure)
7. [Examples](#examples)
8. [License](#license)

## Features

- **PubMed Retrieval**: Retrieve articles from PubMed using the `pubmed_retriever.py` script, based on predefined lists of PubMed IDs ([PMIDs](https://en.wikipedia.org/wiki/PubMed#PubMed_identifier)).
- **Preprocessing and Postprocessing**: Clean and process text data to facilitate NER and RE.
- **Entity and Relation Extraction**: Use [GLiNER](https://github.com/urchade/GLiNER) and [GraphER](https://github.com/urchade/GraphER) modules to predict relevant entities and relations from the retrieved text.
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
   Edit the configuration files in the `config` directory as needed (see [5. Configuration Parameters](#configuration-parameters)).

## Usage

To run the pipeline, use the following command:

```sh
python main.py
```

### Main Components

- **PubMed Retriever**: Retrieves articles based on IDs listed in `pmid_lists/`.
  ```sh
  python pubmed_retriever.py
  ```
  **Output**:
  - Files are saved in `retrieved_articles/`.
  - **retrieved_articles.csv**: Contains metadata and content of the retrieved articles.
  - **retrieved_articles.txt**: Plain text versions of the retrieved articles, in PubTator format.

- **Gliner Interface**: Runs Named Entity Recognition .
  ```sh
  python gliner_interface.py
  ```
  **Output**:
  - Files are saved in `predicted_entities/`.
  - **<pmid>_entities.txt**: NER results for individual articles, in customized PubTator format.
  - **predicted_entities.txt**: Single text file containing all processed documents in PubTator format.
  - **predicted_entities.json**: JSON file containing all processed documents with predicted entities.

- **Grapher Interface**: Runs Relation Extraction.
  ```sh
  python grapher_interface.py
  ```
  **Output**:
  - Files are saved in `predicted_relations/`.
  - **<pmid>.txt**: Relation extraction results for individual articles, in customized PubTator format.
  - **predicted_relations.txt**: Single text file containing all processed documents in PubTator format.
  - **predicted_relations.json**: JSON file containing all processed documents with extracted relations.

The pipeline is modular, meaning you can run individual components as needed or execute `main.py` to run the complete workflow.

  
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
├── config/
│   ├── config.yaml                # Config for PubMed Retriever component
│   ├── gliner_config.yaml         # Config for Gliner component
│   └── grapher_config.yaml        # Config for Grapher component
├── EnriCO/						   # Folder containing the GraphER model
├── logs/
│   ├── gliner_pipeline.log        # Log file for Gliner component
│   └── grapher_pipeline.log       # Log file for Grapher component
├── pmid_lists/
│   ├── pmid_list.csv              # Full list of PubMed IDs
│   └── pmid_list_short.csv        # Short list of PubMed IDs 
├── predicted_entities/            # Output folder for GLiNER predictions (NER)
├── predicted_relations/           # Output folder for GraphER predictions (RE)
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

The configuration files located in the `config/` directory allow users to customize the pipeline's behavior. Below is a detailed description of the parameters that can be set in each configuration file:

### `config.yaml`
- **email**: Specifies the email address used for accessing PubMed resources.
- **ncbi_api_key**: Specifies the API key to be used to query NCBI database.
- **pmid_list_path**: Path to the file containing the list of PubMed IDs (PMIDs) that should be processed.
- **store_retrieved_articles_path**: Path to the directory where the retrieved articles will be saved.
- **store_retrieved_articles_file_name**: Name of the text and CSV files that will store the retrieved articles.

### `gliner_config.yaml`
- **model_name**: The name of the pretrained Named Entity Recognition (NER) model to be loaded (from local or from [HuggingFace](https://huggingface.co/)).
- **threshold**: A float value representing the minimum confidence level required for an entity prediction to be considered valid.
- **processing**: Defines the strategy for processing documents. Possible values include:
  - **document**: Processes each document as a single unit.
  - **sentence**: Processes each document by splitting it into sentences (delimited by periods).
  - **label**: Processes each document by splitting it into sentences (delimited by periods) and predicting one entity label at a time.
- **flat_ner**: A boolean flag indicating whether to perform Flat NER, which prevents overlapping entities.
- **multi_label**: A boolean flag indicating whether to allow multiple labels for the same entity.
- **corpus_file_path**: Path to the input file (in PubTator format) containing documents to be used for NER.
- **has_ground_truth**: A boolean flag specifying whether the corpus includes ground truth entities. If set to `True`, evaluation metrics such as precision, recall, and F1-score will be computed, along with an output file comparing ground truth and predicted entities.
- **output_directory**: Path to the directory where predicted entities will be stored. Outputs include separate files for each document in PubTator format, as well as aggregated files in both PubTator and JSON formats.
- **include_configuration_in_output**: A boolean flag specifying whether to include the configuration parameters at the top of the output file containing the aggregated predictions in PubTator format.
- **pubtator_aggregated_predictions_file_name**: Name of the text file in PubTator format that will store the aggregation of the articles and predicted entities.
- **json_aggregated_predictions_file_name**: Name of the JSON file that will store the aggregation of the articles and predicted entities.
- **pubtator_comparison_ground_truth_predicted_file_name**: Name of the text file in PubTator format that will store the comparison of the ground truth and predicted entities.
- **entity_labels**: A dictionary specifying the entity labels to be predicted by the model. The values are the entity labels considered by the model, while the keys are not used in the current implementation.

### `grapher_config.yaml`
- **model_name**: The name of the pretrained Relation Extraction (RE) model to be loaded.
- **threshold**: A float value representing the minimum confidence level required for a relation prediction to be considered valid.
- **one_by_one_prediction**: A boolean flag indicating whether to extract one relation type at a time.
- **corpus_file_path**: Path to the input file (in either PubTator or JSON format) containing documents to be used for RE.
- **load_from_json**: A boolean flag specifying whether the input corpus is in JSON format.
- **output_directory**: Path to the directory where predicted relations will be stored. Outputs include separate files for each document in PubTator format, as well as aggregated files in both PubTator and JSON formats. If the loaded corpus contains NER predictions, the output will also include those entity predictions.
- **include_configuration_in_output**: A boolean flag specifying whether to include the configuration parameters at the top of the output files.
- **pubtator_aggregated_predictions_file_name**: Name of the text file in PubTator format that will store the aggregation of the articles and predicted relations.
- **json_aggregated_predictions_file_name**: Name of the JSON file that will store the aggregation of the articles and predicted relations.
- **pubtator_comparison_ground_truth_predicted_file_name**: Name of the text file in PubTator format that will store the comparison of the ground truth and predicted relations.
- **has_ground_truth**: A boolean flag specifying whether the corpus includes ground truth relations. If set to `True`, evaluation metrics such as precision, recall, and F1-score will be computed (note: not implemented for RE).
- **relation_labels**: A dictionary specifying the relation labels to be predicted by the model. The values are the relation labels considered by the model, while the keys are not used in the current implementation.

## Output Structure

The outputs produced by the GutBrainIE Pipeline are stored in different formats to facilitate different use cases and integrations. Below is a detailed explanation of the output structures:

### PubTator Format

The PubTator format is a widely used format for storing biomedical text annotations, providing both raw text and labeled entities. In the GutBrainIE Pipeline, the PubTator format has been customized to include entity and relation predictions, allowing for an integrated representation of extracted knowledge. The customized structure includes:

- **Document ID**: The PubMed ID (PMID) of the document.
- **Title**: The title of the document.
- **Abstract**: The abstract of the document.
- **Predicted Entities**: Extracted entities, each consisting of:
    - **Start Position**: The starting character index of the entity in the text.
    - **End Position**: The ending character index of the entity in the text.
    - **Tag**: The tag of the entity, defining if it is located in the title (t) or in the abstract (a)
    - **Entity Text Span**: The actual text of the entity.
    - **Entity Label**: The predicted entity label (e.g., Chemical, Disease).
    - **Score**: The confidence of the model in the predicted entity.
- **Predicted Relations**: Extracted relations, each consisting of:
    - **Head Start Position**: The starting word index of the head entity in the text.
    - **Head End Position**: The ending word index of the head entity in the text.
    - **Head Tag**: The tag of the head entity, defining if it is located in the title (t) or in the abstract (a).
    - **Head Text Span**: The actual text of the head entity.
    - **Head Entity Label**: The predicted head entity label (e.g., Chemical, Disease).
    - **Tail Start Position**: The starting word index of the tail entity in the text.
    - **Tail End Position**: The ending word index of the tail entity in the text.
    - **Tail Tag**: The tag of the tail entity, defining if it is located in the title (t) or in the abstract (a).
    - **Tail Text Span**: The actual text of the tail entity.
    - **Head Entity Label**: The predicted tail entity label (e.g., Chemical, Disease).
    - **Relation Label**: The predicted relation label (e.g., interacts_with).
    - **Score**: The confidence of the model in the predicted relation.   

For instance:
```
33955443|t|Gut microbiota in mental health and depression: role of pre/pro/synbiotics in their modulation.
33955443|a|The microbiome residing in the human gut performs ...
33955443	0	14	t	Gut microbiota	microbiome	0.973515510559082
33955443	0	1	t	Gut microbiota	microbiome	24	25	a	human gut	anatomical location	has microbiome	0.601126968860626233955443	
```

### JSON Format

The predicted entities and relations are also stored in JSON format, providing a more structured representation that is easier to process programmatically. The JSON structure includes all the fields included in the PubTator format, and it is structured as follows:
```
{
    "33955443": {
        "title": "Gut microbiota in mental health and depression: role of pre/pro/synbiotics in their modulation.",
        "author": "Hasnain N Methiwala; Bhupesh Vaidya; Vamsi Krishna Addanki; Mahendra Bishnoi; Shyam Sunder Sharma; Kanthi Kiran Kondepudi",
        "journal": "Food & function",
        "year": "2021",
        "abstract": "The microbiome residing in the human gut performs ...",
        "pred_entities": [
            {
                "start_idx": 0,
                "end_idx": 14,
                "tag": "t",
                "text_span": "Gut microbiota",
                "entity_label": "microbiome",
                "score": 0.973515510559082
            }    
		]    
		"pred_relations": [
            {
                "head_start_idx": 0,
                "head_end_idx": 1,
                "head_tag": "t",
                "head_text_span": "Gut microbiota",
                "head_entity_label": "microbiome",
                "tail_start_idx": 24,
                "tail_end_idx": 25,
                "tail_tag": "a",
                "tail_text_span": "human gut",
                "tail_entity_label": "human",
                "relation_type": "has microbiome",
                "score": 0.6011269688606262
            }    
		]
	}    
}
```

## Examples

### Entire Pipeline Example
Here is an example use case for the entire pipeline, i.e., what is being done when launching `main.py`:
1. **Retrieve Articles**: Runs `pubmed_retriever.py` to fetch PubMed articles listed in `pmid_list.csv` (depending on the input path defined in `config/config.yaml`). The retrieved articles are saved in `retrieved_articles/`, both in PubTator format, in a file named `retrieved_articles.txt`, and in CSV format, in a file named `retrieved_articles.csv`. 
2. **Predict Entities**: Use `gliner_interface.py` to run NER on the corpus specified in the `config/gliner_config.yaml` file and save results in `predicted_entities/` (depending on the output path defined in the `config/gliner_config.yaml` file). The stored results include:
    - A separate text file for each processed document, named `<pmid>.txt`, in PubTator format
    - A single text file containing all the processed documents in PubTator format, named `predicted_entities.txt`
    - A single JSON file containig all the processed documents, named `predicted_entities.json`
3. **Predict Relations**: Use `grapher_interface.py` to run RE on the corpus specified in the `config/grapher_config.yaml`, predict relations between concepts, and save the results in `predicted_relations/` (depending on the output path defined in the `config/grapher_config.yaml` file). If the loaded corpus contains the predictions for the entities, these will also be included in the stored results. The stored results include:
    - A separate text file for each processed document, named `<pmid>.txt`, in PubTator format
    - A single text file containing all the processed documents in PubTator format, named `predicted_relations.txt`
    - A single JSON file containig all the processed documents, named `predicted_relations.json`

### Individual Components Examples
Here are a few example use cases:

1. **Retrieve Articles**: Run `pubmed_retriever.py` to fetch PubMed articles listed in `pmid_list.csv`. The retrieved articles are saved in `retrieved_articles/`.
2. **Predict Entities**: Use `gliner_interface.py` to run entity predictions and save results in `predicted_entities/`.
3. **Extract Relations**: Use `grapher_interface.py` to run relation predictions between concepts and save the results in `predicted_relations/`.

<!--- ## Contributing

Contributions are welcome! Please fork the repository, create a feature branch, and submit a pull request. --->

## License

This project is licensed under the [Apache 2.0 License](https://www.apache.org/licenses/LICENSE-2.0.txt). 
<!--- See the `LICENSE` file for details. --->
