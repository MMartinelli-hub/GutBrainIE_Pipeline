[![-----------------------------------------------------](https://raw.githubusercontent.com/andreasbm/readme/master/assets/lines/colored.png)](#gutbrainie_pipeline)

# GutBrainIE Pipeline

## Project Description

The **GutBrainIE Pipeline** is a comprehensive tool designed to facilitate the retrieval, processing, and analysis of scientific literature related to the gut-brain axis. It integrates multiple components, such as Data Retrieval from PubMed, Preprocessing, Named Entity Recognition (NER), Relation Extraction (RE), and Visualization, to provide insights into the interactions between the gut and brain. The main goal of this pipeline is to assist researchers in the biomedical field to extract meaningful knowledge from vast amounts of literature.

## Table of Contents

1. [Features](#features)
2. [Installation](#installation)
3. [Usage](#usage)
4. [Project Structure](#project-structure)
5. [Examples](#examples)
6. [Contributing](#contributing)
7. [License](#license)

## Features

- **PubMed Retrieval**: Retrieve articles from PubMed using the `pubmed_retriever.py` script, based on predefined lists of PubMed IDs ([PMIDs](https://en.wikipedia.org/wiki/PubMed#PubMed_identifier)).
- **Preprocessing and Postprocessing**: Clean and process text data to facilitate annotation and relation extraction.
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

## Examples

Here are a few example use cases:

1. **Retrieve Articles**: Run `pubmed_retriever.py` to fetch PubMed articles listed in `pmid_list.csv`. The retrieved articles are saved in `retrieved_articles/`.
2. **Predict Annotations**: Use `gliner_interface.py` to run annotation predictions and save results in `predicted_annotations/`.
3. **Extract Relations**: Use `grapher_interface.py` to predict relations between concepts and save the results in `predicted_relations/`.

<!--- ## Contributing

Contributions are welcome! Please fork the repository, create a feature branch, and submit a pull request. --->

## License

This project is licensed under the Apache 2.0 License. 
<!--- See the `LICENSE` file for details. --->
