# HEREDITARY Literature

## Folder Structure
- **old_literature**: Contains literature located in the folder *users/admin/Hereditary/Data/Literature* of the HEREDITARY Wiki retrieved from PubMed on 09/05/2024, referred to as "old".
- **new_literature**: Contains literature retrieved from PubMed on *31/10/2024*, using the same queries as for "old_literature".
- **merged_literature**: Literature obtained from merging both old and new collections.

## Retrieval Process
### Mental Health
- **Query**: “Gut microbiota” AND “Mental Health”
- **Platform**: [PubMed](https://pubmed.ncbi.nlm.nih.gov/)
- **Old Retrieval Date**: 09/05/2024
- **New Retrieval Date**: 31/10/2024

### Parkinson
- **Query**: “Gut microbiota” AND “Parkinson”
- **Platform**: [PubMed](https://pubmed.ncbi.nlm.nih.gov/)
- **Old Retrieval Date**: 09/05/2024
- **New Retrieval Date**: 31/10/2024

*Note*: For the new literature, documents retrieved for years 2014-2019 (mental health) and 2013-2020 (Parkinson) were not included in the final collection.

## Table Details

The rows represent the specific year or merging statistics, and the columns compare the old and new datasets. The "Combined Literature" section shows the overall statistics when merging separately the two datasets for both Mental Health and Parkinson topics.

- **Category**: The specific topic (Mental Health or Parkinson) or the overall combined literature.
- **Time Period**: The year or range of years from which the entries were retrieved.
- **Old Literature (Entries)**: Number of entries retrieved from the old dataset.
- **New Literature (Entries)**: Number of entries retrieved from the new dataset.
- **Merged Literature (Entries)**: Total number of entries after merging the old and new datasets without duplicates.

- **Combined Literature**: Represents the merging of both the mental health and Parkinson topics without duplicates.
- **hereditary_new_not_in_old**: Represents the entries that are present in the new dataset but not in the old dataset.
- **hereditary_old_new_merged**: Represents the result of merging the old and new datasets.

## Statistics Overview

| Category                   | Time Period     | Old Literature (Entries) | New Literature (Entries) | Merged Literature Without Duplicates (Entries) |
|----------------------------|-----------------|--------------------------|--------------------------|-----------------------------|
| **Mental Health**          | 2020            | 156                      | 87                       |                             |
|                            | 2021            | 88                       | 119                      |                             |
|                            | 2022            | 121                      | 158                      |                             |
|                            | 2023            | 192                      | 189                      |                             |
|                            | 2024            | 72                       | 188                      |                             |
|                            | **Merged Total** | **629**                  | **741**                  | **Merged: 1261**            |
|                            | **Unique**       | **581**                  | **680**                  | **Unique: 828**            |
|                            | **Duplicates**   | **48**                   | **61**                   | **Duplicates Removed: 433**   |
| **Parkinson**              | 2013-2020       | 68                       | 251                      |                             |
|                            | 2021            | 80                       | 146                      |                             |
|                            | 2022            | 114                      | 192                      |                             |
|                            | 2023            | 127                      | 190                      |                             |
|                            | 2024            | 10                       | 152                      |                             |
|                            | **Merged Total** | **399**                  | **782**                  | **Merged: 953**            |
|                            | **Unique**       | **249**                  | **704**                  | **Unique: 864**             |
|                            | **Duplicates**   | **150**                  | **78**                   | **Duplicates Removed: 89** |
| **Combined Literature**    | **Merged Total** | **1028**                 | **1523**                 | **Merged: 2183**            |
|                            | **Unique**       | **828**                  | **1354**                 | **Unique: 1663**            |
|                            | **Duplicates**   | **200**                  | **169**                  | **Duplicates Removed: 520** |

### Merged Literature Summary 

| Description                            | Value   |
|----------------------------------------|---------|
| **mental_health_new_not_in_old**          |         |
| Entries in `mental_health_new_merged_no_duplicates.csv`  | **680** |
| Entries in `mental_health_old_merged_no_duplicates.csv`  | **581** |
| Entries included in `mental_health_old_merged_no_duplicates.csv` | **435** |
| Entries not included in `mental_health_old_merged_no_duplicates.csv` | **247**| 
| **parkinson_new_not_in_old**          |         |
| Entries in `parkinson_new_merged_no_duplicates.csv`  | **704** |
| Entries in `parkinson_old_merged_no_duplicates.csv`  | **249** |
| Entries included in `parkinson_old_merged_no_duplicates.csv` | **89** |
| Entries not included in `parkinson_old_merged_no_duplicates.csv` | **615** | 
| **hereditary_new_not_in_old**          |         |
| Entries in `hereditary_new_merged_no_duplicates.csv`  | **1354** |
| Entries in `hereditary_old_merged_no_duplicates.csv`  | **828** |
| Entries included in `hereditary_old_merged_no_duplicates.csv` | **520** |
| Entries not included in `hereditary_old_merged_no_duplicates.csv` | **834** |
| **hereditary_old_new_merged**          |         |
| Before duplicates removal              | **2183** |
| After duplicates removal               | **1663** |
| Not-duplicated                         | **1143** |
| Duplicated                             | **520**  |
| Removed                                | **520**  |
| Unique Entries                         | **1663** |

