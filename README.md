

# TLERAD: Transfer Learning for Enhanced Ransomware Attack Detection

This repo has the following structure:
1. Code
2. Raw dataset
3. Sample Dataset

The code includes all the python files for the proposed TLERAD algorithm.
The raw dataset is unprocessed ransomware traffic files, which could be processed using the pre-processing file in the “Code” folder.
Sample Dataset: Includes a small pre-processed dataset which contains the dynamic analysis of 582 samples of ransomware and 942 of good applications (goodware), i.e., 1524 samples in total. The dataset was retrieved and analyzed with a Sandbox environment.

## Ransomware Samples
The Ransomware samples belong to different families that are identified with the following codes:

| FAMILY NAME       | ID  |
| ----------------- | --- |
| Goodware          | 0   |
| 'Locker'          | 1   |
| 'CryptLocker'     | 2   |
| 'CryptoWall'      | 3   |
| 'Scareware'       | 4   |
| 'RaaS'            | 5   |
| 'MalwareLocker'   | 6   |
| 'Coinminer'       | 7   |
| 'PGPCODER'        | 8   |
| 'Reveton'         | 9   |
| 'TeslaCrypt'      | 10  |
| 'Trojan-Ransom'   | 11  |

## IDS.txt File
The file IDS.txt contains the correspondence of the local IDS we use in our dataset with the SHA1 and MD5 of the software analyzed (both goodware and ransomware). The description of the header in that file is the following:

- ID: local identifier used in our dataset.
- SHA1: SHA1 hash identifier for the software.
- MD5: MD5 hash identifier for the software.
- Ransomware: 1 if it's ransomware / 0 for Goodware.
- Ransomware_Family: numeric identifier for the ransomware family (same codification as explained above).


