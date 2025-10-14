# CTI Analysis
We are developing a framework for converting unstructured Cyber Threat Intelligence (CTI) reports to document-scope knowledge graphs and aligning them to a unified cybersecurity ontology, allowing large-scale analysis of the CTI landscape.

Our framework utilizes LLMs for extracting entities and relationships from CTI sentences. It then inserts resulting triples to graph storage (Neo4j), generates embeddings over observed sentence contexts for semantic alignment, and computes similarity scores against a source graph. We utilize the Unified Cybersecurity Knowledge Graph for this task as it contains a large dataset scope, such as MITRE ATT&CK, CAPEC, and CVE/CVSS.

## Prerequisites
Our CTI Analysis project utlilizes the Unified Cybersecurity Knowledge Graph for graph alignment. Please follow the directions in the UCKG repo to download and run the UCKG. https://github.com/edogdu/UCKG

## Usage
To pull the repository into your file system:
```bash
git clone https://github.com/edogdu/CTI.git ./CTI
```
Then, move to the project root:
```bash
cd CTI
```

Install our project's dependencies:
```bash
pip install -r requirements.txt
```

Finally, run the pipeline with:
```bash
python main.py
```

## Dataset
We utilize the CTI-HAL dataset for testing. This dataset contains a set of 81 CTI PDF documents annotated by human Cybersecurity analysts. These annotations label MITRE ATT&CK entities including Techniques, Subtechniques, Tactics, and Softwares. This allows us to evaluate the alignment capabilities of our system, considering the annotations our ground truth.<br>
https://github.com/dessertlab/CTI-HAL

## Goal

Our target is a fully automated pipeline that handles ingestion of CTI material in various formats, structures it into a pdf-level knowledge graph, and aligns that graph with known Cybersecurity entities in a managed "source" graph.
<br><br>
![Image visualizing the workflow of CTI-Align](/resources/images/CTI-Align.png "CTI-Align")
<br><br>
