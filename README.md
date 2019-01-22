# CardEst
Cardinality Estimation in a Virtualized Network Device Using Online Machine Learning.

This code is used to reproduce the results described in:

"Cardinality Estimation in a Virtualized Network Device Using Online Machine Learning" by Cohen et. al.

TODO: add link to paper once published.

## Installation
First, install requiremnts by using your favourite package manager:

- pip
`pip install -r requirements.txt`

- conda
`conda install --file=requirements.yml`

### TShark

In order the use the preprocessing module you need to install tshark, run:

`sudo apt install tshark`

for OSX use homebrew to install the wireshark package:

`brew install wireshark`

## Data
The data in this repo was extracted from 4 data traces.
Each .pickle file in ./data contains a pandas dataframe which holds specific features (as described in the paper)
from the following datasets.

- CAIDA-2016 - <http://www.caida.org/data/passive/passive_2016_dataset.xml>
- CAIDA-DDoS - <http://www.caida.org/data/passive/ddos-20070804_dataset.xml>
- DARPA-DDoS - <http://www.darpa2009.netsec.colostate.edu/>
- UCLA-CSD - <https://lasr.cs.ucla.edu/ddos/traces/>

The preprocessing module (./src/preprocessing.py) was used to create the dataframes.

** Since only UCLA-CSD is available publicly, all other traces were excluded from this public repository. **

## Plotting

All plotting is done by the visualization module (./src/visualization.py).
The code to generate plots found in the paper is included in Jupyter notebooks (./ipython)

for questions and more information please contact:
yuvalnezri [at] gmail.com
