# Transformers
This repo hosts code for PyTorch implementation of the paper "Attention is all you need"
<br>

"Attention is All You Need," a seminal paper released by Google in 2017, revolutionized the field of natural language processing and machine learning. Authored by researchers at Google Brain, the paper introduced the concept of the Transformer model, a novel architecture that relies on attention mechanisms to process input sequences. Unlike previous sequence-to-sequence models, the Transformer's attention mechanism allows it to weigh the significance of different parts of the input, enabling more effective language understanding and generation
## How to use?
To use this repo for your desired source to target language, you need to make changes to the `config/config.yaml` file. <br>
1. Setup conda environment:<br>
   `conda create -n transformer python=3.8 -y`
2. Activate the environment:<br>
   `activate transformer`
3. Clone the repo:<br>
   `git clone github.com/shariqhameedca/transformers`
4. Cd into the repo:<br>
   `cd transformers`
5. Install the dependencies:<br>
   `pip install -r requirements.txt`
6. Make changes to the `config/config.yaml` file
7. To train run:<br>
   `python src/train.py`
