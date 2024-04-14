# Yokhal
욕쟁이 할머니

# Get started

## Installation
```bash
git clone https://github.com/seonglae/yokhal && cd yokhal
pip3 install torch==2.1.2 --index-url https://download.pytorch.org/whl/cu121
pip install .
pip install git+https://github.com/NVIDIA/TransformerEngine.git@stable
```


## Test
```bash
# Generate test text
python test.py generate
# Calculate Perplexity
python test.py ppl
```


## Train
```bash
# Finetune a full Yokhal
python train.py finetune
# PEFT with QLoRA
python train.py adapt
```
