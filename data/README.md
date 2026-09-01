# Data preparation

The public repository intentionally does not include Amazon review data, model checkpoints, item embeddings, or generated predictions.

Prepare the Amazon Review 2018 `Industrial_and_Scientific` split under this layout:

```text
data/Amazon/
├── index/
│   ├── Industrial_and_Scientific.emb-qwen-td.npy
│   ├── Industrial_and_Scientific.index.json
│   ├── Industrial_and_Scientific.item.json
│   └── Industrial_and_Scientific.{train,valid,test}.inter
├── train/Industrial_and_Scientific_5_2016-10-2018-11.csv
├── valid/Industrial_and_Scientific_5_2016-10-2018-11.csv
├── test/Industrial_and_Scientific_5_2016-10-2018-11.csv
└── info/Industrial_and_Scientific_5_2016-10-2018-11.txt
```

The processed split follows the upstream [MiniOneRec](https://github.com/AkaliKong/MiniOneRec) data format. Obtain the raw Amazon Review data from its official distributor and comply with its terms. Do not commit private data, model weights, API keys, or machine-specific paths.

After preparing `index/*.inter`, `index/*.index.json`, and `index/*.item.json`, regenerate the CSV/info files with:

```bash
bash convert_dataset.sh
```

To rebuild item embeddings locally, run the embedding script with a valid model path:

```bash
cd rq/text2emb
EMBED_MODEL=/path/to/embedding/model NUM_PROCESSES=1 bash amazon_text2emb.sh
```
