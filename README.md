# TEASEL: A Transformer-based Speech-Prefixed Language Model

This is an implemention of [TEASEL: A Transformer-based Speech-Prefixed Language Model](https://arxiv.org/pdf/2109.05522.pdf). We have listed below the steps to reproduce the experiments. Our model files are downloadable [here](https://drive.google.com/drive/folders/1mjfqjuDjceBWaE-IU4Xh0tKpOtX9y30g?usp=sharing) and the results closely match those published in the paper. We are not the original authors, but are open to discuss any issue with code or paper's understanding. This repo is developed and maintained by [Atharva Moroney](https://github.com/ath-08) and myself.

## Installation

Please check that you have all the requirements in sync with the requirements.txt file, alteranatively you could execute the command below:

```
pip install -r requirements.txt
```

We have not tested the code to work on other versions, especially for the transformers and pytorch libraries.

## Pretraining

For the pretraining phase, please download librispeech dataset ({train/test/dev}-clean-100 version), alter the filepath column in data/librispeech_{train/test/dev}_df.csv to point to the place where you have stored the data. Then execute the following command to start the pretraining process:

```
python train.py --config config.yaml
```

## Finetuning

For the finetuning phase, please download CMU MOSI dataset, alter the filepath column in data/mosi_{train/test/dev}_df.csv to point to the place where you have stored the data. Then execute the following command to start the finetuning process:

```
python fine_tune_mosi.py --config config_mosi.yaml
```

## Results

Due to some discrepancies and unknowns in the paper our results don't exactly match as reported in the paper but are nevertheless very close to them. 

| Metric      | BA    | F1   | MAE  | Corr  |
| ---------- |:------:| :----:|-----:| :-----:|
| [TEASEL (Original)](https://arxiv.org/pdf/2109.05522.pdf)  | 89.3  | 89.31| 0.644| 0.84	|
| Ours        | 87.5  | 85.0 | 0.647| 0.836 |


## Citations

@misc{arjmand2021teasel,
      title={TEASEL: A Transformer-Based Speech-Prefixed Language Model}, 
      author={Mehdi Arjmand and Mohammad Javad Dousti and Hadi Moradi},
      year={2021},
      eprint={2109.05522},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}