# CMeEE-BERT
Project for AI3612

我们的**部分**结果在`./result`目录下，最好结果在`./result/roberta_large_crf_nested_2022`中。若对任何未给出的预测结果有问题或需要已训练好的模型，可以随时与我们联系。

## Installation

```shell
$ conda create -n cmeee python=3.9
$ conda activate cmeee
$ pip install -r requirements.txt
```

## Framework

- `data/`: 数据集
- `ckpt/`: 预训练好的模型
- `src/`: 源代码
  - `src/models/`: defination of models
  - `src/datasets/`: dataloader and dataset
  - `src/utils/`: utils
- `{bert-pretrained-model}/`: BERT 预训练模型，放在最外层，与 release 版保持一致
- `test_files/`: 测试文件

## Download Models

Dowload models
```shell
$ cd src
$ python scripts/download_model.py -d {download_path} -m {model_name} -c {cache_dir}
# E.g., (-c is optional)
$ python scripts/download_model.py -d ../chinese_roberta_wwm_ext -m hfl/chinese-roberta-wwm-ext
$ python scripts/download_model.py -d ../mc-bert -m freedomking/mc-bert
```
在报告中，我们还使用了：`hfl/chinese-roberta-wwm-ext-large`，`allenyummy/chinese-bert-wwm-ehr-ner-sl`，`weiweishi/roc-bert-base-zh`等模型

## Training

```shell
$ cd src
$ bash run_cmeee.sh {task_id} {model_type} {model_path} {replace} {num_gpu} {opt} {learning_rate}
# E.g.,
$ bash run_cmeee.sh 3 bert ../bert-base-chinese false 1
$ bash run_cmeee.sh 3 roberta ../xlm-roberta-base false 1
$ bash run_cmeee.sh 3 mcbert ../mc-bert false 1
```
