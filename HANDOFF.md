# DeepFM CTR 项目交接文档

更新时间：2026-09-07

## 1. 项目与目标

项目用于训练社区内容 CTR DeepFM，当前目标是建立可靠的离线基线、改善同一用户内部排序，并与线上树模型公平比较。

```text
本地：D:\咪咕\deepfm_ctr_project
云端：/data/ganjiajing/deepfm_ctr_project
环境：/data/ganjiajing/ab_test/ab_env
数据：comm_sample_all_20260902.parquet
```

文件名中的日期是清洗时间。数据内 `part_date` 覆盖 2026-02-28 至 2026-09-06，共186天。服务器无 GPU，环境为 Python 3.10.20、TensorFlow 2.13.0、NumPy 1.23.5、pandas 1.5.3、PyArrow 25.0.1、scikit-learn 1.2.2。

## 2. 已完成的架构改造

`deepfm_ctr/DataProcess.py` 是唯一特征定义者，集中负责特征列表、合法范围、派生规则、词汇表、vocab size、数值统计、模型输入和预处理 artifact。

类别编码契约：

```text
0 = OOV
1 = MISSING
2... = 真实类别（包括原始业务值0）
```

`DataProcessor.fit()` 是词汇表和 vocab size 的唯一生成入口。模型接收 `processor.vocabularies`，不再猜测或写死大小。训练与推理必须使用配套的 `.keras` 和预处理 JSON。当前 `ARTIFACT_VERSION = 6`。

已经修复：

- `hotArticle` 按连续特征处理；
- `newArticle` 不直接入模，派生 `has_new_article` 表示有效值是否存在；
- `indexno` 派生长尾位置桶 `indexno_bucket`；
- 类别 OOV、Missing、真实业务0不再冲突；
- 类别非整数值不再被静默截断；
- Trainer 只负责训练和评估，`epochs/batch_size/patience` 必传；
- `train.py` 显式传入全部训练配置；
- 连续特征以1维权重进入 Linear，以 `embed_dim` 字段向量进入 FM/DNN。

## 3. 当前评估体系

训练结束输出：

```text
AUC、GAUC、LogLoss、NDCG@10
预测CTR均值、真实CTR
新用户GAUC、老用户GAUC、新文章GAUC及各自样本数
Embedding参数量
```

口径：

- GAUC按原始 `userid` 分组，剔除单一标签组，按有效组样本数加权；
- NDCG@10按原始 `userid` 分组，取有效组宏平均；
- 新用户：测试集非缺失用户未在训练集出现；
- 老用户：测试集非缺失用户曾在训练集出现；
- 新文章：测试集非缺失文章未在训练集出现。

评估性能已优化。旧实现对每个用户全局扫描，接近 `O(N×用户数)`；现在同步排序 `userid/label/prediction`，一次建立连续组边界，GAUC和NDCG复用边界。

评估使用辅助原始用户列 `__eval_userid`，它不会进入模型。这样即使未来加入用户 Embedding，新用户在模型侧都映射为 OOV，也不会在评估侧被错误合并。

## 4. 当前代码状态：version0

今天结束时已经回退并保留 version0：

```text
userid      不进入模型
article_id  不进入模型
theme_id    不进入模型
```

它们只作为评估元数据。当前状态是“version0模型特征 + 新评估体系”。同一批10%数据下，类别 Embedding 参数量应约为954。

version0 已记录结果：

```text
AUC              0.7095
GAUC             0.6492
LogLoss          0.1851
NDCG@10          0.7269
预测CTR均值       0.065812
真实CTR           0.049441
新用户GAUC        0.6184（113,199条）
新文章GAUC        0.6811（77,673条）
Embedding参数量   954
```

当时尚未输出老用户 GAUC。当前代码已经增加该指标，需要重新运行 version0 补齐。

## 5. 今日 ID Embedding 消融实验

每轮都从 version0 单独增加一个 ID，失败特征没有叠加到下一轮。

| 指标 | version0 | +article_id | +theme_id |
| --- | ---: | ---: | ---: |
| AUC | **0.7095** | 0.6770 | 0.6991 |
| GAUC | **0.6492** | 0.6351 | 0.6487 |
| LogLoss | **0.1851** | 0.1872 | 0.1866 |
| NDCG@10 | **0.7269** | 0.7190 | 0.7238 |
| 新用户GAUC | 0.6184 | 0.6089 | **0.6191** |
| 新文章GAUC | **0.6811** | 0.6758 | 0.6753 |
| Embedding参数 | **954** | 206,460 | 4,356 |

结论：

- `article_id` 约产生22,834个词项，新增约20.5万参数，明显降低所有主要指标；高基数、稀疏、文章热度漂移和新文章OOV是主要风险。
- `theme_id` 约378个词项，GAUC仅下降0.0005但AUC、LogLoss、NDCG和新文章GAUC均变差，没有证据支持保留。
- `userid` Embedding 代码曾实现，但尚无本交接可用的评估结果，随后已回退。若重做，必须从 version0 单独实验，并判断AUC提升是否只是记住用户基础点击率。
- 目前所有直接 ID Embedding 消融均未超过 version0，因此 version0 继续作为初始迭代版本。

## 6. 数据切分与树模型比较

当前 `config.py`：

```python
'start_date': '20260420'
'split_date': '20260520'
'end_date': '20260620'
'val_ratio': 0.1
```

实际切分：

```text
2026-04-20～2026-05-20：训练池
训练池按时间排序后的最后10%样本：验证集
2026-05-21～2026-06-20：测试集
```

验证按样本比例切割，可能将同一天拆入训练和验证。数据实际覆盖到9月6日，当前只使用部分窗口。

下一阶段要与线上树模型比较。时间窗口必须一致，但还不够；还必须对齐训练/验证/测试具体样本、标签观察窗口、过滤去重、负采样、特征可用时间、GAUC/NDCG分组和权重。最好用固定 `sample_id` 对齐两个模型的测试预测，再由同一评估脚本计算指标。

不要在确认树模型口径前自行扩大 DeepFM 窗口。后续可把日期配置改为 `train_start/train_end/val_end/test_end` 四个边界，按完整日期严格隔离，但应先取得树模型配置。

## 7. 下一步计划

1. 确认树模型完整数据窗口、样本和指标口径，建立公平基线。
2. 重跑当前 version0，补齐老用户 GAUC，并确认 Embedding 参数量约954。
3. 报告 GAUC有效用户数/样本覆盖率，对新老用户、头尾用户、新老文章进一步分层。
4. 当前 EarlyStopping 监控 `val_auc`，但目标偏组内排序；研究用验证集 GAUC或NDCG选择最佳 epoch。
5. 暂停盲目添加原始高基数 ID，转向用户—内容交叉、历史兴趣聚合、行为序列、频次截断、正则化、小维度 Embedding或哈希分桶。
6. 改善校准：version0预测CTR为6.58%，真实CTR为4.94%。
7. 优化全量读取：当前 `sample_rate` 在完整读取 parquet 后执行，不能降低首次内存峰值；应按列和 row group/batch 流式读取。
8. 使用多个随机种子或用户级 bootstrap 置信区间，避免把0.001以内变化误判为收益。

## 8. 当前运行命令

```bash
cd /data/ganjiajing/deepfm_ctr_project

CUDA_VISIBLE_DEVICES=-1 \
TF_CPP_MIN_LOG_LEVEL=2 \
python scripts/train.py \
  --data_dir /data/ganjiajing/deepfm_ctr_project/data \
  --sample_rate 0.10 \
  --batch_size 512 \
  --epochs 25 \
  --patience 5 \
  --output_dir /data/ganjiajing/deepfm_ctr_project/artifacts_version0_user_segments
```

## 9. 重要经验

1. `DataProcess.py` 必须是唯一特征定义来源。
2. 模型和 artifact 必须严格配套，实验输出目录不要覆盖。
3. 原始实体 ID 与编码 ID 不能混用；分组评估必须用原始 ID。
4. `sample_rate` 目前不是读取优化，只减少读取后的处理量。
5. 消融实验必须固定数据、日期、随机种子和训练参数。
6. AUC高不等于同用户内部排序好，应重点看GAUC、NDCG和分群指标。
7. 高基数类别会建立 Linear 1维和FM/DNN 8维两套表，参数增量约 `vocab_size×9`。
8. 测试期新实体只能使用OOV向量，原始ID难以改善冷启动。
9. TensorFlow 2.13依赖较旧，不要随意升级NumPy或`typing-extensions`。
10. 源码同步到云端后，Notebook/进程必须重启才能加载新类。
11. 与树模型比较时，日期一致只是最低条件，具体样本和评估实现也必须一致。

## 10. 主要文件与验证

```text
deepfm_ctr/DataProcess.py  特征、词表、转换、评估分组、artifact
deepfm_ctr/model.py        DeepFM结构
deepfm_ctr/trainer.py      训练和指标
deepfm_ctr/config.py       日期与超参数
scripts/train.py           训练、评估、保存入口
scripts/predict.py         推理入口
tests/                     回归测试
```

最近针对 version0、分群和 Trainer 的相关测试为 `18 passed`，源码编译通过。新会话开始应先确认本地文件已同步到云端，再运行 `python -m pytest -q`。
