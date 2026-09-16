# DeepFM CTR 项目交接文档

更新时间：2026-09-14

## 1. 项目与当前任务

本项目用于双排视频推荐页 CTR 预估和排序，并与线上树模型比较。当前重点是建立可信的时间切分对照，定位 `userid`、`article_id`、`theme_id` 三个高基数 ID 加入 Embedding 后严重退化的原因。

```text
本地：D:\咪咕\deepfm_ctr_project
云端：/data/ganjiajing/deepfm_ctr_project
环境：/data/ganjiajing/ab_test/ab_env（TensorFlow 2.13，CPU）
仓库：https://github.com/dumplinggeganjiajing-star/deepfm_ctr_project
```

数据文件名日期是清洗日期，样本日期以 `part_date` 为准。

## 2. 已完成的架构改造

`deepfm_ctr/DataProcess.py` 是唯一特征定义者，负责特征列表、合法范围、派生规则、词表、`vocab_size`、数值统计、模型输入和预处理 artifact。

类别编码契约：

```text
0 = OOV；1 = MISSING；2... = 真实类别（业务值0也是正常类别）
```

主要修复：

- `hotArticle` 改为连续特征；
- `newArticle` 派生为 `has_new_article` 缺失状态特征；
- `indexno` 改为 `indexno_bucket`；
- Linear 连续字段使用1维权重，FM/DNN 使用 `embed_dim`；
- Trainer 只负责训练评估，训练配置由入口显式传入；
- 模型和完整预处理 artifact 配套保存、加载并贯通推理；
- 指标包含 AUC、GAUC、LogLoss、NDCG@10、预测/真实CTR、新/老用户GAUC、新文章GAUC和Embedding参数量；
- GAUC/NDCG 按原始 `userid` 分组，不能用编码后的ID。

## 3. version0 与树模型正式基准

对齐窗口：训练 `2026-03-09～2026-09-08`，测试 `2026-09-09`。DeepFM 先用验证集选出最佳 epoch=6，再用完整训练窗口固定训练6轮。

| 指标 | 树模型 | DeepFM version0 |
| --- | ---: | ---: |
| AUC | 0.762981 | 0.7659 |
| GAUC | 0.645793 | 0.6475 |

DeepFM 其他指标：LogLoss=0.1813，NDCG@10=0.5699，预测/真实CTR=0.057410/0.053502，新用户GAUC=0.6556，老用户GAUC=0.6417，新文章GAUC=0.5336，Embedding参数=972。

差值不足0.003，未做置信区间或多随机种子验证，不能声称显著领先；新文章排序是明显短板。

## 4. 历史单ID消融（较早10%样本）

| 指标 | version0 | +article_id | +theme_id | +userid |
| --- | ---: | ---: | ---: | ---: |
| AUC | 0.7095 | 0.6770 | 0.6991 | 0.6876 |
| GAUC | 0.6492 | 0.6351 | 0.6487 | 0.6754 |
| LogLoss | 0.1851 | 0.1872 | 0.1866 | 0.1859 |
| NDCG@10 | 0.7269 | 0.7190 | 0.7238 | 0.7294 |
| Embedding参数 | 954 | 206,460 | 4,356 | 506,151 |

窗口与正式比较不同，只能看组内趋势。article/theme 无收益；userid 当时提高GAUC但降低AUC，可能主要学习用户偏置。

## 5. 三ID + 80/10/10 失败实验

窗口内按 `part_date` 排序，再按样本行数切分：

```text
train 6,757,784：2026-03-09～2026-08-07
val     844,723：2026-08-07～2026-08-25
test    844,724：2026-08-25～2026-09-09
边界日期在相邻集合共享。
```

三个ID同时加入后 Total params=7,128,671：

```text
epoch1 train_auc=.7661 val_auc=.6877 val_loss=.2021
epoch2 train_auc=.8377 val_auc=.6224 val_loss=.2597
epoch3 train_auc=.8794 val_auc=.5784 val_loss=.3591
epoch4 train_auc=.9019 val_auc=.5502 val_loss=.4579
```

这是严重的高基数ID记忆型过拟合。

## 6. 最新控制组：80/10/10，无三ID

控制分支仅增加相同比例切分，不增加三个ID。训练前四轮验证AUC稳定在约0.695～0.699，没有三ID方案的失控退化。

| 指标 | 结果 |
| --- | ---: |
| AUC | 0.6970 |
| GAUC | 0.6128 |
| LogLoss | 0.2022 |
| NDCG@10 | 0.5494 |
| 预测/真实CTR | 0.061472 / 0.055835 |
| 新用户GAUC | 0.6151（655,247条） |
| 老用户GAUC | 0.6075（189,477条） |
| 新文章GAUC | 0.5993（391,359条） |
| Embedding参数量 | 972 |

测试集约844,724条，新用户约77.57%，新文章约46.33%；CTR相对高估约10.1%。AUC比GAUC高0.0842，说明全局区分强于同用户内部排序。

关键结论：ratio 时间窗口/分布变化使无ID基线也下降，但无ID训练稳定；三ID在相同切分下严重崩溃，因此主要增量原因是高基数ID过拟合，时间漂移是共同背景因素。后续ID实验必须与本控制组比较，不能与原固定日期version0直接比较。

## 7. 当前代码与Git状态

```text
分支：experiment/version1-ratio-split-control
提交：c748715 add ratio split control without ID embeddings
```

提交包含 `DataProcess.py`、`config.py`、`train.py`、`tests/test_ratio_split.py`，新增：

- `--split_strategy {date,ratio}`；
- `--train_ratio`、`--validation_ratio`；
- ratio模式在 `[start_date,end_date]` 内按日期稳定排序后按行数切分；
- 默认仍为原date模式；三个ID不在 `CATEGORICAL_FEATURES`。

验证：`37 passed`、compileall通过、三个ID与类别特征交集为空。

GitHub推送曾因443超时和 `Recv failure: Connection was reset` 失败；这是网络/代理问题，本地提交安全。恢复网络后：

```bash
git push -u origin experiment/version1-ratio-split-control
```

历史仓库错误跟踪了一些 `__pycache__`。测试后提交前执行 `git status`，不要提交缓存；必要时：

```bash
git restore deepfm_ctr/__pycache__ scripts/__pycache__ tests/__pycache__
```

## 8. 切分与穿越

ratio是按日期排序后的样本比例切分，不是完整自然日隔离；边界日可跨集合。这不等同于标签穿越。用户/文章跨集合出现也属推荐场景常态。

真正需要审计：重复曝光/`sample_id` 跨集合、上游统计使用样本发生后的数据、当日全量统计包含当前标签，以及1d/3d/7d/14d CTR、用户行为、文章热度是否严格满足 `feature_timestamp < sample_timestamp`。

严格因果验证应使用完整日期隔离并断言：

```python
assert train[DATE_COL].max() < val[DATE_COL].min()
assert val[DATE_COL].max() < test[DATE_COL].min()
```

## 9. 下一步计划

1. 固化无三ID 8/1/1控制组的窗口、随机种子、配置和产物。
2. 检查边界时间、各集合CTR/每日样本量及重复 `sample_id`。
3. 审计上游统计特征时间窗口，排除未来信息。
4. 不再一次加入三个ID；一次只改一个变量，优先低基数 `theme_id`。
5. 高基数ID需尝试最低频次、hash/截断、2～4维Embedding、L2正则和强早停。
6. 统计各ID的OOV、新实体比例、频次及集合交集，并报告新老实体指标。
7. 优先改善同用户排序和新文章：内容、作者、发布时间、时效性、页面上下文及历史聚合特征。
8. 最终方案做多随机种子或用户级bootstrap。
9. 与树模型正式比较时，必须完全对齐窗口、过滤、标签、候选集和GAUC口径。

## 10. 长训练注意事项

JupyterLab终端关闭后先检查进程：

```bash
ps -ef | grep "[p]ython scripts/train.py"
```

以后使用 `tmux` 加日志运行长训练。当前若没有逐epoch checkpoint，进程停止后不能续接中断epoch，只能重训；后续可增加 `ModelCheckpoint` 和 optimizer/`initial_epoch` 恢复。

## 11. 关键经验

1. 一次只改变一个因素。
2. ratio无ID控制组证明时间漂移降低基线，三ID则造成额外严重过拟合。
3. 训练AUC急升而验证AUC、LogLoss恶化是记忆，不是提升。
4. 双排推荐不能只看AUC，GAUC/NDCG更接近同用户内部排序。
5. 新用户比例、文章更新率和时间漂移必须随指标报告。
6. 模型与预处理artifact必须配套。
7. push失败不会丢失本地commit；本地分支也不会自动成为远程分支。
8. 不应跟踪 `__pycache__`。

## 12. 新会话启动检查

```bash
git status -sb
git branch -vv
git log --oneline --decorate --all -8
python -m pytest -q
```

确认当前分支、`c748715`是否已推送、工作区是否只有预期修改、三个ID是否仍未进入类别特征，并确保下一项实验只改变一个变量。
