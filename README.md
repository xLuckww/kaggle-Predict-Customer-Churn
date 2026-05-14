这是一个非科班的计算机小白第一次参加kaggle竞赛项目的源代码，竞赛名叫：Predict Customer Churn，竞赛主要是根据给定的train集进行训练，构建自己的模型，然后对test集验证自己模型，预测出其中各个客户可能的流失率并将结果文件上传到kaggle，最后我取得了0.90509的成绩（榜单第一名是0.91762）。
<img width="2496" height="1100" alt="image" src="https://github.com/user-attachments/assets/163317d5-9c79-43ae-8171-c0eaea97feb8" />
感谢gemini，这个模型全程是依靠gemini的辅助下完成的
最开始我也只是想抱着试试的态度，在现有ai能写出比较出色的代码下，我作为编程小白能不能也试着参加一下kaggle上的竞赛，在此之前我只完成过kaggle上的部分课程，对Python和机器学习只有一定的了解。我开始只想着能出分就算成功，于是我很简单粗暴的将所有的特征都转换成0或1的形式，对其他3个及以上的选项进行独热编码，然后直接采用最简单的决策树模型训练，取得了0.7分的成绩，这对我来说是个很好的开始！

后来我把模型换成了更好的XGB模型，直接让的分数提到0.82，后来在与gemini的不断沟通下改进我的特征，其中gemini有一段话很触动我：“特征决定了模型的上限，而调优只是为了接近这个上限”。
在gemini的帮助下改进了一些特征，但是分数始终都在0.8+打转，无法突破到0.9，gemini帮我构建了一张图，指出了各个特征对分数的影响程度，令我没想到的是，支付方式居然才是影响客户流失最重要的一个因素！
<img width="3388" height="1964" alt="picture" src="https://github.com/user-attachments/assets/37c03f89-9e05-46c6-81f5-34a0c175ca4b" />

有了这个信息，我又根据支付方式构建了一些更好的特征，成功把成绩提升到了0.89797！
但这剩下的0.01的提升是何其之难，其中我尝试过运行Optuna 自动调优跑了20多分钟找出了xgb模型的最佳参数，结果跑出来的分数反而更低了；同时再又添加了好几个相关特征后，提升要么只有0.00001要么反而更低！
在最后，我直接不把训练集划分，直接用100%的全量数据来构建我的模型，最后用于测试集上，成功在kaggle上将分数跑进0.9以上！

全部的代码都是gemini帮我写的，说实话我一个都写不出来，但是我能看的懂是在做什么，这是我的第一次参加竞赛的项目，这让我很有成就感！

以下是正式的README
# 📉 Customer Churn Prediction — Kaggle Playground S6E3

预测电信客户是否会流失（Churn），基于用户合约、消费行为和服务订阅等特征构建分类模型。

**比赛成绩：ROC-AUC 0.91542 | 排名 1097 / ~3,700**

---

## 项目简介

本项目参加 [Kaggle Playground Series Season 6 Episode 3](https://www.kaggle.com/competitions/playground-series-s6e3/overview) 比赛。数据集包含约 594,000 条电信客户记录，共 21 个特征，目标是预测客户流失概率，评估指标为 ROC-AUC。

核心思路：在原始特征的基础上，手动构造聚合特征和交叉特征，捕捉特征组合带来的额外流失信号，再用 XGBoost 进行分类。

---

## 技术栈

| 类别 | 工具 |
|------|------|
| 语言 | Python 3 |
| 数据处理 | pandas, numpy |
| 建模 | XGBoost |
| 可视化 | matplotlib, seaborn |

---

## 特征工程

原始特征经过以下处理：

**聚合特征**
- `Service_Count`：用户订阅的增值服务数量（OnlineSecurity、TechSupport 等 6 项）
- `TotalCharges_Log`：总消费额的对数变换，缓解右偏分布
- `Monthly_Ratio`：月均消费 / 总消费，反映消费稳定性
- `Tenure_Group`：按在网时长分组（New / Medium / Loyal）
- `Is_High_Risk_Newbie`：在网 < 6 个月且月费 > 70 的高风险新用户标志

**交叉特征**（编码后构造）
- `Easy_To_Leave`：电子支付 + 非长期合约（最容易流失的组合）
- `Fiber_Without_Support`：光纤用户但无技术支持
- `Payment_Pain_Index`：电子支付 × 高消费额
- `High_Monthly_Check_Risk`：电子支付 × 月费比例高于均值

---

## 模型参数

```python
XGB_PARAMS = dict(
    n_estimators     = 350,
    max_depth        = 4,
    learning_rate    = 0.02,
    subsample        = 0.8,
    colsample_bytree = 0.8,
    reg_alpha        = 1.5,
    reg_lambda       = 1.5,
    random_state     = 42,
)
```

---

## 如何运行

**1. 安装依赖**

```bash
pip install pandas numpy matplotlib seaborn xgboost
```

**2. 准备数据**

从 [比赛页面](https://www.kaggle.com/competitions/playground-series-s6e3/data) 下载 `train.csv` 和 `test.csv`，放置于项目根目录。

**3. 运行**

```bash
python kaggle-Predict_Customer_Churn.py
```

运行后会在当前目录生成 `submission.csv`，可直接提交至 Kaggle。

---

## 文件结构

```
├── kaggle-Predict_Customer_Churn.py   # 主脚本
├── train.csv                          # 训练集（需自行下载）
├── test.csv                           # 测试集（需自行下载）
└── submission.csv                     # 预测结果（运行后生成）
```

---

## 作者

xLuck · [GitHub](https://github.com/)
