# GSLAformer R3：机制型合成验收

> 配套：[设计](design.md)、[实施规格](implementation.md)。本文件是未来模型开发的验收合同，不是已经跑出的模型结果。
>
> 数据与数值均为人为构造的机制示例，不代表真实功率或业务规律。首轮不启动真实数据验证，不要求 full 在所有任务胜过对照。

## 1. 为什么需要独立机制任务

R2 周期 base 可以用季节性 naive 几乎精确预测，只适合检查读写、窗口、指标、保存与重载。R3 分开三类证据：

- **数据／公式探针**：证明任务确实具有要检验的信息差异；可在模型实现前核算。
- **机制可表达与可学习**：模型实现后，通过成对输入、受控路径和小集拟合验证；不是靠一次随机 forward 恰好变化。
- **外推结果**：冻结同一模型，评估未见轨迹／关系／实体组合，如实记录失败，不称真实业务效果。

表中“必须”描述未来验收，不代表当前已通过。两阶段训练可能仍学不到需要的状态，这属于待实验问题，不用文档自洽代替证据。

## 2. 数据组织与统一协议

### 2.1 任务目录和配置

生成器 `--suite all` 产出：

- `base/`：原周期 smoke；
- `mechanisms/M1/` 至 `mechanisms/M4/`：独立 schema、episode、完整 train/valid/test 配置；
- `mechanisms/M5/`：数据契约和数值边界夹具，不训练独立模型；
- `mechanisms/suite.json`：任务 ID、数据包、配置路径、预期断言、split 与 family 清单；
- `new_entity_cold/` 等请求：供 base 入口 smoke，不代替 M4。

每个 family 是一组只在指定因素上不同的成对或多变体 episode；同 family 全部进入同一 split。每任务采用 48 个 train family、16 个 valid family、16 个 test family。编号 0..47/48..63/64..79，split 规则在 manifest 中写明，不能依据结果重新分配。

RNG 为 `default_rng(SeedSequence([2023, task_number, family_number]))`。模型训练种子仍为 2023。不同 family 独立采样，ID 前缀不同；family 编号、任务编号、文件路径均不输入模型。

每 episode 包含 L=24 个历史位置与 H=6 个未来标签，只有一个 origin；不跨 episode 切窗。各 family 使用相同示意时间轴，避免从时间戳推断目标差异。生产 forecast 包只含历史与原点已知条件，标签另存于测试包。

机制配置使用 `split_mode=episode_family`、`split_index=null`，collection manifest 记录 family split。配置的 groups、known_future、target_name、freq 和 L/H 必须与该任务 manifest 精确一致；生成器保存完整配置而非运行时修改 base 字段。对每个 pair 变体创建独立 episode id、共用 family_id。

### 2.2 避免伪造信息差异

- 所有目标先由生成规则计算，再施加输入缺失；填充值不能反向改变 target。
- 不把 simulator hidden state、family ID、未来真实动态值作为模型输入。
- 各任务单独拟合训练 scaler，不使用 valid/test 统计；属性词表只来自训练任务。
- M1/M2 成对版本的 target_history 必须逐元素相同，不能让模型从历史总负荷直接区分本应检验的机制。
- 变体 schema 的全部字段、关系和未来列在该任务 train/valid/test 间固定。不同任务可以有不同 schema，分别训练 checkpoint；不宣称同一模型跨任务零样本迁移。
- 先运行公式 oracle，验证成对输入“相同部分相同、不同部分确实导致不同目标”，才允许用来判断模型。

## 3. M1：保留同一实体的时间轨迹

### 3.1 构造

组 assets 有两个动态节点（仅 `load`），zones 为一个静态父节点；所有实体属性相同，两个 asset 归同一 zone。

基础历史：前 L-1 步恒为 `[1,3]`；origin 时甲为 `[2,4]`，乙为 `[4,2]`。逐时刻无序集合、sum、count、属性、关系与日历完全一致。每个 family 采样正尺度 a∈[0.5,2]、偏移 b∈[0,2]，把每个历史数值变成 b+a*x；同一 family 两个版本共用 a/b，a/b 不额外作为属性输入。

未来逐实体规则：

\[
x_{i,s+1}=x_{i,s}+\max(x_{i,s}-x_{i,s-1},0),\quad y_s=\sum_i x_{i,s}.
\]

没有外部未来协变量，K=0。无 mask 缺失。静态关系固定。

### 3.2 断言

- 原始基础对的下一总量分别为 8、9；生成器实际核算并保存，不手填预测。
- 将整段实体轨迹连同 ID／关系一致重排，预测不变。
- 只交换 origin 时刻数值并保持 ID 不变，代表不同轨迹；模型必须具有区分它们的参数路径。
- `no_entity_history` 在该对上的每帧不变池化表示应相同，构成结构性负对照。full 不要求任意初始化自动正确，但受控历史编码与实际小集拟合必须能够区分。

小集：train 最前 8 个 family 的成对样本，目标为全部 H。A 与 B-full 按 implementation §9.2 训练；每对预测差与真实差同号，并且每个预测值误差小于该对相应 horizon 目标差绝对值的一半，才算不再把两种情况压成一个答案。

valid/test 的同项指标完整报告；首轮不以全体泛化胜出为交付硬门槛，但必须报告成对区分率与误差，不能只报 pooled MSE。

## 4. M2：归属关系必须承载可识别信息

### 4.1 构造

assets 两个动态节点，zones 两个静态父节点，属性 `gain` 分别为 1、2；两个子节点属性相同。前 L-1 步资产 load 都为 1，origin load 为 `[1,3]`。

两个版本在 origin 前使用相同的一一归属；在 origin 时，甲保持 asset0→zone0、asset1→zone1，乙交换父级。两边每个节点的入／出度相同，属性、数值及时间相同，唯一区别是端点配对。该测试也验证 R3 支持历史最后一步的关系切换。

目标采用一步滞后关系：

\[
y_s=\sum_i x_{i,s-1}\,gain_{parent(i,s-1)}.
\]

origin 前的 load 相同，因此全部 target_history 相同；未来 asset load 固定为 origin 值，未来关系固定为 origin 关系。基础下一目标甲为 7、乙为 5。family 采样正 a∈[0.5,2] 乘全部 load，以避免仅记住一个数值组合。

第一条历史 target 所需的前驱状态取 x[-1]=x[0]、parent[-1]=parent[0]，只用于生成器初值，不作为额外模型历史；两版本采用相同初值。

### 4.2 断言

- 单看节点集合、属性集合、度数、历史 target，两个版本不可区分；加入端点关系后可区分。
- 数据层必须保留历史旧边和 origin 新边，不能把新关系回填到全部历史。
- 关系层受控消息遵循接收行／发送列定义；这是局部层测试，不宣称整个模型只沿归属传播。
- full 的前 8 个 train family 小集拟合采用 M1 的成对符号／半间距标准。
- `no_relation_all` 不可消费度数、父级属性聚合、关系质量权重等旁路；在这种成对构造下应表现为无法同时恢复两个不同答案。只移除关系消息的 `no_relation_message` 独立报告。

## 5. M3：未来计划路径与状态积累

### 5.1 构造

assets 一个动态节点、zones 一个静态父节点，关系固定。状态即已观测 load，y=x；不是隐藏在数据文件中再偷偷输入模型的变量。

\[
x_{s+1}=0.8x_s+u_{s+1}.
\]

每个 family 的历史由独立均匀计划 u∈[0,2] 和 x 初值∈[0,10] 顺序生成。历史已过去计划不用额外输入，因为历史 x 已观测。未来 u 是起点前已公布的计划，存为 `scheduled_activity`。

成对未来路径甲全为 0，乙只在第一步加 5，其余为 0；target step2 的直接协变量相同，但状态因第一步计划不同而不同。两个版本历史、属性、关系、原点完全相同。另保留未来计划随机脉冲版本作 valid/test，计划不是日历确定性函数。

基础检查以 x_origin=10 为例：用上述递推实际计算第二步目标；不得沿用旧评估中系数为 1 的示例数字。

### 5.2 断言

- direct、A 和 AR 都收到相同 `[B,H,K]`、mask 和计划发布版本，且均有从 u[:,0] 到 step2 预测的计算路径。
- 将第一步计划改成晚于 origin 才发布，数据层必须屏蔽，不能在模拟第二步时刷新计划。
- 未知 future target 的改变不能影响 forecast；已知未来计划的改变可以影响任何合法预测步，不与历史因果 mask 测试混淆。
- 共享状态的 teacher 顺序编码与一次连续编码一致；production 没有 teacher 输入。
- full 与 direct 的前 8 个 family 小集均通过拟合标准；不要求 AR 优于 direct。若 direct 看不到前缀计划，判接口失败，而不是记录为“AR 的性能优势”。

## 6. M4：规模变化与新实体

### 6.1 构造

同一任务 schema 使用 assets(load)、zones(gain) 与固定属性 capacity。训练 family 的 asset 数按编号循环取 2、4、6；valid/test 取 3、5、7。父节点数量按 family 奇偶取 1 或 2，gain 按节点取 1、2；每个 asset 用 RNG 均匀选择父节点，capacity∈[0.5,2]。

\[
x_{i,0}=capacity_i,\quad x_{i,s+1}=0.8x_{i,s}+0.2capacity_i(1+u_{s+1}),
\]
\[
y_s=\sum_i gain_{parent(i)}x_{i,s}.
\]

u 为非周期的已知 Bernoulli 计划（概率 0.5），origin 后的计划已在 origin 公布。训练也包含 whole-entity history 缺失：每 family 两个版本共享同一物理数据，一份全观测，一份屏蔽一个随机 asset 的全部历史观测，属性、关系和真实 target_history 保留；缺失不改变物理目标。

新 ID 不在训练表中，新节点数不同，但属性与关系类型一致。另构造“新实体恰在 origin 生效”的请求：此前 present=false、origin 时 present=true 且无观测，其状态按生成规则的初值 capacity 定义；target_history 在 origin 真实反映激活后的总量，不能复用旧系统 target 装作一致。

### 6.2 断言

- 相同 checkpoint 不新增参数、不重拟合 scaler，就能处理不同数量、未知 ID、新父子节点、有／无历史观测。
- 静态属性、数量、归属边都存在真实消费路径；用受控参数验证不是悄悄丢弃新节点。
- 整段重排与 ID 重命名保持数值预测；new_entity 元数据允许因训练溯源集合而变化。
- cold_start、unknown_category、orphan 标记正确。
- 训练规模与测试规模分别报告，不能以 shape 成功宣称外推精度成功。测试输出误差和随数量变化趋势，但不设硬单调性，因为当前模型没有物理单调约束。
- 新实体启动的确切未来状态通常未知，本任务刻意采用可确定初值，只证明此受控条件；真实冷启动精度仍未知。

## 7. M5：缺失、版本、质量与数值边界

不另训练模型，从 M1–M4 及独立数值夹具执行：

- 比较全观测、部分缺失、整个组近 L 步全缺失、未生效、padding；每种语义有单独断言。
- 质量测试只调用 q helper：保持每组 observed 比例，复制单元扩大规模，q 不发生指数衰减。可测试大于模型节点上限的 mask 向量，但不得据此声称大规模模型 forward 已验证。
- 一组 q=0 不删除另一组 q>0 的辅助监督；所有 q=0 时只剩主损失；整 batch 无主目标跳过，整轮无主目标报错。
- variance 从 0 到较大值扫描，`1/(variance+0.01)` 连续、有限、不大于 100；近常量维不会突然扩大百万倍。
- 统计每个训练 episode/origin 只采最后 z 一次；不重复计入窗口内部时间点，valid/test 和未来 teacher 不参与统计。
- horizon 有效数量不同，分别核算 micro 与 macro；验证选优必须用全 split macro，不平均 batch loss。
- 父级在历史中切换，逐时刻使用对应边；发布晚于 origin 的边或计划不可见。未来已知结构变化明确拒绝，不错误声称能自动模拟变化。
- 安全坏输入：ID 重复、悬空边、循环、错误单位、无穷值、未知 schema、路径越界、对象 NPZ、维度超过 limits；必须明确失败，不能静默截断或重拟合。

## 8. 运行范围、预算与判定

### 8.1 必需运行矩阵

避免盲目执行任务×所有变体的笛卡尔积，采用有针对性的矩阵：

| 任务 | 必需路线 |
| --- | --- |
| 周期 base | persistence、seasonal_naive、full 的流程 smoke；不用于证明架构优势 |
| M1 | full、no_entity_history、direct |
| M2 | full、no_relation_message、no_relation_all |
| M3 | full、direct、no_latent_loss、total_only |
| M4 | full、direct、no_spatial、total_only；同权重新实体请求 |
| M5 | helper／接口／数值测试，不训练独立模型 |

各任务的前 8 个 family 小集先做机制拟合；之后再按 implementation 的 A/B epoch 上限运行任务 train/valid/test 并报告单种子结果。机制拟合与正式任务训练使用不同 run_id，不把在小集上训练的权重当正式 test 模型。

### 8.2 候选验收阈值的边界

小集采用 implementation §9.2 的 1000 update 预算、绝对误差或相对下降标准；M1/M2 另需成对区分。它们是实现前声明的候选标准，当前未训练验证可达性。

- 失败必须区分：信息路径缺失、错误标签／mask、梯度问题、表达能力限制、优化预算不足。
- 不能为交付临时删掉成对断言，或在看到 test 后调整数据生成器让模型更容易。
- 如确定是预算／优化限制，记录证据并申请修订验收，不伪称测试通过；不强制真实业务或对照胜出。
- 同权重 oracle teacher 诊断单列，永不用于选模型、早停或对照排名。

## 9. 必需产物与证据状态

每任务输出生成器版本、schema、split-family 清单、原始包校验值、oracle 检查、resolved_config、实际训练预算、每模型 per-horizon 指标和机制断言结果。M4 单列未见节点数量及冷启动情况，M5 单列坏输入预期异常。

`suite.json` 只定义任务，运行后 `suite_result.json` 记录每任务 `passed/failed/not_run`、命令、退出码、artifact 路径；只运行部分任务时总结果不能是 passed。

当前没有生成正式任务包，没有训练模型，也没有 M1–M5 模型验收结果。独立公式探针和文档校验记录在 [review_r3.md](review_r3.md)，只证明数学示例和文档规则，不能越级。
