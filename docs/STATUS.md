# 当前状态

更新日期：2026-10-17（Asia/Shanghai）

## 当前结论

- 当前题集为120题（开发30、测试90），每主题10/30；SHA-256：`10f098142b700bbf91d40f1cbb82a49176382991cd0ed957ff92315480ca109b`。当前语料清单18份，语料正文SHA-256：`481fed36687503978bcf8a1bfb5a2c96fbaa542f5d29e3be72e0e520e1ef4f8e`。题集、来源/许可与完整知识范围仍待小组实际确认。
- 完整30题开发选参已完成：两种分词指标并列，按预先规则采用`jieba`；R3的k=4/6质量指标并列，按预先规则采用较小的k=4。候选正式配置为`jieba + top-k=4`，不是正式测试结果。
- 用户已实际核验三道开发关键样例。记录在本机私有文件`review/private/development-key-case-confirmations-20261017.json`，公开证据`evidence/private-review-digests.json`只含该私有记录哈希。原始AI评分、逐题结果和运行日志均未改写。三题复核不代表全量题目或30题结果已人工核验。
- 尚无小组签署的`review/protocol.json`，90题正式测试未启动。本地候选模板状态为pending；未填写审核人、未生成approved协议、未启动正式回答或AI评分调用。

## 启动前预检

- `uv run --locked campus-health doctor --network`：本机API密钥已配置（密钥值未输出）；DeepSeek官方`/models`返回HTTP 200，列出`deepseek-flash`；发现18份语料；题集和语料状态仍标记为待小组冻结。公开记录：`evidence/formal-preflight-doctor.public.log`；本机原件保留完整路径以匹配原时间戳清单。
- 题集结构校验无错误：`evidence/formal-preflight-question-validation.log`。
- BGE-small-zh-v1.5固定revision=`7999e1d3359715c523056ef9478215996d62a620`；本地模型文件SHA-256校验通过，CPU加载并输出512维向量：`evidence/formal-preflight-embedding.log`。
- 正式预检当时的52项测试和ruff检查通过；本次推送前对当前工作树重新检查，53项测试通过、ruff检查通过，记录分别为`evidence/push-preflight-tests.public.log`和`evidence/push-preflight-lint.public.log`。原预检日志仍保留，公开副本为`evidence/formal-preflight-tests.public.log`与`evidence/formal-preflight-lint.log`。
- 24题结果复核样本已按当前题库文件顺序预选并保存为`review/formal-review-sample.json`：每主题×题型2题，共24题；每主题1道直接题和1道无答案题双人重叠，共6题；R1/R2各15题。R1/R2仍是待实际审核人认领的占位ID，样本不表示任何正式输出已复核。
- `review/protocol.template.json`现含题集、语料正文与清单、配置、DeepSeek供应商/地址/模型、回答与评分提示词、源代码/依赖锁、Python/包版本和复核样本指纹。API密钥不写入模板。冻结代码SHA-256：`ccf0e894dee870f54376423c45e482286ca239799db304893f9d39ac3bf0ca87`。
- 指纹模板逐项比对、样本配额和密钥不入协议的检查已通过；当前结果`ready_pending_group_signoff`见`evidence/formal-preflight-freeze-check.json`。

## 正式运行仍需完成

1. 小组实际核对语料来源及许可、题集采用方式、已披露的跨集概念重叠、指标口径、审核人安排及正式配置。由两名真实学生审核人确认后，记录真实日期和审核范围；姓名、原始确认文本留在本机私有映射文件中。
2. 价格会在正式运行前再次核对DeepSeek官方页面。当前预计90×4=360条结果、270次回答调用及360次AI评分调用；开发期粗估费用约¥1.30—¥2.60，不含重试且不是实际扣费。
3. 通过小组签署与新鲜预检后，正式运行固定`B0/R1/R2/R3`，保留完整请求/响应、用量、重试、错误、汇总和真实运行截图。只有360条输出身份齐全且运行/评分错误均为0，才报告完整通过；任何失败都保留并明确标记。
4. 正式结果产生后，按上述24题样本复核四种方法输出，记录6题双人分歧；该结果复核尚未开始。Windows队友电脑的实际API全链路验收也尚未完成。

## 时间证据与历史范围

- 既有RFC 3161证明`evidence/timestamps/sha256-3762874859e549d9-ee5c1791/`绑定82份文件、其中14张截图；TSA签发时间为2026-10-17 01:12:44 UTC。其签名验证通过，证明清单内容最迟当时已存在，不证明作者身份或内容正确。本轮状态、候选协议、预检日志和对应代码另有RFC 3161证明：`evidence/timestamps/sha256-d46bc8f737767e91-d3337362/`，TSA签发于2026-10-17 01:44:11 UTC。本机原始日志保留在本地以核对原清单；公开仓库日志只把本机项目根路径替换为`<PROJECT_ROOT>`，并以新的清单单独固定。私有复核原文不上传，公开摘要文件的SHA-256已纳入清单。该证明不表示正式90题实验已执行。
- 本轮没有启动正式90题运行，因此没有伪造正式运行截图。已有开发截图及运行日志保留在`evidence/`和`runs/`；正式阶段产生新截图后再与运行日志、汇总一并加时间戳。

## 其他已完成工作与剩余交付

- 项目采用Python 3.11、uv、Streamlit，本地CPU检索和DeepSeek官方API；四种检索路线及Mac演示环境已完成开发验证。公开仓库为`Harry6400/campus-health-rag`。
- 资料与题目经历AI初核、用户逐项修订和少量人工核验；不能称120题均已人工或专家复核。结果分析须继续区分事实忠实、检索遗漏、错误拒答、API/评分失败。
- 尚待正式结果的24题人工复核、至少3类真实错误归因、Windows个人电脑验收、报告PDF、≤2分钟录屏和最终≤50MB ZIP。未验收事项不写完成。

## 已记录的问题

- uv默认缓存位于不可写路径；使用项目内`.cache/uv`，未改全局设置。
- 部分来源存在证书链错误、超时或错误页；拒绝把错误页面纳入语料，保留失败和修复记录。
- AI题目曾超量、重复或把目标子集缺失误当全库缺失；按配额核对并保留原始输出与修订记录。
- AI评委曾出现评分数组格式错误；保留失败响应，不补假分数。
- Windows默认cp1252曾导致中文CLI帮助异常；已修复并通过旧编码回归检查。
