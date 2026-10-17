# 课程项目证据的可信时间戳

## 它能证明什么

`tools/timestamp_evidence.py` 使用 RFC 3161 时间戳服务为证据清单的 SHA-256 摘要申请 TSA 签名。清单逐项记录证据文件的相对路径、字节数和 SHA-256。核验时会重新计算每个文件的摘要，并验证 TSA 签名、信任链、原始请求 nonce 和清单摘要。

通过核验的时间戳证明：清单所列的这些文件内容最迟在 TSA 签发的 UTC 时间已经存在。它不证明文件内容真实、由谁完成，也不能把历史文件补成过去的时间证明。第一次对历史材料申请时间戳，只能证明这些材料在**这次申请时**已经存在。目录名、文件修改时间和本机时钟都不作为可信时间来源。

服务端只收到时间戳请求中的清单摘要；源文件不会上传。证明目录保存原始清单、RFC 3161 请求和签名响应，以及用于离线验证的证书副本。

## 为证据申请时间戳

在项目根目录运行。每个阶段完成后，对该阶段的状态说明、真实运行输出和截图一起申请时间戳：

```sh
uv run --locked python tools/timestamp_evidence.py stamp \
  --label "完整开发选参与阶段记录" \
  docs/STATUS.md docs/ACCEPTANCE.md \
  evidence/development-full-summary.md \
  evidence/development-sweep-console.log \
  evidence/development-comparison.csv \
  evidence/development-data-ready.png \
  evidence/development-bm25-comparison.png \
  evidence/development-r3-k2.png evidence/development-r3-k4.png evidence/development-r3-k6.png \
  runs/20261016T161906918893Z_parameter-sweep-dev \
  runs/20261016T161906969857Z_exploratory-dev \
  runs/20261016T161911089470Z_exploratory-dev \
  runs/20261016T161911255389Z_exploratory-dev \
  runs/20261016T162032051879Z_exploratory-dev \
  runs/20261016T162159622552Z_exploratory-dev
```

输出在 `evidence/timestamps/`，目录名按清单摘要命名以便定位；真正的签发时间在 `timestamp-info.txt` 的 TSA 响应内。每次申请会生成唯一清单，因此同一文件集合也可以在不同阶段再次留证。

## 离线核验

保留项目文件和时间戳证明目录，并在项目根目录运行：

```sh
uv run --locked python tools/timestamp_evidence.py verify evidence/timestamps/<证明目录名>
```

看到 `Verification: OK` 且提示源文件均匹配，才表示当前文件与 TSA 签发时的清单一致。也可直接用 OpenSSL 对清单与签名响应核验：

```sh
openssl ts -verify \
  -in evidence/timestamps/<证明目录名>/response.tsr \
  -data evidence/timestamps/<证明目录名>/manifest.json \
  -CAfile evidence/timestamps/<证明目录名>/cacert.pem \
  -untrusted evidence/timestamps/<证明目录名>/tsa.crt
```

课程提交时，应把证明目录和清单列出的原文件一起打包，保留相对目录结构；否则老师无法核对文件摘要。`runs/` 被 Git 忽略，需在提交包中显式包含要展示的原始运行证据。

公开仓库若要发布含本机绝对路径的日志，应保留原件供本机原时间戳验证，并发布将项目根路径替换为`<PROJECT_ROOT>`的`*.public.log`副本。公开副本须单独申请时间戳；原清单仍对应本机原件，不能用来声称公开副本与原件字节相同。

## 服务与边界

本项目使用 FreeTSA 的 RFC 3161 公共 TSA。脚本固定核对 FreeTSA 官网公布的 CA 与 TSA 证书 SHA-256 指纹；若证书轮换，脚本会停止，需先人工查验官网指纹。FreeTSA 公共服务按最佳努力提供，没有可用性 SLA；时间戳属于外部服务签名，不等同于学校、政府或合格电子认证机构出具的证明。若课程要求指定服务，应按教师要求替换 TSA。

Mac 上已使用 OpenSSL 3.6.2。其他电脑需 Python 3.11 和 OpenSSL 3.x，并应单独实际核验；目前不宣称 Windows 端已经验证。

标准依据：[RFC 3161](https://www.rfc-editor.org/rfc/rfc3161.html)。服务地址、调用方式和证书指纹见 [FreeTSA 使用说明](https://freetsa.org/index_en.php)；其服务限制见 [FreeTSA 使用政策](https://freetsa.org/freetsa.html)。
