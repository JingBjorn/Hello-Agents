# AutoGen + Responses 网关

当前案例默认通过本地协议中转调用只支持 `/v1/responses` 的网关：

```text
AutoGen OpenAIChatCompletionClient
    → http://127.0.0.1:<随机端口>/v1/chat/completions
    → OPENAI_BASE_URL/responses
```

中转由 `client.py` 自动启动，退出时自动关闭，无须额外终端。
`.env` 中的 `OPENAI_BASE_URL` 仍是上游 API 根地址（通常以 `/v1` 结尾），
**不要改成 localhost，也不要追加 `/responses`**。
`OPENAI_API_KEY`、`OPENAI_MODEL_NAME` 沿用现有配置。不会读取 Codex 登录文件。

## 运行

在项目根目录、已激活 `.venv` 的终端执行：

```bash
# 先验证连通性：发送短消息，空响应可能重试，可能产生 API 费用。
python chapter6/AutoGen/client.py --check

# 启动团队；用户代理等待输入时，输入 TERMINATE 可结束。
python chapter6/AutoGen/client.py

# 离线测试：不访问真实网关，不消耗模型额度。
python -m unittest discover -s chapter6/AutoGen -p 'test_responses_proxy.py' -v
```

如使用一个本身支持 Chat Completions 的服务，可在 `.env` 中设置
`AUTOGEN_API_MODE=chat_completions`，跳过本地中转。默认值为 `responses`。
上游单次读写超时默认 180 秒，可用 `AUTOGEN_PROXY_TIMEOUT` 调整。
禁用 SDK 自动重试。中转只对已收到完整响应、但没有可用文本的情况默认重试
1 次，不重试认证、限流、连接错误、内容过滤或不完整 SSE；重试可能产生额外 API 费用。
成功后的 token 统计包含空响应重试的已知用量。

以下设置仅作用于 Responses 中转，无须修改现有 `.env` 即可使用默认值：

```dotenv
# 推理与最终文本共用的上游输出预算；显式请求参数优先
AUTOGEN_MAX_OUTPUT_TOKENS=32768
# 0 禁用；允许 0～3，重试沿用原请求，不擅自突破输出预算
AUTOGEN_EMPTY_RESPONSE_RETRIES=1
# 可选：仅在模型和网关支持时设置，不默认强加给所有模型
# AUTOGEN_REASONING_EFFORT=low
```

下游客户端的等待超时随重试次数增加，避免仍按单次请求的时长提前断开。
预算是上限，不是固定消耗；若模型/网关不支持该上限，请按其限制调整。

## 支持范围与安全边界

- 适配当前案例的纯文本多轮消息、system/developer 指令、角色文本前缀、
  token 用量、截断状态及上游错误。
- 智能体内部标识使用 `product_manager`、`engineer`、`code_reviewer`、
  `user_proxy`，以满足 AutoGen 的名字校验；中文职责和提示词保持不变。
- 上游使用 Responses SSE，并设置 `store=false`；也能解析返回 JSON 的兼容网关。
  兼容实际返回 SSE、但 `Content-Type` 被网关标为 `text/plain` 的情况。
  如果终止事件省略文本，会从此前的文本增量、文本完成或输出项完成事件恢复，
  不重复拼接全文，也不会把内部推理当作回复。
- 下游支持非流式请求与**缓冲式 SSE**：必须等上游生成结束才返回，
  不是逐 token 实时流。`Console(team.run_stream(...))` 的团队事件输出仍可使用。
- 不支持工具调用、图片/音频、JSON Schema、多个候选回复等功能；
  不支持的参数会明确报错，不会静默丢弃。
- 网关对 `temperature`、输出长度等可选参数的支持程度仍以网关为准。
- 仅监听 `127.0.0.1` 随机端口，使用随机临时 Bearer 密钥。
  真实上游密钥不会交给 AutoGen 本地客户端，日志不输出消息正文或密钥。
  不应将此教学用中转暴露到公网。
- 不跨请求共享 Responses 会话；每次传递 AutoGen 提供的完整上下文。

## 常见错误

- `unsupported_endpoint`：检查运行的是这里修改后的 `client.py`，
  并确认未设置 `AUTOGEN_API_MODE=chat_completions`。
- 401/403：检查上游 Key 和模型权限；本地中转不会赋予额外权限。
- 504：上游响应超时，可适当提高 `AUTOGEN_PROXY_TIMEOUT`。
- `incomplete_stream`：网关没有发送 Responses 终止事件，不能视为成功。
- `empty_response`：即使恢复 SSE 文本并进行有限重试后，上游仍没有可用文本。
  若明确提示输出预算耗尽，可提高 `AUTOGEN_MAX_OUTPUT_TOKENS`，
  或在支持的模型上设置 `AUTOGEN_REASONING_EFFORT=low`；其他空输出需要排查网关。
  该错误之后的 `Unhandled message ... GroupChatError` 是 AutoGen 传播原始错误时
  产生的次生异常，并非用户输入“可以”有问题。不要用空字符串假装请求成功。
