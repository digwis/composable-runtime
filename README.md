# Composable Runtime — 可组合能力编排引擎

> **软件 = 一组可组合的能力，运行在统一的运行时上，通过消息协作，并由 AI 或开发者按需编排。**

## 架构

```
┌─────────────────────────────────────┐
│         编排层 (Orchestrator)        │
│   Workflow YAML / AI Agent / CLI     │
├─────────────────────────────────────┤
│       消息总线 (Message Bus)          │
│   路由 · 日志 · 审计                  │
├─────────────────────────────────────┤
│     能力层 (Capabilities)             │
│  ┌───────┐ ┌────────┐ ┌───────┐     │
│  │ greet │ │compute │ │ store │ ... │
│  └───────┘ └────────┘ └───────┘     │
├─────────────────────────────────────┤
│       统一运行时 (Runtime)            │
│  Capability Trait · Message · Bus    │
└─────────────────────────────────────┘
```

## 核心概念

| 概念 | 实现 | 说明 |
|------|------|------|
| **能力 (Capability)** | `Capability` trait | 可组合的软件单元，声明动作并通过消息交互 |
| **运行时 (Runtime)** | `MessageBus` + `Registry` | 统一的执行环境，负责能力注册和消息路由 |
| **消息 (Message)** | `Message` struct | 能力间通信的统一协议，包含来源/目标/动作/负载 |
| **编排 (Workflow)** | `Workflow` + `Orchestrator` | 按步骤编排能力，支持变量引用、条件分支、并行组、重试和错误处理 |
| **AI 编排** | `execute_dynamic` / `execute_json` | AI Agent 可动态构建步骤并执行，无需预定义工作流 |
| **自省** | `introspect()` | 运行时能力发现，列出所有能力及其动作 |

## 快速开始

### 编译

```bash
cargo build --release
```

### 运行工作流

```bash
# 问候并存储
cargo run --release -- run -w examples/greet_and_store.yaml -v

# 计算流水线
cargo run --release -- run -w examples/compute_pipeline.yaml -v

# 条件工作流
cargo run --release -- run -w examples/conditional_workflow.yaml -v

# 并行工作流
cargo run --release -- run -w examples/parallel_workflow.yaml -v

# 重试与错误处理
cargo run --release -- run -w examples/retry_and_error.yaml -v
```

### 直接发送消息

```bash
cargo run --release -- send --to greet --action hello --payload '{"name": "张三", "greeting": "你好"}'
```

### 列出已注册能力

```bash
cargo run --release -- list
```

### 能力自省

```bash
cargo run --release -- introspect
```

### 动态执行（模拟 AI Agent 调用）

```bash
cargo run --release -- exec -j '{"name":"dyn","capability":"compute","action":"multiply","input":{"a":6,"b":7}}'
```

## 编写自定义能力

```rust
use runtime::{Capability, Message, MessageResult};

pub struct MyCapability;

#[async_trait::async_trait]
impl Capability for MyCapability {
    fn name(&self) -> &str { "my-cap" }
    fn version(&self) -> &str { "0.1.0" }
    fn actions(&self) -> Vec<&str> { vec!["do-thing"] }

    async fn handle(&self, msg: &Message) -> MessageResult {
        match msg.action.as_str() {
            "do-thing" => {
                // 处理逻辑...
                Ok(Message::builder()
                    .from("my-cap")
                    .to(msg.from.as_deref().unwrap_or("orchestrator"))
                    .action("do-thing.response")
                    .payload(serde_json::json!({"status": "done"}))
                    .build())
            }
            _ => Err(runtime::MessageError::UnsupportedAction {
                capability: "my-cap".into(),
                action: msg.action.clone(),
            }),
        }
    }
}
```

注册到运行时：

```rust
let bus = RegistryBuilder::new()
    .with(GreetCapability)
    .with(MyCapability)  // 添加你的能力
    .build();
```

## 编写工作流

```yaml
name: my-workflow
description: 我的工作流

steps:
  - name: step-1
    capability: greet
    action: hello
    input:
      name: "世界"

  - name: step-2
    capability: store
    action: set
    input:
      key: "result"
      value: "${step-1.message}"    # 引用上一步输出
    condition: "step-1.message != null"  # 条件分支

  # 并行执行组
  - name: parallel-tasks
    parallel:
      - name: task-a
        capability: compute
        action: add
        input: { a: 1, b: 2 }
      - name: task-b
        capability: greet
        action: hello
        input: { name: "并行" }

  # 带重试和错误处理的步骤
  - name: risky-step
    capability: compute
    action: divide
    input: { a: 100, b: 0 }
    retry:
      max_retries: 3
      delay_ms: 100
      backoff_multiplier: 2.0
    on_error: continue    # stop | continue | record
```

### 变量引用

- `${step-name}` — 引用整步输出
- `${step-name.field}` — 引用输出中的字段
- 支持嵌套对象和数组索引：`${step-name.data.0.name}`

### 条件表达式

- `field == "value"` — 相等
- `field != null` — 非空判断
- `field == true` — 布尔判断
- `field > 5` / `>=` / `<` / `<=` — 数值比较

### 并行执行组

使用 `parallel` 关键字定义并行组，组内所有步骤同时执行：

```yaml
- name: parallel-group
  parallel:
    - name: task-a
      capability: compute
      action: add
      input: { a: 1, b: 2 }
    - name: task-b
      capability: greet
      action: hello
      input: { name: "并行" }
```

### 重试策略

```yaml
retry:
  max_retries: 3          # 最大重试次数
  delay_ms: 100           # 初始延迟（毫秒）
  backoff_multiplier: 2.0 # 指数退避倍数
```

### 错误处理策略

- `stop` — 出错即停止工作流（默认）
- `continue` — 出错后跳过该步骤，继续执行
- `record` — 出错后将错误存入上下文，继续执行

### 超时

```yaml
timeout_ms: 5000   # 步骤超时（毫秒）
```

## AI Agent 编排接口

Orchestrator 提供动态执行接口，AI Agent 可根据上一步结果决定下一步执行什么：

```rust
// 动态构建步骤并执行
let step = Step::new("ai-step", "compute", "add", json!({"a": 1, "b": 2}));
let (output, retries, failed) = orchestrator.execute_dynamic(&step, &context).await;

// 从 JSON 指令直接执行（适用于 LLM 返回 JSON 的场景）
let json = json!({"name":"ai-step","capability":"compute","action":"add","input":{"a":1,"b":2}});
let (output, _, _) = orchestrator.execute_json(json, &context).await?;
```

## AI Agent — LLM 驱动的自我进化编排

Agent 模块实现了完整的 **Plan-Execute-Observe** 循环，接入 Anthropic Claude API：

```
┌──────────────────────────────────────────┐
│              AI Agent 循环                │
│                                          │
│  1. 自省 → 获取所有能力及其动作           │
│  2. 规划 → LLM 根据任务+能力生成步骤      │
│  3. 执行 → Orchestrator 逐步执行          │
│  4. 观察 → 将结果反馈给 LLM               │
│  5. 适应 → LLM 决定继续/调整/完成         │
│                                          │
│  🧠 自我进化:                             │
│  • 成功工作流自动保存为模板（强化学习）    │
│  • 失败记录供 LLM 避免重复错误            │
│  • 下次类似任务可复用已有模板             │
└──────────────────────────────────────────┘
```

### 使用方式

```bash
# 设置 API Key
export ANTHROPIC_API_KEY=sk-ant-...

# 执行自然语言任务
cargo run --release -- agent -t "计算 3+5 并将结果存入存储"

# 自定义模型和迭代次数
cargo run --release -- agent -t "问候张三并存储问候语" --max-iterations 5 --model claude-sonnet-4-20250514

# 使用代理
cargo run --release -- agent -t "..." --base-url https://your-proxy.com
```

### Agent 工作流程

1. **自省**：调用 `introspect()` 获取所有已注册能力（名称、动作、描述）
2. **记忆检索**：检查 `AgentMemory` 中是否有类似任务的成功模板
3. **LLM 规划**：将能力清单 + 记忆 + 任务发送给 Claude，获取 JSON 格式的步骤计划
4. **逐步执行**：通过 `execute_json()` 执行每个步骤，结果存入上下文
5. **观察反馈**：将执行结果反馈给 LLM，LLM 决定下一步
6. **自我进化**：
   - 成功 → 保存工作流模板，成功次数 +1（强化学习）
   - 失败 → 记录错误，LLM 下次可参考避免

### 编程接口

```rust
use runtime::{Agent, OrchestratorBuilder, RegistryBuilder};

let bus = RegistryBuilder::new()
    .with(GreetCapability)
    .with(ComputeCapability)
    .with(StoreCapability::new())
    .build();

let orchestrator = OrchestratorBuilder::new()
    .with_bus(bus)
    .build();

let mut agent = Agent::new(orchestrator, "sk-ant-...")
    .with_max_iterations(10)
    .with_model("claude-sonnet-4-20250514");

let result = agent.run("计算 3+5 并存储结果").await?;

// 查看记忆
let memory = agent.memory();
for w in &memory.successful_workflows {
    println!("已学习: '{}' (成功 {} 次)", w.task, w.success_count);
}
```

## 能力自省

```rust
let caps = orchestrator.introspect().await;
// 返回所有能力的名称、版本、动作列表和描述
```

## WIT 接口契约

`wit/capabilities.wit` 定义了所有能力的接口契约，
未来可通过 WASM 组件模型实现跨语言能力组合。

## 项目结构

```
new/
├── Cargo.toml                      # Workspace
├── crates/
│   ├── runtime/                    # 统一运行时
│   │   ├── src/
│   │   │   ├── lib.rs              # 模块导出
│   │   │   ├── capability.rs       # Capability trait
│   │   │   ├── message.rs          # Message 定义
│   │   │   ├── message_bus.rs      # 消息总线
│   │   │   ├── registry.rs         # 能力注册中心
│   │   │   ├── orchestrator.rs     # 编排引擎
│   │   │   ├── workflow.rs         # 工作流定义
│   │   │   ├── agent.rs            # AI Agent (Plan-Execute-Observe)
│   │   │   ├── genome.rs           # 能力基因组 (DNA 驱动)
│   │   │   ├── evolution.rs        # 进化引擎
│   │   │   ├── auto_evolve.rs      # 自主进化循环
│   │   │   ├── meta_evolve.rs      # 元进化 (变异执行器本身)
│   │   │   ├── autonomous.rs       # 自主运行时 (感知→目标→执行)
│   │   │   ├── ab_test.rs          # A/B 测试
│   │   │   ├── failure_driver.rs   # 失败驱动进化
│   │   │   ├── daemon.rs           # 系统级常驻服务
│   │   │   ├── mcp_server.rs       # MCP Server (JSON-RPC)
│   │   │   ├── memory.rs           # 持久化记忆
│   │   │   ├── sandbox.rs          # 沙箱隔离
│   │   │   └── platform.rs         # 平台检测
│   │   └── tests/
│   │       └── evolution_e2e.rs    # 端到端测试 (10 个)
│   └── capabilities/               # 原生能力
│       └── src/
│           ├── greet.rs            # 问候能力
│           ├── compute.rs          # 计算能力
│           ├── store.rs            # 存储能力
│           ├── fs.rs               # 文件系统能力
│           ├── shell.rs            # Shell 命令能力
│           ├── http.rs             # HTTP 请求能力
│           ├── code.rs             # 代码分析能力
│           └── web.rs              # Web 搜索能力
├── bin/orchestrator/               # CLI 入口 (12 个子命令)
├── wit/                            # WIT 接口定义
├── .evolution/                     # 进化数据 (genomes.json)
└── examples/                       # 示例工作流
    ├── greet_and_store.yaml
    ├── compute_pipeline.yaml
    ├── conditional_workflow.yaml
    ├── parallel_workflow.yaml
    └── retry_and_error.yaml
```

## 路线图

- [x] Capability trait + Message 协议
- [x] MessageBus 消息路由
- [x] Workflow YAML 编排
- [x] 变量引用 + 条件分支（支持 `>` `>=` `<` `<=`）
- [x] 并行步骤执行（Parallel Groups）
- [x] 步骤重试 + 指数退避
- [x] 错误处理策略（stop / continue / record）
- [x] 步骤超时
- [x] AI Agent 动态编排接口（`execute_dynamic` / `execute_json`）
- [x] AI Agent LLM 驱动编排（Plan-Execute-Observe 循环，接入 Claude API）
- [x] 自我进化与学习（工作流模板记忆 + 失败记录 + 强化学习）
- [x] 能力自省（Introspection）
- [x] 能力基因组（DNA 驱动，LLM/规则/组合/脚本/原生/自定义 6 种实现）
- [x] 自主进化循环（自省→归因→变异→测试→选择）
- [x] 持续进化 & 定向进化模式
- [x] A/B 测试（变异体对比 + 自动晋升/回滚）
- [x] 失败驱动进化（能力缺口发现 + 自动填补）
- [x] 元进化（变异执行器本身，Python/Rust-WASM/Rust-Native）
- [x] MCP Server（18 个原子 tool，stdio JSON-RPC）
- [x] Daemon 模式（Unix socket + PATH 注入 + 后台进化）
- [x] 自主运行时（环境感知→目标生成→主动执行）
- [x] 沙箱隔离（超时 + 安全检查）
- [x] 持久化记忆（磁盘存储工作流模板 + 失败记录）
- [ ] WASM 组件加载（wasmtime）
- [ ] 能力热加载/卸载（原生插件已支持，WASM 待实现）
- [ ] 分布式消息总线（跨进程/跨节点）
