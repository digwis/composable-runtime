# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- GitHub Actions CI (build + test + clippy + fmt, Ubuntu/macOS 矩阵)
- LICENSE (MIT)
- CHANGELOG.md
- 217 个单元测试（覆盖全部 20 个 runtime 模块 + 3 个 capabilities 模块）
- E2E 进化系统测试（10 个场景）

### Changed
- 消除全部 42 个 Clippy 警告
- 消除全部编译器警告
- 代码统一格式化 (cargo fmt)
- 重构 main.rs 重复基因组注册逻辑为 register_genomes 函数
- 更新 README 项目结构和路线图

### Fixed
- E2E 测试编译失败（CapabilityGenome 缺少 test_suite 字段）
- CompositeStep 初始化缺少 name 字段
- 多处未使用变量和字段警告
- auto_evolve.rs 中 if 分支重复代码
- autonomous.rs / memory.rs 中 Default::default() 后字段赋值

## [0.1.0] - 2025-01-01

### Added
- 核心运行时 (runtime crate)
  - 消息总线 (MessageBus)
  - 编排引擎 (Orchestrator)
  - 工作流定义 (Workflow)
  - 能力基因组 (CapabilityGenome)
  - 进化引擎 (EvolutionEngine)
  - 自动进化器 (AutoEvolver)
  - 元进化器 (MetaEvolver)
  - 失败驱动进化 (FailureDriver)
  - A/B 测试 (ABTester)
  - 沙箱验证 (Sandbox)
  - 多层记忆系统 (ShortTermMemory / LongTermMemory / PersistentMemory)
  - AI Agent (LLM 驱动编排)
  - 自主运行时 (AutonomousRuntime)
  - Daemon 服务 (Unix socket)
  - MCP Server (JSON-RPC)
  - 平台检测 (Platform)
  - 能力注册中心 (RegistryBuilder)
- 原生能力 (capabilities crate)
  - greet — 问候
  - compute — 数学运算
  - shell — 系统命令执行
  - fs — 文件系统操作
  - http — HTTP 请求
  - store — 键值存储
  - code — 代码执行
  - web — 网页抓取
- CLI 编排器 (orchestrator bin)
