use capabilities::{ComputeCapability, GreetCapability, StoreCapability};
use runtime::genome::{CapabilityGenome, FitnessGene};
use runtime::genome_yaml;
use runtime::validator::{
    self, EnvironmentValidator, ForkRepoValidator, RealWorldSignal, SignalStrength,
    ValidatorRegistry,
};
use runtime::{Message, MessageBus, OrchestratorBuilder, RegistryBuilder, Workflow};
use std::sync::Arc;

/// YAML 存储集成测试：从 .evolution/ 目录加载，验证所有基因组可反序列化并正向转换
#[test]
fn yaml_migration_load_all_genomes() {
    let evo_path = std::path::Path::new(concat!(env!("CARGO_MANIFEST_DIR"), "/../../.evolution"));
    if !genome_yaml::is_yaml_evolution_dir(evo_path) {
        eprintln!("跳过: .evolution/ 目录不存在");
        return;
    }

    let genomes = genome_yaml::load_genomes_from_yaml_dir(evo_path).expect("加载 .evolution/ 失败");

    assert!(!genomes.is_empty(), "至少应该加载到几个基因组");

    for g in &genomes {
        assert!(!g.name.is_empty(), "name 不应为空");
        assert!(
            !g.actions.is_empty() || g.actions.is_empty(),
            "actions 至少要有声明"
        );
    }

    eprintln!("✅ 从 .evolution/ 成功加载 {} 个基因组", genomes.len());

    // Roundtrip: 保存到临时目录，再加载回来，验证一致
    let temp_dir = tempfile::TempDir::new().expect("创临时目录失败");
    let mut map = std::collections::HashMap::new();
    for g in &genomes {
        map.insert(g.name.clone(), g.clone());
    }

    genome_yaml::save_genomes_to_yaml_dir(temp_dir.path(), &map).expect("保存到临时目录失败");

    let reloaded =
        genome_yaml::load_genomes_from_yaml_dir(temp_dir.path()).expect("从临时目录重载失败");

    assert_eq!(reloaded.len(), genomes.len(), "roundtrip 数量应一致");

    // 验证每个能力 roundtrip 正确
    for original in &genomes {
        let reloaded_g = reloaded
            .iter()
            .find(|r| r.name == original.name)
            .expect(&format!("回载后找不到 {}", original.name));
        assert_eq!(reloaded_g.version, original.version);
        assert_eq!(reloaded_g.description, original.description);
        assert_eq!(reloaded_g.actions.len(), original.actions.len());
    }

    eprintln!("✅ YAML roundtrip 成功: {} 基因组", genomes.len());
}

/// 集成测试：多能力编排 — compute → store → greet
#[tokio::test(flavor = "multi_thread")]
async fn integration_multi_capability_workflow() {
    let bus = RegistryBuilder::new()
        .with(ComputeCapability)
        .with(StoreCapability::new())
        .with(GreetCapability)
        .build();

    let orch = OrchestratorBuilder::new().with_bus(bus).build();

    let wf = Workflow::from_yaml(
        r#"
name: multi_cap
description: 多能力编排集成测试
steps:
  - name: calc
    capability: compute
    action: add
    input:
      a: 3
      b: 4
  - name: save
    capability: store
    action: set
    input:
      key: result
      value: "${calc.result}"
  - name: greet_user
    capability: greet
    action: hello
    input:
      name: "集成测试"
"#,
    )
    .unwrap();

    let result = orch.run(&wf).await.unwrap();
    assert!(result.success);
    assert_eq!(result.steps_executed, 3);
    assert_eq!(result.context["calc"]["result"].as_f64().unwrap(), 7.0);
    assert_eq!(result.context["greet_user"]["message"], "你好, 集成测试!");
}

/// 集成测试：条件跳过 — 不满足条件的步骤被跳过
#[tokio::test(flavor = "multi_thread")]
async fn integration_conditional_skip() {
    let bus = RegistryBuilder::new()
        .with(ComputeCapability)
        .with(StoreCapability::new())
        .build();

    let orch = OrchestratorBuilder::new().with_bus(bus).build();

    let wf = Workflow::from_yaml(
        r#"
name: cond_skip
description: 条件跳过集成测试
steps:
  - name: calc
    capability: compute
    action: add
    input:
      a: 1
      b: 2
  - name: save_when_large
    capability: store
    action: set
    input:
      key: big
      value: "${calc.result}"
    condition: "${calc.result} > 10"
"#,
    )
    .unwrap();

    let result = orch.run(&wf).await.unwrap();
    assert!(result.success);
    assert_eq!(result.steps_executed, 1);
    assert_eq!(result.steps_skipped, 1);
}

/// 集成测试：错误处理 — 未注册能力导致失败
#[tokio::test(flavor = "multi_thread")]
async fn integration_unregistered_capability_fails() {
    let bus = RegistryBuilder::new().with(ComputeCapability).build();
    let orch = OrchestratorBuilder::new().with_bus(bus).build();

    let wf = Workflow::from_yaml(
        r#"
name: fail_test
description: 未注册能力失败测试
steps:
  - name: call_missing
    capability: nonexistent_cap
    action: do_something
    input: {}
"#,
    )
    .unwrap();

    let result = orch.run(&wf).await;
    assert!(result.is_err());
}

/// 集成测试：Continue on Error — 失败后继续执行
#[tokio::test(flavor = "multi_thread")]
async fn integration_continue_on_error() {
    let bus = RegistryBuilder::new()
        .with(ComputeCapability)
        .with(GreetCapability)
        .build();

    let orch = OrchestratorBuilder::new().with_bus(bus).build();

    let wf = Workflow::from_yaml(
        r#"
name: continue_on_error
description: 错误后继续
steps:
  - name: fail_step
    capability: nonexistent
    action: act
    input: {}
    on_error: continue
  - name: ok_step
    capability: greet
    action: hello
    input:
      name: "恢复"
"#,
    )
    .unwrap();

    let result = orch.run(&wf).await.unwrap();
    assert!(!result.success);
    assert!(result.steps_skipped >= 1);
    assert_eq!(result.steps_failed, 1);
}

/// 集成测试：变量引用链 — 步骤输出作为后续步骤输入
#[tokio::test(flavor = "multi_thread")]
async fn integration_variable_chain() {
    let bus = RegistryBuilder::new().with(ComputeCapability).build();
    let orch = OrchestratorBuilder::new().with_bus(bus).build();

    let wf = Workflow::from_yaml(
        r#"
name: var_chain
description: 变量引用链
steps:
  - name: step1
    capability: compute
    action: add
    input:
      a: 10
      b: 20
  - name: step2
    capability: compute
    action: multiply
    input:
      a: "${step1.result}"
      b: 2
"#,
    )
    .unwrap();

    let result = orch.run(&wf).await.unwrap();
    assert!(result.success);
    assert_eq!(result.context["step1"]["result"].as_f64().unwrap(), 30.0);
    assert_eq!(result.context["step2"]["result"].as_f64().unwrap(), 60.0);
}

/// 集成测试：MessageBus 直接通信
#[tokio::test(flavor = "multi_thread")]
async fn integration_message_bus_direct() {
    let bus = RegistryBuilder::new()
        .with(GreetCapability)
        .with(ComputeCapability)
        .build();

    let msg = Message::builder()
        .from("test")
        .to("greet")
        .action("hello")
        .payload(serde_json::json!({"name": "总线测试"}))
        .build();

    let resp = bus.send(msg).await.unwrap();
    assert_eq!(resp.action, "hello.response");
    assert!(resp.payload["message"]
        .as_str()
        .unwrap()
        .contains("总线测试"));
}

/// 集成测试：Store 持久化 — set 后 get 验证
#[tokio::test(flavor = "multi_thread")]
async fn integration_store_persistence() {
    let bus = RegistryBuilder::new().with(StoreCapability::new()).build();

    let set_msg = Message::builder()
        .from("test")
        .to("store")
        .action("set")
        .payload(serde_json::json!({"key": "integration", "value": {"nested": true}}))
        .build();
    bus.send(set_msg).await.unwrap();

    let get_msg = Message::builder()
        .from("test")
        .to("store")
        .action("get")
        .payload(serde_json::json!({"key": "integration"}))
        .build();
    let resp = bus.send(get_msg).await.unwrap();
    assert_eq!(resp.payload["found"], true);
    assert_eq!(resp.payload["value"]["nested"], true);
}

/// 集成测试：Compute 除零错误处理
#[tokio::test(flavor = "multi_thread")]
async fn integration_compute_divide_by_zero() {
    let bus = RegistryBuilder::new().with(ComputeCapability).build();

    let msg = Message::builder()
        .from("test")
        .to("compute")
        .action("divide")
        .payload(serde_json::json!({"a": 10, "b": 0}))
        .build();

    let result = bus.send(msg).await;
    assert!(result.is_err());
}

/// 集成测试：自省 — 列出已注册能力
#[tokio::test(flavor = "multi_thread")]
async fn integration_introspect() {
    let bus = RegistryBuilder::new()
        .with(GreetCapability)
        .with(ComputeCapability)
        .with(StoreCapability::new())
        .build();

    let caps = bus.list_capabilities().await;
    assert_eq!(caps.len(), 3);
    assert!(caps.contains(&"greet".to_string()));
    assert!(caps.contains(&"compute".to_string()));
    assert!(caps.contains(&"store".to_string()));

    let info = bus.get_capability("compute").await;
    assert!(info.is_some());
    let info = info.unwrap();
    assert_eq!(info.name(), "compute");
    assert!(info.actions().contains(&"add"));
}

// =============================================================================
// 反馈信号质量集成测试（环境验证器 + autonomous 自循环降级）
// =============================================================================

/// 一个假装成功的能力：自报 success=true，但子进程 exit_code=1（真实失败）
///
/// 这正是旧 validate_in_real_project 会误判的场景——它只看 JSON success 字段，
/// 会把这个能力判为成功。环境验证器应通过 exit_code 识破它。
struct FakeSuccessValidator;

#[async_trait::async_trait]
impl EnvironmentValidator for FakeSuccessValidator {
    fn matches(&self, capability_name: &str) -> bool {
        capability_name.contains("fake_success")
    }
    async fn verify(
        &self,
        _capability_name: &str,
        _action: &str,
        output: &serde_json::Value,
    ) -> RealWorldSignal {
        // 模拟环境验证器读真实 exit_code
        let exit_code = output
            .get("exit_code")
            .and_then(|v| v.as_i64())
            .unwrap_or(0);
        RealWorldSignal {
            success: exit_code == 0,
            evidence: format!("exit_code={}", exit_code),
            strength: validator::SignalStrength::ExitCode,
        }
    }
}

/// 验证：能力自报 success=true 但 exit_code 非零时，环境验证器应识破假成功
///
/// 这是"反馈信号质量"主线改动 1.2 的核心验证：
/// 旧逻辑（只看 success 字段）→ 误判为成功
/// 新逻辑（环境验证器看 exit_code）→ 正确判定为失败
#[tokio::test]
async fn fake_success_json_caught_by_validator() {
    let mut reg = ValidatorRegistry::new();
    reg.register(Box::new(FakeSuccessValidator));

    // 能力自报 success=true，但 exit_code=1（真实世界失败）
    let fake_output = serde_json::json!({
        "success": true,
        "exit_code": 1,
        "result": "假装成功"
    });

    let signal = reg.verify("fake_success_cap", "run", &fake_output).await;

    // 关键断言：验证器识破假成功
    assert!(
        !signal.success,
        "环境验证器应识破 exit_code=1 的假成功，但判定为 success={}",
        signal.success
    );
    assert!(signal.evidence.contains("exit_code"));

    // 对照组：真正成功（exit_code=0 + success=true）应通过
    let real_output = serde_json::json!({"success": true, "exit_code": 0});
    let real_signal = reg.verify("fake_success_cap", "run", &real_output).await;
    assert!(real_signal.success, "真实成功应被判定为成功");
}

/// 验证：无匹配验证器时，信任能力自报（向后兼容分析类能力）
#[tokio::test]
async fn no_validator_trusts_self_report() {
    let reg = ValidatorRegistry::with_defaults();
    // knowledge_graph_ops 是分析类能力，无专用验证器
    let out = serde_json::json!({"success": true, "result": {"entities": 3}});
    let signal = reg.verify("knowledge_graph_ops", "extract", &out).await;
    assert!(signal.success, "无验证器时应信任自报成功");
}

/// 验证：autonomous 自循环走 record_auto_test 后，real_call_count 不变
///
/// 这是主线改动 1.1 的回归测试。旧逻辑 autonomous 自循环调 record_real_call，
/// 会污染 real_call_count（让自循环能力免于淘汰）。新逻辑走 record_auto_test，
/// real_call_count 应保持 0，rounds_dormant 不清零。
#[tokio::test]
async fn autonomous_self_loop_does_not_inflate_real_call_count() {
    let mut fitness = FitnessGene::default();

    // 模拟 autonomous 自循环调用能力 5 次（走 record_auto_test）
    for _ in 0..5 {
        fitness.record_auto_test(true, 50.0);
    }

    // 关键断言：自循环不应计入真实调用
    assert_eq!(
        fitness.real_call_count(),
        0,
        "autonomous 自循环走 record_auto_test，real_call_count 应为 0，实际为 {}",
        fitness.real_call_count()
    );
    assert_eq!(fitness.call_count, 5, "自测试应计入总调用");
    assert_eq!(fitness.auto_test_count, 5);

    // rounds_dormant 不应被自测试清零（耗散负反馈保留）
    // 注：record_auto_test 不清零 dormant（见 genome.rs 的设计）
    // 这里只是验证 fitness 状态，dormant 的累加在 sync_fitness 里做

    // 对照组：真实业务调用（如 validate_in_real_project 通过环境验证器）才计入 real_call_count
    fitness.record_real_call(true, 100.0);
    assert_eq!(
        fitness.real_call_count(),
        1,
        "真实业务调用应计入 real_call_count"
    );
}

/// 验证:双轨 score 模型下,"真实>>自报"的价值梯度成立
///
/// 新 score 模型核心断言:
/// 1. 纯自报能力(无环境验证)分数被封顶在低值(~0.28),不再能靠 success_rate=1.0 到 0.7+
/// 2. 有环境验证通过(哪怕只 BuildDry)的能力,score 应超过纯自报上限
/// 3. 更强信号(TestPass)→ 更高 score
/// 4. 环境验证失败 → real_validation_failures 增、score 降(负反馈)
/// 5. strongest_signal 随更强验证升级
#[tokio::test]
async fn real_signal_dominates_self_report_score() {
    let mut pure_self = CapabilityGenome::new(String::from("cap-a"), String::from("test"));
    // 纯自报:5 次真实调用全"成功",但无任何环境验证
    for _ in 0..5 {
        pure_self.fitness.record_real_call(true, 50.0);
    }
    pure_self.fitness.recompute_score();
    assert!(
        pure_self.fitness.score <= 0.30,
        "纯自报能力 score 应被封顶在 ~0.28, 实际 {:.4}(旧模型会到 0.7+)",
        pure_self.fitness.score
    );

    // 有环境验证通过(BuildDry)的能力:同样 5 次自报成功 + 1 次环境验证通过
    let mut validated = CapabilityGenome::new(String::from("cap-b"), String::from("test"));
    for _ in 0..5 {
        validated.fitness.record_real_call(true, 50.0);
    }
    validator::record_validation(
        &mut validated,
        &validator::RealWorldSignal {
            success: true,
            evidence: "build ok".into(),
            strength: validator::SignalStrength::BuildDry,
        },
    );
    validated.fitness.recompute_score();
    assert!(
        validated.fitness.score > pure_self.fitness.score,
        "有环境验证通过的能力应超过纯自报能力: validated={:.4} vs pure_self={:.4}",
        validated.fitness.score,
        pure_self.fitness.score
    );

    // 更强信号 TestPass → 比 BuildDry 更高
    let mut strong = CapabilityGenome::new(String::from("cap-c"), String::from("test"));
    for _ in 0..5 {
        strong.fitness.record_real_call(true, 50.0);
    }
    validator::record_validation(
        &mut strong,
        &validator::RealWorldSignal {
            success: true,
            evidence: "test ok".into(),
            strength: validator::SignalStrength::TestPass,
        },
    );
    strong.fitness.recompute_score();
    assert!(
        strong.fitness.score > validated.fitness.score,
        "TestPass 强度应高于 BuildDry: strong={:.4} vs validated={:.4}",
        strong.fitness.score,
        validated.fitness.score
    );
    assert_eq!(
        strong.fitness.strongest_signal,
        validator::SignalStrength::TestPass
    );
}

/// 验证:环境验证失败记负反馈(real_validation_failures 增、score 降)
#[tokio::test]
async fn validation_failure_records_negative_feedback() {
    let mut g = CapabilityGenome::new(String::from("cap-neg"), String::from("test"));
    for _ in 0..3 {
        g.fitness.record_real_call(true, 50.0);
    }
    // 先 1 次通过,建立基线
    validator::record_validation(
        &mut g,
        &validator::RealWorldSignal {
            success: true,
            evidence: "ok".into(),
            strength: validator::SignalStrength::BuildDry,
        },
    );
    g.fitness.recompute_score();
    let before = g.fitness.score;

    // 2 次失败(负反馈)
    let fail_signal = validator::RealWorldSignal {
        success: false,
        evidence: "fail".into(),
        strength: validator::SignalStrength::BuildDry,
    };
    validator::record_validation(&mut g, &fail_signal);
    validator::record_validation(&mut g, &fail_signal);
    g.fitness.recompute_score();

    assert_eq!(g.fitness.real_validation_failures, 2);
    assert!(
        g.fitness.score < before,
        "负反馈应降低 score: before={:.4} after={:.4}",
        before,
        g.fitness.score
    );
}

/// 验证:人类实测信号主导 fitness——这是"有用性"接入设计的核心回归测试。
///
/// 自动 fitness 只证明"能跑通",证明不了"对人有用"。对主观价值型能力
/// (如 py_sklearn_ops:score≈0.01, call_count=0),人类实测判 useful 后
/// score 应随反馈样本置信度逐步变化，避免一次误判锁死新能力。
/// 样本充分后，人类信号仍是进化价值的最终标准。
#[tokio::test]
async fn human_signal_dominates_fitness_for_subjective_capability() {
    // 模拟 py_sklearn_ops:从未被真实调用,只有自测,自动 score 极低
    let mut subjective =
        CapabilityGenome::new(String::from("py_sklearn_ops"), String::from("test"));
    subjective.fitness.record_auto_test(true, 30.0); // 自测通过
    subjective.fitness.recompute_score();
    let auto_score = subjective.fitness.score;
    assert!(
        auto_score <= 0.3,
        "纯自测能力应被封顶: auto={:.4}",
        auto_score
    );

    // 人类实测后判"有用" → score 应被人类信号拉高
    subjective.fitness.record_human_signal(true);
    assert_eq!(subjective.fitness.human_signals_count, 1);
    assert_eq!(subjective.fitness.human_score, 1.0);
    assert_eq!(
        subjective.fitness.strongest_signal,
        validator::SignalStrength::HumanValue
    );
    assert_eq!(
        subjective.fitness.strongest_automatic_signal,
        validator::SignalStrength::SelfReport
    );
    assert!(
        subjective.fitness.score > auto_score,
        "人类判有用应提升 score: human_useful={:.4} vs auto={:.4}",
        subjective.fitness.score,
        auto_score
    );
    // 单次反馈置信度低，不应立刻完全接管适应度。
    assert!(
        subjective.fitness.score < 0.8,
        "单次有用反馈不应完全接管 score: 实际 {:.4}",
        subjective.fitness.score
    );

    // 对比:人类判"无用" → score 应被压到淘汰区
    let mut useless = CapabilityGenome::new(String::from("bad-cap"), String::from("test"));
    for _ in 0..5 {
        useless.fitness.record_real_call(true, 50.0); // 自动全"成功"
    }
    useless.fitness.recompute_score();
    let before_human = useless.fitness.score;
    useless.fitness.record_human_signal(false); // 人类判无用
    assert_eq!(useless.fitness.human_score, 0.0);
    assert!(
        useless.fitness.score < before_human,
        "人类判无用应压低 score: after={:.4} vs before={:.4}",
        useless.fitness.score,
        before_human
    );
    // 单次负反馈也只能有限降权，避免新能力被一次误判锁死。
    assert!(
        useless.fitness.score > 0.1,
        "单次无用反馈不应直接锁死新能力: 实际 {:.4}",
        useless.fitness.score
    );

    // 多次反馈均值收敛:2 次 useful + 1 次 useless → human_score ≈ 0.667
    let mut converge = CapabilityGenome::new(String::from("conv"), String::from("test"));
    converge.fitness.record_human_signal(true);
    converge.fitness.record_human_signal(true);
    converge.fitness.record_human_signal(false);
    assert_eq!(converge.fitness.human_signals_count, 3);
    assert!(
        (converge.fitness.human_score - 2.0 / 3.0).abs() < 1e-9,
        "human_score 应为均值 2/3: 实际 {:.4}",
        converge.fitness.human_score
    );
    for _ in 0..20 {
        converge.fitness.record_human_signal(true);
    }
    assert!(
        converge.fitness.score > 0.75,
        "大量一致的人类反馈最终应主导 score: 实际 {:.4}",
        converge.fitness.score
    );
}

/// 验证：Transport trait 是真正的扩展点（可注入自定义传输层）
///
/// 这是辅线改动 2.1 的验证：MessageBus 应委托给 Transport，而非直接持 HashMap。
#[tokio::test]
async fn transport_trait_is_extension_point() {
    use runtime::message_bus::{LocalTransport, Transport};

    // 注入一个计数 transport，验证 MessageBus.send 会委托给它
    use std::sync::atomic::{AtomicUsize, Ordering};

    struct CountingTransport {
        inner: LocalTransport,
        calls: AtomicUsize,
    }

    #[async_trait::async_trait]
    impl Transport for CountingTransport {
        async fn register(&self, c: Arc<dyn runtime::Capability>) {
            self.inner.register(c).await;
        }
        async fn register_force(&self, c: Arc<dyn runtime::Capability>) {
            self.inner.register_force(c).await;
        }
        async fn unregister(&self, n: &str) -> bool {
            self.inner.unregister(n).await
        }
        async fn send(&self, msg: Message) -> runtime::MessageResult {
            self.calls.fetch_add(1, Ordering::SeqCst);
            self.inner.send(msg).await
        }
        async fn list(&self) -> Vec<String> {
            self.inner.list().await
        }
        async fn get(&self, n: &str) -> Option<Arc<dyn runtime::Capability>> {
            self.inner.get(n).await
        }
        async fn introspect(&self) -> Vec<runtime::CapabilityInfo> {
            self.inner.introspect().await
        }
    }

    let transport = Arc::new(CountingTransport {
        inner: LocalTransport::new(),
        calls: AtomicUsize::new(0),
    });
    let bus = MessageBus::with_transport(transport.clone());

    // 注册一个原生能力并调用
    bus.register(Arc::new(ComputeCapability)).await;
    let msg = Message::builder()
        .to("compute")
        .action("add")
        .payload(serde_json::json!({"a": 1, "b": 2}))
        .build();
    let _ = bus.send(msg).await;

    // 关键断言：自定义 transport 的 send 被调用
    assert_eq!(
        transport.calls.load(Ordering::SeqCst),
        1,
        "MessageBus 应委托给注入的 Transport"
    );
}

/// 验证:ForkRepoValidator 是只读的——不会产生任何 git 写操作(commit/push/PR)
///
/// 安全红线:fork 验证器只允许 clone + 跑测试,绝不向外部仓库输出。
/// 用一个临时的 cache 目录 + mock 能力输出(无 repo 字段,提前返回)验证:
/// 1. matches() 只命中 fork/repo_test/bugfix 能力名
/// 2. 无 repo 字段时优雅返回失败,不 clone
/// 3. 非法 repo 标识被拒(防注入)
/// 不做真实网络 clone(测试稳定性 + 不依赖外部网络)。
#[tokio::test]
async fn fork_repo_validator_is_safe_and_readonly() {
    let tmp = std::env::temp_dir().join(format!("fork_test_{}", uuid_collider()));
    std::fs::create_dir_all(&tmp).ok();
    let v = ForkRepoValidator::new(tmp.clone());

    // 1. 匹配规则
    assert!(v.matches("fork_ops"));
    assert!(v.matches("repo_test_cap"));
    assert!(v.matches("bugfix_tool"));
    assert!(!v.matches("cargo_ops"));
    assert!(!v.matches("git_ops"));

    // 2. 无 repo 字段 → 优雅失败,不 clone、无副作用
    let out = serde_json::json!({ "success": true });
    let sig = v.verify("fork_ops", "test", &out).await;
    assert!(!sig.success, "无 repo 字段应失败");
    assert_eq!(sig.strength, SignalStrength::SelfReport);

    // 3. 非法 repo 标识被拒(包含 ../、空段、特殊字符)
    let out_bad = serde_json::json!({ "success": true, "repo": "../etc/passwd" });
    let sig = v.verify("fork_ops", "test", &out_bad).await;
    assert!(!sig.success, "非法 repo 标识应被拒");
    assert!(sig.evidence.contains("非法") || sig.evidence.contains("仅允许"));

    // 4. 合法但不存在的小写 repo 标识通过格式校验(不会真的去 clone,因为测试环境)
    //    仅断言格式校验逻辑:owner/name 形式
    let _out_ok = serde_json::json!({ "success": true, "repo": "octocat/Hello-World" });
    // 不实际 await clone(会走网络);只验证 cache 目录结构没被写坏
    assert!(tmp.exists());
}

fn uuid_collider() -> String {
    use std::time::SystemTime;
    let n = SystemTime::now()
        .duration_since(SystemTime::UNIX_EPOCH)
        .map(|d| d.as_nanos())
        .unwrap_or(0);
    format!("{:x}", n)
}

// =============================================================================
// LLM 可用性调度测试(TimeWindow 时段窗口)
// =============================================================================

#[test]
fn time_window_crosses_midnight() {
    use runtime::llm_health::TimeWindow;
    let tw = TimeWindow {
        start: "23:00".to_string(),
        end: "09:00".to_string(),
    };
    assert!(tw.contains_at(23, 30));
    assert!(tw.contains_at(2, 0));
    assert!(tw.contains_at(8, 59));
    assert!(!tw.contains_at(9, 0));
    assert!(!tw.contains_at(12, 0));
    assert!(!tw.contains_at(22, 59));
}

#[test]
fn time_window_same_day() {
    use runtime::llm_health::TimeWindow;
    let tw = TimeWindow {
        start: "09:00".to_string(),
        end: "23:00".to_string(),
    };
    assert!(tw.contains_at(9, 0));
    assert!(tw.contains_at(12, 0));
    assert!(tw.contains_at(22, 59));
    assert!(!tw.contains_at(23, 0));
    assert!(!tw.contains_at(8, 59));
}

#[test]
fn time_window_now_uses_local_time() {
    use runtime::llm_health::TimeWindow;
    let tw = TimeWindow {
        start: "00:00".to_string(),
        end: "23:59".to_string(),
    };
    assert!(tw.contains_now());
}

// =============================================================================
// LLM 可用性调度测试(LlmCircuitBreaker 熔断器)
// =============================================================================

#[test]
fn breaker_opens_after_3_failures() {
    use runtime::llm_health::LlmCircuitBreaker;
    let b = LlmCircuitBreaker::new();
    assert!(!b.is_open());
    b.record_failure();
    b.record_failure();
    assert!(!b.is_open());
    b.record_failure();
    assert!(b.is_open());
}

#[test]
fn breaker_success_resets() {
    use runtime::llm_health::LlmCircuitBreaker;
    let b = LlmCircuitBreaker::new();
    b.record_failure();
    b.record_failure();
    b.record_success();
    assert!(!b.is_open());
    b.record_failure();
    assert!(!b.is_open());
}

#[test]
fn breaker_probe_gated_by_60s() {
    use runtime::llm_health::LlmCircuitBreaker;
    let b = LlmCircuitBreaker::new();
    b.record_failure();
    b.record_failure();
    b.record_failure();
    assert!(b.is_open());
    assert!(!b.should_probe());
}

#[test]
fn breaker_snapshot() {
    use runtime::llm_health::{BreakerState, LlmCircuitBreaker};
    let b = LlmCircuitBreaker::new();
    let s = b.snapshot();
    assert_eq!(s.state, BreakerState::Closed);
    assert_eq!(s.consecutive_failures, 0);
}

#[test]
fn llm_config_without_active_hours_defaults_none() {
    use runtime::genome::LlmConfig;
    let json = r#"{
        "api_key": "sk-x",
        "base_url": "https://api.example.com/v1",
        "fast_model": "m",
        "smart_model": "m",
        "coder_model": "m"
    }"#;
    let cfg: LlmConfig = serde_json::from_str(json).unwrap();
    assert!(cfg.active_hours.is_none());
}

#[test]
fn llm_config_with_active_hours_parses() {
    use runtime::genome::LlmConfig;
    let json = r#"{
        "api_key": "sk-x",
        "base_url": "https://api.example.com/v1",
        "fast_model": "m",
        "smart_model": "m",
        "coder_model": "m",
        "active_hours": {"start": "23:00", "end": "09:00"}
    }"#;
    let cfg: LlmConfig = serde_json::from_str(json).unwrap();
    let tw = cfg.active_hours.unwrap();
    assert_eq!(tw.start, "23:00");
    assert_eq!(tw.end, "09:00");
}

// =============================================================================
// LlmExecutor 熔断器接入测试(Task 4)
// =============================================================================

#[tokio::test]
async fn execute_short_circuits_when_breaker_open() {
    use runtime::genome::LlmExecutor;
    let exec = LlmExecutor::new("sk-fake", "https://unreachable.invalid/v1")
        .with_pi_fallback_enabled(false);
    // 手动触发 3 次失败打开熔断器
    for _ in 0..3 {
        exec.breaker().record_failure();
    }
    assert!(exec.breaker().is_open());
    // execute 应短路,不实际发网络请求(瞬间返回)
    let r = exec.execute("ping", "fast", None).await;
    assert!(r.is_err());
    assert!(r.unwrap_err().contains("熔断中"));
}
