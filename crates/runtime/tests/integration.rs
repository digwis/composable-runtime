use capabilities::{ComputeCapability, GreetCapability, StoreCapability};
use runtime::{Message, MessageBus, OrchestratorBuilder, RegistryBuilder, Workflow};

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
