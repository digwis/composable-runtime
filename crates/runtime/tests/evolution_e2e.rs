//! 进化系统端到端测试
//!
//! 验证每个组件是否真正工作：
//! 1. 平台检测（wasmtime / wasm32-wasi / rustc）
//! 2. ExecutorRegistry — 注册、执行（Python / Rust-WASM / Rust-Native）
//! 3. RuntimeSpec — 序列化、持久化
//! 4. ScriptedCapability — Custom 执行器分发
//! 5. MetaEvolver — 元进化报告（不需要 LLM）
//! 6. 代码渲染 — Python / WASM / Native 模板

use runtime::meta_evolve::{ExecutorRegistry, CustomExecutorSpec, ExecutorLineage, ExecutorContext};
use runtime::platform::Platform;
use runtime::genome::{CapabilityGenome, ActionGene, ActionImpl, ScriptedCapability};
use runtime::message_bus::MessageBus;
use std::sync::Arc;

fn make_executor(name: &str, language: &str, code: &str) -> CustomExecutorSpec {
    CustomExecutorSpec {
        type_name: name.into(),
        description: format!("测试执行器 ({})", language),
        params_schema: serde_json::json!({}),
        executor_code: code.into(),
        language: language.into(),
        timeout_secs: 30,
        created_at: "0".into(),
        lineage: ExecutorLineage::default(),
    }
}

#[tokio::test]
async fn test_01_platform_detection() {
    let platform = Platform::detect();
    println!("\n=== 1. 平台检测 ===");
    println!("  OS: {} ({})", platform.os, platform.arch);
    println!("  has_rustc: {}", platform.env.get("has_rustc").unwrap_or(&"false".into()));
    println!("  has_wasmtime: {}", platform.env.get("has_wasmtime").unwrap_or(&"false".into()));
    println!("  has_wasm32_wasi: {}", platform.env.get("has_wasm32_wasi").unwrap_or(&"false".into()));
    println!("  has_python3: {}", platform.env.get("has_python3").unwrap_or(&"false".into()));

    assert!(platform.supports_process, "需要进程支持");
    assert!(
        platform.env.get("has_python3").map(|v| v == "true").unwrap_or(false),
        "需要 python3"
    );
    println!("  ✅ 平台检测通过");
}

#[tokio::test]
async fn test_02_executor_registry_register() {
    println!("\n=== 2. ExecutorRegistry 注册 ===");
    let tmp_dir = std::env::temp_dir().join(format!("evo_test_{}", uuid::Uuid::new_v4()));
    let registry = ExecutorRegistry::new(&tmp_dir);

    let exec = make_executor(
        "echo_py",
        "python",
        "import json; print(json.dumps({'success': True, 'echo': __EXECUTOR_INPUT__}))",
    );
    registry.register(exec).await;

    let types = registry.all_executor_types().await;
    println!("  执行器类型: {:?}", types);
    assert!(types.contains(&"echo_py".to_string()), "应包含 echo_py");
    println!("  ✅ 注册通过");
}

#[tokio::test]
async fn test_03_python_executor() {
    println!("\n=== 3. Python 执行器 ===");
    let tmp_dir = std::env::temp_dir().join(format!("evo_test_{}", uuid::Uuid::new_v4()));
    let registry = ExecutorRegistry::new(&tmp_dir);

    registry.register(make_executor(
        "echo_py",
        "python",
        "import json; print(json.dumps({'success': True, 'lang': 'python', 'input_received': True, 'echo': __EXECUTOR_INPUT__.get('input', {}).get('task', 'none')}))",
    )).await;

    let ctx = ExecutorContext {
        capability_name: "test_cap".into(),
        action_name: "test_action".into(),
    };
    let result = registry.execute(
        "echo_py",
        &serde_json::json!({"key": "value"}),
        &serde_json::json!({"task": "hello"}),
        &ctx,
    ).await;

    println!("  结果: {:?}", result);
    assert!(result.is_ok(), "Python 执行器应成功");
    let v = result.unwrap();
    assert_eq!(v["success"], true);
    assert_eq!(v["lang"], "python");
    println!("  ✅ Python 执行器通过");
}

#[tokio::test]
async fn test_04_rust_wasm_executor() {
    println!("\n=== 4. Rust/WASM 执行器 ===");
    let platform = Platform::detect();
    let has_wasmtime = platform.env.get("has_wasmtime").map(|v| v == "true").unwrap_or(false);
    let has_wasi = platform.env.get("has_wasm32_wasi").map(|v| v == "true").unwrap_or(false);

    if !has_wasmtime || !has_wasi {
        println!("  ⏭️  跳过: 需要 wasmtime + wasm32-wasi target");
        println!("    has_wasmtime={}, has_wasm32_wasi={}", has_wasmtime, has_wasi);
        return;
    }

    let tmp_dir = std::env::temp_dir().join(format!("evo_test_{}", uuid::Uuid::new_v4()));
    let registry = ExecutorRegistry::new(&tmp_dir);

    let code = r#"
    // 简单回显：解析输入并返回
    let parsed: serde_json::Value = serde_json::from_str(__input).unwrap_or(serde_json::json!({}));
    println!("{}", serde_json::json!({
        "success": true,
        "lang": "wasm",
        "input_keys": parsed.get("input").map(|v| v.to_string()).unwrap_or_default()
    }).to_string());
"#;

    // 注意：WASM 只能用标准库，上面的 serde_json 不可用
    // 用纯标准库版本：
    let code = r#"
    // 纯标准库：直接回显输入
    println!("{{\"success\": true, \"lang\": \"wasm\", \"input_len\": {}}}", __input.len());
"#;

    registry.register(make_executor("echo_wasm", "rust", code)).await;

    let ctx = ExecutorContext {
        capability_name: "test_cap".into(),
        action_name: "test_action".into(),
    };
    let result = registry.execute(
        "echo_wasm",
        &serde_json::json!({}),
        &serde_json::json!({"task": "compute"}),
        &ctx,
    ).await;

    println!("  结果: {:?}", result);
    assert!(result.is_ok(), "WASM 执行器应成功");
    let v = result.unwrap();
    assert_eq!(v["success"], true);
    assert_eq!(v["lang"], "wasm");
    println!("  ✅ Rust/WASM 执行器通过");
}

#[tokio::test]
async fn test_05_rust_native_executor() {
    println!("\n=== 5. Rust/Native 执行器（热加载） ===");
    let platform = Platform::detect();
    let has_rustc = platform.env.get("has_rustc").map(|v| v == "true").unwrap_or(false);
    if !has_rustc {
        println!("  ⏭️  跳过: 需要 rustc");
        return;
    }

    let tmp_dir = std::env::temp_dir().join(format!("evo_test_{}", uuid::Uuid::new_v4()));
    let registry = ExecutorRegistry::new(&tmp_dir);

    let code = r#"
    // 原生执行器：直接操作字符串
    __output = format!("{{\"success\": true, \"lang\": \"native\", \"input_len\": {}}}", __input.len());
"#;

    registry.register(make_executor("echo_native", "rust_native", code)).await;

    let ctx = ExecutorContext {
        capability_name: "test_cap".into(),
        action_name: "test_action".into(),
    };
    let result = registry.execute(
        "echo_native",
        &serde_json::json!({}),
        &serde_json::json!({"task": "native_test"}),
        &ctx,
    ).await;

    println!("  结果: {:?}", result);
    assert!(result.is_ok(), "Native 执行器应成功: {:?}", result);
    let v = result.unwrap();
    assert_eq!(v["success"], true);
    assert_eq!(v["lang"], "native");
    println!("  ✅ Rust/Native 执行器通过");

    // 测试热替换：修改代码后再次执行
    println!("  --- 测试热替换 ---");
    let code2 = r#"
    __output = format!("{{\"success\": true, \"lang\": \"native_v2\", \"input_len\": {}}}", __input.len());
"#;
    registry.mutate_executor("echo_native", code2.to_string(), None).await;

    let result2 = registry.execute(
        "echo_native",
        &serde_json::json!({}),
        &serde_json::json!({"task": "hot_swap"}),
        &ctx,
    ).await;

    println!("  热替换结果: {:?}", result2);
    assert!(result2.is_ok());
    let v2 = result2.unwrap();
    assert_eq!(v2["lang"], "native_v2");
    println!("  ✅ 热替换通过");
}

#[test]
fn test_06_runtime_spec_persistence() {
    println!("\n=== 6. RuntimeSpec 持久化 ===");
    let tmp_dir = std::env::temp_dir().join(format!("evo_test_{}", uuid::Uuid::new_v4()));
    std::fs::create_dir_all(&tmp_dir).unwrap();

    {
        let registry = ExecutorRegistry::new(&tmp_dir);
        // 同步注册（block_on 在非 async 上下文中可以工作）
        let rt = tokio::runtime::Runtime::new().unwrap();
        rt.block_on(registry.register(make_executor(
            "persist_test",
            "python",
            "print(json.dumps({'ok': True}))",
        )));
    }

    // 重新加载
    let registry2 = ExecutorRegistry::new(&tmp_dir);
    let types = rt_block_on(registry2.all_executor_types());
    println!("  重新加载后执行器: {:?}", types);
    assert!(types.contains(&"persist_test".to_string()));
    println!("  ✅ 持久化通过");
}

fn rt_block_on<F: std::future::Future>(f: F) -> F::Output {
    tokio::runtime::Runtime::new().unwrap().block_on(f)
}

#[tokio::test]
async fn test_07_scripted_capability_custom_executor() {
    println!("\n=== 7. ScriptedCapability + Custom 执行器 ===");
    let tmp_dir = std::env::temp_dir().join(format!("evo_test_{}", uuid::Uuid::new_v4()));
    let registry = Arc::new(ExecutorRegistry::new(&tmp_dir));

    registry.register(make_executor(
        "simple_echo",
        "python",
        "import json; print(json.dumps({'result': __EXECUTOR_INPUT__.get('input', {}).get('task', 'unknown')}))",
    )).await;

    // 创建使用 Custom 执行器的基因组
    let genome = CapabilityGenome {
        name: "echo_capability".into(),
        version: "1.0.0".into(),
        description: "回显能力".into(),
        actions: vec![ActionGene {
            name: "echo".into(),
            description: "回显输入".into(),
            input_schema: serde_json::json!({"type": "object"}),
            implementation: ActionImpl::Custom {
                executor_type: "simple_echo".into(),
                params: serde_json::json!({}),
            },
        }],
        fitness: Default::default(),
        lineage: Default::default(),
    };

    let bus = Arc::new(MessageBus::new());
    let cap = ScriptedCapability::from_genome(genome)
        .with_executor_registry(registry)
        .with_bus(bus.clone());

    // 注册并执行
    bus.register(Arc::new(cap)).await;

    let msg = runtime::Message::builder()
        .from("test")
        .to("echo_capability")
        .action("echo")
        .payload(serde_json::json!({"task": "hello_evolution"}))
        .build();

    let response = bus.send(msg).await;
    println!("  响应: {:?}", response);
    assert!(response.is_ok(), "Custom 执行器能力应成功");
    let resp = response.unwrap();
    println!("  payload: {}", serde_json::to_string_pretty(&resp.payload).unwrap());
    println!("  ✅ ScriptedCapability + Custom 通过");
}

#[tokio::test]
async fn test_08_executor_registry_report() {
    println!("\n=== 8. ExecutorRegistry 报告 ===");
    let tmp_dir = std::env::temp_dir().join(format!("evo_test_{}", uuid::Uuid::new_v4()));
    let registry = Arc::new(ExecutorRegistry::new(&tmp_dir));

    registry.register(make_executor(
        "report_test",
        "python",
        "print(json.dumps({'ok': True}))",
    )).await;

    let spec = registry.spec().await;
    println!("  自定义执行器数: {}", spec.custom_executors.len());
    println!("  元进化事件数: {}", spec.meta_history.len());
    println!("  执行器列表:");
    for e in &spec.custom_executors {
        println!("    • {} [{}] gen={} — {}", e.type_name, e.language, e.lineage.generation, e.description);
    }
    assert!(spec.custom_executors.iter().any(|e| e.type_name == "report_test"));
    println!("  ✅ ExecutorRegistry 报告通过");
}

#[tokio::test]
async fn test_09_code_renderers() {
    println!("\n=== 9. 代码渲染器 ===");

    // Python
    let py = runtime::meta_evolve::render_python_code(
        "print(json.dumps(__EXECUTOR_INPUT__))",
        r#"{"test": true}"#,
    );
    assert!(py.contains("__EXECUTOR_INPUT__"));
    assert!(py.contains("json.loads"));
    println!("  ✅ Python 渲染通过");

    // WASM
    let wasm = runtime::meta_evolve::render_rust_wasm_code(
        r#"println!("hello");"#,
    );
    assert!(wasm.contains("fn main()"));
    assert!(wasm.contains("__input"));
    println!("  ✅ WASM 渲染通过");

    // Native
    let native = runtime::meta_evolve::render_rust_native_code(
        r#"__output = "ok".to_string();"#,
    );
    assert!(native.contains("#[no_mangle]"));
    assert!(native.contains("extern \"C\""));
    assert!(native.contains("execute"));
    assert!(native.contains("free_string"));
    assert!(native.contains("__output"));
    println!("  ✅ Native 渲染通过");
}

#[tokio::test]
async fn test_10_executor_mutation_and_elimination() {
    println!("\n=== 10. 执行器变异与淘汰 ===");
    let tmp_dir = std::env::temp_dir().join(format!("evo_test_{}", uuid::Uuid::new_v4()));
    let registry = ExecutorRegistry::new(&tmp_dir);

    // 注册
    registry.register(make_executor(
        "mutate_me",
        "python",
        "print(json.dumps({'version': 1}))",
    )).await;

    // 变异
    registry.mutate_executor(
        "mutate_me",
        "print(json.dumps({'version': 2}))".into(),
        Some("v2".into()),
    ).await.unwrap();

    let spec = registry.spec().await;
    let exec = spec.get_executor("mutate_me").unwrap();
    assert_eq!(exec.executor_code, "print(json.dumps({'version': 2}))");
    assert_eq!(exec.lineage.generation, 1);
    drop(spec);
    println!("  ✅ 变异通过 (generation=1)");

    // 淘汰
    registry.eliminate_executor("mutate_me").await.unwrap();
    let types = registry.all_executor_types().await;
    assert!(!types.contains(&"mutate_me".to_string()));
    println!("  ✅ 淘汰通过");
}
