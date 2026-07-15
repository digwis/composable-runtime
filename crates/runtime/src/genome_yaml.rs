/// 基因组 YAML 多文件存储 — 从 .evolution/<capability>/ 目录加载/保存基因组
///
/// 这是进化引擎的统一存储格式。每个能力一个子目录，内部包含：
///   - genome.yaml  ：YAML 元数据（人类可读、LLM 友好、支持注释）
///   - actions/*.py ：Script 类型的独立 Python 代码
///   - actions/*.sh ：Shell 类型的独立 bash 命令
///   - actions/*.md ：LLm 类型的 prompt 模板
///   - versions/history.md ：每轮变异决策日志
/// 全局 manifest.yaml 和 shared/safety_rules.yaml 放在 .evolution/ 根目录。
use crate::genome::{
    ActionGene, ActionImpl, CapabilityGenome, CompositeStep, FitnessGene, LineageGene,
    MutationRecord, Origin, TestCase,
};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::Path;

/// 默认安全规则模板（内联，避免跨 crate include_str! 路径问题）
pub const DEFAULT_SAFETY_RULES: &str = r#"# 全局安全规则 — 适用于所有 Script/Shell 能力执行

rules:
  sandbox:
    allowed_paths:
      - "/tmp"
      - "{project_root}"
    blocked_paths:
      - "/etc/passwd"
      - "/etc/shadow"
      - "~/.ssh"
      - "~/.aws"
    blocked_network:
      - description: "禁止访问内网地址"
        pattern: "^(10\\.|172\\.(1[6-9]|2[0-9]|3[0-1])\\.|192\\.168\\.)"
      - description: "禁止访问 localhost 非标准端口"
        pattern: "^127\\.0\\.0\\.1:[0-9]+"

  execution:
    default_timeout: 30
    max_timeout: 3600
    blocked_commands:
      - "rm -rf /"
      - "mkfs"
      - "dd if="
      - "> /dev/sda"
      - "curl.*|.*sh"

  output:
    require_json_output: true
    max_output_size: 1048576

  prompt_injection:
    require_input_sanitize: true
    block_dynamic_eval: true
"#;

// ────────────────────────────────── YAML 格式定义 ──────────────────────────────────

/// manifest.yaml 中的能力条目
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ManifestEntry {
    pub name: String,
    pub version: String,
    pub description: String,
    pub score: f64,
    pub success_rate: f64,
    pub call_count: u32,
    pub rounds_dormant: u32,
    pub action_count: u32,
    pub action_names: Vec<String>,
    pub directory: String,
}

/// manifest.yaml 顶级结构
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Manifest {
    pub format_version: String,
    #[serde(default)]
    pub migrated_at: Option<String>,
    #[serde(default)]
    pub source: Option<String>,
    pub total_capabilities: u32,
    pub capabilities: Vec<ManifestEntry>,
}

/// genome.yaml 中的能力基因组
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GenomeYaml {
    pub name: String,
    pub version: String,
    pub description: String,
    pub actions: Vec<ActionYaml>,
    pub fitness: FitnessYaml,
    pub lineage: LineageYaml,
    #[serde(default)]
    pub test_suite: TestSuiteYaml,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ActionYaml {
    pub name: String,
    pub description: String,
    #[serde(default)]
    pub input_schema: serde_json::Value,
    pub implementation: RawImplRef,
}

/// 实现引用 — 统一反序列化为 RawImplRef，再通过 impl_type + code_file/code 等字段判定
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RawImplRef {
    #[serde(rename = "type")]
    pub impl_type: String,
    // Script
    #[serde(default)]
    pub code: Option<String>,
    #[serde(default)]
    pub code_file: Option<String>,
    #[serde(default)]
    pub language: Option<String>,
    #[serde(default)]
    pub timeout_secs: Option<u64>,
    // Shell
    #[serde(default)]
    pub command: Option<String>,
    #[serde(default)]
    pub command_file: Option<String>,
    // Llm
    #[serde(default)]
    pub prompt: Option<String>,
    #[serde(default)]
    pub prompt_file: Option<String>,
    #[serde(default)]
    pub model: Option<String>,
    #[serde(default)]
    pub system: Option<String>,
    // Composite
    #[serde(default)]
    pub steps: Option<Vec<serde_json::Value>>,
    #[serde(default)]
    pub steps_file: Option<String>,
    // Rule
    #[serde(default)]
    pub template: Option<serde_json::Value>,
    #[serde(default)]
    pub template_file: Option<String>,
    // Native
    #[serde(default)]
    pub capability: Option<String>,
    #[serde(default)]
    pub action: Option<String>,
    // Custom
    #[serde(default)]
    pub executor_type: Option<String>,
    #[serde(default)]
    pub params: Option<serde_json::Value>,
}

impl RawImplRef {
    /// 判定这是文件引用还是内联：如果有任何 _file 字段非空，就是文件引用
    pub fn is_file_ref(&self) -> bool {
        self.code_file.is_some()
            || self.command_file.is_some()
            || self.prompt_file.is_some()
            || self.steps_file.is_some()
            || self.template_file.is_some()
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct FitnessYaml {
    #[serde(default)]
    pub score: f64,
    #[serde(default)]
    pub success_rate: f64,
    #[serde(default)]
    pub call_count: u32,
    #[serde(default)]
    pub success_count: u32,
    #[serde(default)]
    pub failure_count: u32,
    #[serde(default)]
    pub auto_test_count: u32,
    #[serde(default)]
    pub real_call_count: u32,
    #[serde(default)]
    pub avg_latency_ms: f64,
    #[serde(default)]
    pub rounds_dormant: u32,
    #[serde(default)]
    pub output_quality: f64,
    #[serde(default)]
    pub coverage_score: f64,
    #[serde(default)]
    pub dependency_complexity: f64,
    #[serde(default)]
    pub non_empty_output_count: u32,
    #[serde(default)]
    pub innovation_score: f64,
    #[serde(default)]
    pub utility_score: f64,
    #[serde(default)]
    pub last_evaluated: Option<String>,
    #[serde(default)]
    pub llm_evaluated_at: u64,
    #[serde(default)]
    pub real_validation_passes: u32,
    #[serde(default)]
    pub real_validation_failures: u32,
    #[serde(default)]
    pub strongest_signal: crate::validator::SignalStrength,
    #[serde(default)]
    pub strongest_automatic_signal: crate::validator::SignalStrength,
    #[serde(default)]
    pub human_signals_count: u32,
    #[serde(default)]
    pub human_score: f64,
    #[serde(default)]
    pub total_token_cost: u64,
    #[serde(default)]
    pub last_token_cost: u64,
    #[serde(default)]
    pub profit_ratio: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct LineageYaml {
    pub origin: String,
    #[serde(default)]
    pub generation: u32,
    #[serde(default)]
    pub parent: Option<String>,
    #[serde(default)]
    pub mutations: Vec<MutationYaml>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MutationYaml {
    #[serde(rename = "type")]
    pub mutation_type: String,
    pub description: String,
    #[serde(default)]
    pub timestamp: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct TestSuiteYaml {
    #[serde(default)]
    pub cases: Vec<serde_json::Value>,
    #[serde(default)]
    pub count: u32,
}

// ────────────────────────────────── 加载 ──────────────────────────────────

/// 从 .evolution/ 多文件目录加载所有能力基因组
///
/// 若 manifest.yaml 存在则按索引加载，否则扫描子目录。
pub fn load_genomes_from_yaml_dir(dir: &Path) -> Result<Vec<CapabilityGenome>, String> {
    let manifest_path = dir.join("manifest.yaml");

    if manifest_path.exists() {
        let manifest_yaml = std::fs::read_to_string(&manifest_path)
            .map_err(|e| format!("读取 manifest.yaml 失败: {}", e))?;
        let manifest: Manifest = serde_yaml::from_str(&manifest_yaml)
            .map_err(|e| format!("解析 manifest.yaml 失败: {}", e))?;
        tracing::info!(
            "从 YAML manifest 加载 {} 个能力",
            manifest.total_capabilities
        );

        let mut genomes = Vec::new();
        for entry in &manifest.capabilities {
            let cap_dir = dir.join(&entry.directory);
            match load_genome_from_dir(&cap_dir) {
                Ok(g) => genomes.push(g),
                Err(e) => tracing::warn!("跳过 {}: {}", entry.name, e),
            }
        }
        Ok(genomes)
    } else {
        // 无 manifest，扫描目录
        tracing::warn!("无 manifest.yaml，扫描目录加载");
        let mut genomes = Vec::new();
        if let Ok(entries) = std::fs::read_dir(dir) {
            for entry in entries.flatten() {
                let path = entry.path();
                if path.is_dir() && !is_special_dir(&path) {
                    match load_genome_from_dir(&path) {
                        Ok(g) => genomes.push(g),
                        Err(e) => tracing::warn!("跳过 {:?}: {}", path, e),
                    }
                }
            }
        }
        Ok(genomes)
    }
}

fn is_special_dir(path: &Path) -> bool {
    let name = path.file_name().unwrap_or_default().to_string_lossy();
    name == "shared" || name == "versions" || name.starts_with('.')
}

/// 从单个能力目录加载 genome
fn load_genome_from_dir(dir: &Path) -> Result<CapabilityGenome, String> {
    let genome_yaml_path = dir.join("genome.yaml");
    if !genome_yaml_path.exists() {
        return Err(format!("{:?} 中缺少 genome.yaml", dir));
    }

    let yaml_str = std::fs::read_to_string(&genome_yaml_path)
        .map_err(|e| format!("读取 {:?} 失败: {}", genome_yaml_path, e))?;
    let gy: GenomeYaml = serde_yaml::from_str(&yaml_str)
        .map_err(|e| format!("解析 {:?} 失败: {}", genome_yaml_path, e))?;

    // 转换 ActionYaml → ActionGene
    let mut actions = Vec::new();
    for ay in &gy.actions {
        let raw = &ay.implementation;
        let impl_type = &raw.impl_type;

        let implementation = if raw.is_file_ref() {
            match impl_type.as_str() {
                "Script" => {
                    let lang = raw.language.clone().unwrap_or_else(|| "python".into());
                    let code = raw
                        .code_file
                        .as_ref()
                        .map(|f| load_code_file(dir, f))
                        .unwrap_or_else(|| Ok(String::new()))?;
                    ActionImpl::Script {
                        language: lang,
                        code,
                        timeout_secs: raw.timeout_secs.unwrap_or(30),
                    }
                }
                "Shell" => {
                    let command = raw
                        .command_file
                        .as_ref()
                        .map(|f| load_code_file(dir, f))
                        .unwrap_or_else(|| Ok(String::new()))?;
                    ActionImpl::Shell {
                        command,
                        timeout_secs: raw.timeout_secs.unwrap_or(30),
                    }
                }
                "Llm" => {
                    let prompt = raw
                        .prompt_file
                        .as_ref()
                        .map(|f| load_code_file(dir, f))
                        .unwrap_or_else(|| Ok(String::new()))?;
                    ActionImpl::Llm {
                        prompt,
                        model: raw
                            .model
                            .clone()
                            .unwrap_or_else(|| "claude-sonnet-4-6".into()),
                        system: raw.system.clone(),
                    }
                }
                "Composite" => {
                    let steps_raw = raw
                        .steps_file
                        .as_ref()
                        .map(|f| load_code_file(dir, f))
                        .unwrap_or_else(|| Ok(String::new()))?;
                    let yaml_val: serde_yaml::Value =
                        serde_yaml::from_str(&steps_raw).unwrap_or_default();
                    let steps_json =
                        serde_json::to_value(yaml_val.get("steps").cloned().unwrap_or_default())
                            .unwrap_or_default();
                    let steps: Vec<CompositeStep> =
                        serde_json::from_value(steps_json).unwrap_or_default();
                    ActionImpl::Composite { steps }
                }
                "Rule" => {
                    let raw_str = raw
                        .template_file
                        .as_ref()
                        .map(|f| load_code_file(dir, f))
                        .unwrap_or_else(|| Ok("{}".into()))?;
                    let yv: serde_yaml::Value = serde_yaml::from_str(&raw_str).unwrap_or_default();
                    let template: serde_json::Value = serde_json::to_value(&yv)
                        .map(|v| v.get("template").cloned().unwrap_or_default())
                        .unwrap_or_default();
                    ActionImpl::Rule { template }
                }
                "Native" => ActionImpl::Native {
                    capability: raw.capability.clone().unwrap_or_default(),
                    action: raw.action.clone().unwrap_or_default(),
                },
                "Custom" => ActionImpl::Custom {
                    executor_type: raw.executor_type.clone().unwrap_or_default(),
                    params: raw.params.clone().unwrap_or_default(),
                },
                _ => return Err(format!("未知实现类型（文件引用）: {}", impl_type)),
            }
        } else {
            match impl_type.as_str() {
                "Script" => ActionImpl::Script {
                    language: raw.language.clone().unwrap_or_else(|| "python".into()),
                    code: raw.code.clone().unwrap_or_default(),
                    timeout_secs: raw.timeout_secs.unwrap_or(30),
                },
                "Shell" => ActionImpl::Shell {
                    command: raw.command.clone().unwrap_or_default(),
                    timeout_secs: raw.timeout_secs.unwrap_or(30),
                },
                "Llm" => ActionImpl::Llm {
                    prompt: raw.prompt.clone().unwrap_or_default(),
                    model: raw
                        .model
                        .clone()
                        .unwrap_or_else(|| "claude-sonnet-4-6".into()),
                    system: raw.system.clone(),
                },
                "Composite" => {
                    let steps: Vec<CompositeStep> = raw
                        .steps
                        .as_ref()
                        .map(|s| {
                            serde_json::from_value(serde_json::Value::Array(s.clone()))
                                .unwrap_or_default()
                        })
                        .unwrap_or_default();
                    ActionImpl::Composite { steps }
                }
                "Rule" => ActionImpl::Rule {
                    template: raw.template.clone().unwrap_or_default(),
                },
                "Native" => ActionImpl::Native {
                    capability: raw.capability.clone().unwrap_or_default(),
                    action: raw.action.clone().unwrap_or_default(),
                },
                "Custom" => ActionImpl::Custom {
                    executor_type: raw.executor_type.clone().unwrap_or_default(),
                    params: raw.params.clone().unwrap_or_default(),
                },
                _ => return Err(format!("未知实现类型（内联）: {}", impl_type)),
            }
        };

        actions.push(ActionGene {
            name: ay.name.clone(),
            description: ay.description.clone(),
            input_schema: ay.input_schema.clone(),
            implementation,
        });
    }

    // 转换 FitnessYaml → FitnessGene
    let mut fitness = FitnessGene::default();
    fitness.score = gy.fitness.score;
    fitness.success_rate = gy.fitness.success_rate;
    fitness.call_count = gy.fitness.call_count;
    fitness.success_count = gy.fitness.success_count;
    fitness.failure_count = gy.fitness.failure_count;
    fitness.auto_test_count = gy.fitness.auto_test_count;
    fitness.avg_latency_ms = gy.fitness.avg_latency_ms;
    fitness.rounds_dormant = gy.fitness.rounds_dormant;
    fitness.output_quality = gy.fitness.output_quality;
    fitness.coverage_score = gy.fitness.coverage_score;
    fitness.dependency_complexity = gy.fitness.dependency_complexity;
    fitness.non_empty_output_count = gy.fitness.non_empty_output_count;
    fitness.innovation_score = gy.fitness.innovation_score;
    fitness.utility_score = gy.fitness.utility_score;
    fitness.last_evaluated = gy.fitness.last_evaluated.clone();
    fitness.llm_evaluated_at = gy.fitness.llm_evaluated_at;
    fitness.real_validation_passes = gy.fitness.real_validation_passes;
    fitness.real_validation_failures = gy.fitness.real_validation_failures;
    fitness.strongest_signal = gy.fitness.strongest_signal;
    fitness.strongest_automatic_signal = gy.fitness.strongest_automatic_signal;
    fitness.human_signals_count = gy.fitness.human_signals_count;
    fitness.human_score = gy.fitness.human_score;
    fitness.total_token_cost = gy.fitness.total_token_cost;
    fitness.last_token_cost = gy.fitness.last_token_cost;
    fitness.profit_ratio = gy.fitness.profit_ratio;

    // YAML 保存的是证据与统计快照；选择用的派生值必须由当前公式重建，
    // 不能信任旧版本或人工编辑留下的 score/success_rate/profit_ratio。
    // 对内部一致的快照，重算仍保持 save/load 幂等。
    fitness.recompute_score();

    // 转换 LineageYaml → LineageGene
    let mut lineage = LineageGene::default();
    lineage.origin = match gy.lineage.origin.as_str() {
        "Native" => Origin::Native,
        "Generated" => Origin::Generated,
        "Mutated" => Origin::Mutated,
        "Crossbred" => Origin::Crossbred,
        _ => Origin::Other,
    };
    lineage.generation = gy.lineage.generation;
    lineage.parent = gy.lineage.parent.clone();
    lineage.mutations = gy
        .lineage
        .mutations
        .iter()
        .map(|m| MutationRecord {
            mutation_type: m.mutation_type.clone(),
            description: m.description.clone(),
            timestamp: m.timestamp.clone().unwrap_or_default(),
        })
        .collect();

    // 测试套件
    let test_suite: Vec<TestCase> = gy
        .test_suite
        .cases
        .iter()
        .filter_map(|c| {
            let case: serde_json::Value = c.clone();
            serde_json::from_value(case).ok()
        })
        .collect();

    Ok(CapabilityGenome {
        name: gy.name,
        version: gy.version,
        description: gy.description,
        actions,
        fitness,
        lineage,
        test_suite,
    })
}

/// 从文件中读取代码内容
fn load_code_file(cap_dir: &Path, relative_path: &str) -> Result<String, String> {
    let path = cap_dir.join(relative_path);
    std::fs::read_to_string(&path).map_err(|e| format!("读取代码文件 {:?} 失败: {}", path, e))
}

// ────────────────────────────────── 保存 ──────────────────────────────────

/// 将所有基因组保存到 YAML 多文件目录
///
/// 注：此函数用于完整导出。增量保存（如仅更新适应度）应直接写 genome.yaml。
pub fn save_genomes_to_yaml_dir(
    dir: &Path,
    genomes: &HashMap<String, CapabilityGenome>,
) -> Result<(), String> {
    std::fs::create_dir_all(dir).map_err(|e| format!("创目录失败: {}", e))?;

    // 事务性写入：先写入 staging 目录，全部成功后原子切换。
    //
    // 崩溃恢复：如果崩溃在写入 staging 期间，目标目录不受影响；
    // 如果崩溃在 rename 之间，重启时 load 会看到 staging 残留并可手动恢复。
    // 这是目录级事务，保证不会出现"部分 genome 更新、部分旧"的混合状态。
    let staging = dir.join("__staging__");
    // 清理可能残留的 staging 目录
    let _ = std::fs::remove_dir_all(&staging);
    std::fs::create_dir_all(&staging).map_err(|e| format!("创 staging 目录失败: {}", e))?;

    // 1. 保存每个能力到 staging 子目录
    let mut manifest_entries = Vec::new();
    for (name, genome) in genomes {
        let safe_name = safe_filename(name);
        let cap_dir = staging.join(&safe_name);
        std::fs::create_dir_all(&cap_dir).map_err(|e| format!("创 staging 子目录失败: {}", e))?;
        std::fs::create_dir_all(cap_dir.join("actions")).ok();
        std::fs::create_dir_all(cap_dir.join("versions")).ok();

        save_genome_to_dir(&cap_dir, genome)?;

        manifest_entries.push(ManifestEntry {
            name: genome.name.clone(),
            version: genome.version.clone(),
            description: genome.description.clone(),
            score: genome.fitness.score,
            success_rate: genome.fitness.success_rate,
            call_count: genome.fitness.call_count,
            rounds_dormant: genome.fitness.rounds_dormant,
            action_count: genome.actions.len() as u32,
            action_names: genome.action_names(),
            directory: safe_name,
        });
    }

    // 2. manifest.yaml → staging
    let manifest = Manifest {
        format_version: "2.0".into(),
        migrated_at: None,
        source: None,
        total_capabilities: genomes.len() as u32,
        capabilities: manifest_entries,
    };
    let manifest_yaml =
        serde_yaml::to_string(&manifest).map_err(|e| format!("序列化 manifest 失败: {}", e))?;
    let manifest_path = staging.join("manifest.yaml");
    atomic_write_file(
        &manifest_path,
        &format!("# 能力库清单\n\n{}", manifest_yaml),
    )?;

    // 3. shared/safety_rules.yaml
    let shared_dir = staging.join("shared");
    let safety_path = shared_dir.join("safety_rules.yaml");
    if !safety_path.exists() {
        std::fs::create_dir_all(&shared_dir).ok();
        let _ = atomic_write_file(&safety_path, crate::genome_yaml::DEFAULT_SAFETY_RULES);
    }

    // 4. 原子切换：逐个 genome 子目录 rename 到目标
    //
    // 先删除目标中不存在于 staging 的目录（被淘汰的能力），
    // 再把 staging 中的目录 rename 到目标。
    // manifest.yaml 和 shared/ 用原子文件替换。
    let existing_names: std::collections::HashSet<String> =
        genomes.keys().map(|k| safe_filename(k)).collect();
    if dir.exists() {
        for entry in std::fs::read_dir(dir).map_err(|e| format!("读目标目录失败: {}", e))? {
            let entry = entry.map_err(|e| format!("读目录项失败: {}", e))?;
            let fname = entry.file_name().to_string_lossy().to_string();
            if fname.starts_with("__") || fname == "manifest.yaml" || fname == "shared" {
                continue;
            }
            if !existing_names.contains(&fname) {
                // 该能力已被淘汰，删除目录
                let _ = std::fs::remove_dir_all(entry.path());
            }
        }
    }

    // rename staging 子目录到目标
    for entry in std::fs::read_dir(&staging).map_err(|e| format!("读 staging 失败: {}", e))? {
        let entry = entry.map_err(|e| format!("读 staging 项失败: {}", e))?;
        let fname = entry.file_name();
        let fname_str = fname.to_string_lossy();
        if fname_str == "manifest.yaml" || fname_str == "shared" {
            // manifest 和 shared 用原子文件写入目标
            let target_path = dir.join(fname_str.as_ref());
            if fname_str == "shared" {
                // shared 目录：逐文件原子写入
                let target_shared = dir.join("shared");
                std::fs::create_dir_all(&target_shared).ok();
                if let Ok(shared_entries) = std::fs::read_dir(entry.path()) {
                    for se in shared_entries {
                        if let Ok(se) = se {
                            let src = se.path();
                            let dst = target_shared.join(se.file_name());
                            if !dst.exists() {
                                // 只复制不存在的文件（不覆盖已有）
                                let _ = std::fs::copy(&src, &dst);
                            }
                        }
                    }
                }
            } else {
                // manifest.yaml：原子替换
                let content = std::fs::read_to_string(&entry.path())
                    .map_err(|e| format!("读 staging manifest 失败: {}", e))?;
                atomic_write_file(&target_path, &content)?;
            }
        } else {
            // genome 子目录：rename 到目标（覆盖同名）
            let target = dir.join(fname_str.as_ref());
            if target.exists() {
                let _ = std::fs::remove_dir_all(&target);
            }
            std::fs::rename(entry.path(), &target)
                .map_err(|e| format!("rename genome 目录失败: {}", e))?;
        }
    }

    // 5. 清理 staging
    let _ = std::fs::remove_dir_all(&staging);

    Ok(())
}

/// 将单个 genome 保存到其 YAML 子目录
fn save_genome_to_dir(dir: &Path, genome: &CapabilityGenome) -> Result<(), String> {
    // 构建 ActionYaml 列表 + 写入代码文件
    let mut actions_yaml = Vec::new();

    for action in &genome.actions {
        let ay = action_to_yaml(dir, &genome.name, action)?;
        actions_yaml.push(ay);
    }

    let gy = GenomeYaml {
        name: genome.name.clone(),
        version: genome.version.clone(),
        description: genome.description.clone(),
        actions: actions_yaml,
        fitness: FitnessYaml {
            score: genome.fitness.score,
            success_rate: genome.fitness.success_rate,
            call_count: genome.fitness.call_count,
            success_count: genome.fitness.success_count,
            failure_count: genome.fitness.failure_count,
            auto_test_count: genome.fitness.auto_test_count,
            real_call_count: genome.fitness.real_call_count(),
            avg_latency_ms: genome.fitness.avg_latency_ms,
            rounds_dormant: genome.fitness.rounds_dormant,
            output_quality: genome.fitness.output_quality,
            coverage_score: genome.fitness.coverage_score,
            dependency_complexity: genome.fitness.dependency_complexity,
            non_empty_output_count: genome.fitness.non_empty_output_count,
            innovation_score: genome.fitness.innovation_score,
            utility_score: genome.fitness.utility_score,
            last_evaluated: genome.fitness.last_evaluated.clone(),
            llm_evaluated_at: genome.fitness.llm_evaluated_at,
            real_validation_passes: genome.fitness.real_validation_passes,
            real_validation_failures: genome.fitness.real_validation_failures,
            strongest_signal: genome.fitness.strongest_signal,
            strongest_automatic_signal: genome.fitness.strongest_automatic_signal,
            human_signals_count: genome.fitness.human_signals_count,
            human_score: genome.fitness.human_score,
            total_token_cost: genome.fitness.total_token_cost,
            last_token_cost: genome.fitness.last_token_cost,
            profit_ratio: genome.fitness.profit_ratio,
        },
        lineage: LineageYaml {
            origin: match genome.lineage.origin {
                Origin::Native => "Native".into(),
                Origin::Generated => "Generated".into(),
                Origin::Mutated => "Mutated".into(),
                Origin::Crossbred => "Crossbred".into(),
                Origin::Other => "Other".into(),
            },
            generation: genome.lineage.generation,
            parent: genome.lineage.parent.clone(),
            mutations: genome
                .lineage
                .mutations
                .iter()
                .map(|m| MutationYaml {
                    mutation_type: m.mutation_type.clone(),
                    description: m.description.clone(),
                    timestamp: Some(m.timestamp.clone()),
                })
                .collect(),
        },
        test_suite: TestSuiteYaml {
            cases: genome
                .test_suite
                .iter()
                .map(|t| serde_json::to_value(t).unwrap_or_default())
                .collect(),
            count: genome.test_suite.len() as u32,
        },
    };

    let genome_yaml = serde_yaml::to_string(&gy)
        .map_err(|e| format!("序列化 genome.yaml ({}): {}", genome.name, e))?;
    let genome_yaml_path = dir.join("genome.yaml");
    atomic_write_file(
        &genome_yaml_path,
        &format!("# {} — 能力基因组\n\n{}", genome.name, genome_yaml),
    )?;

    Ok(())
}

/// 将 action 转为 YAML + 写入代码文件
fn action_to_yaml(dir: &Path, _cap_name: &str, action: &ActionGene) -> Result<ActionYaml, String> {
    let actions_dir = dir.join("actions");
    let action_name = safe_filename(&action.name);

    let (impl_ref, _written) = match &action.implementation {
        ActionImpl::Script {
            language,
            code,
            timeout_secs,
        } => {
            let ext = if language == "python" || language == "py" {
                "py"
            } else {
                "js"
            };
            let filename = format!("{}.{}", action_name, ext);
            let path = actions_dir.join(&filename);
            atomic_write_file(&path, code)?;
            (
                RawImplRef {
                    impl_type: "Script".into(),
                    code_file: Some(format!("actions/{}", filename)),
                    language: Some(language.clone()),
                    timeout_secs: Some(*timeout_secs),
                    code: None,
                    command: None,
                    command_file: None,
                    prompt: None,
                    prompt_file: None,
                    model: None,
                    system: None,
                    steps: None,
                    steps_file: None,
                    template: None,
                    template_file: None,
                    capability: None,
                    action: None,
                    executor_type: None,
                    params: None,
                },
                true,
            )
        }
        ActionImpl::Shell {
            command,
            timeout_secs,
        } => {
            let filename = format!("{}.sh", action_name);
            let path = actions_dir.join(&filename);
            atomic_write_file(&path, command)?;
            (
                RawImplRef {
                    impl_type: "Shell".into(),
                    command_file: Some(format!("actions/{}", filename)),
                    timeout_secs: Some(*timeout_secs),
                    code: None,
                    code_file: None,
                    language: None,
                    command: None,
                    prompt: None,
                    prompt_file: None,
                    model: None,
                    system: None,
                    steps: None,
                    steps_file: None,
                    template: None,
                    template_file: None,
                    capability: None,
                    action: None,
                    executor_type: None,
                    params: None,
                },
                true,
            )
        }
        ActionImpl::Llm {
            prompt,
            model,
            system,
        } => {
            let filename = format!("{}.md", action_name);
            let path = actions_dir.join(&filename);
            atomic_write_file(&path, prompt)?;
            (
                RawImplRef {
                    impl_type: "Llm".into(),
                    prompt_file: Some(format!("actions/{}", filename)),
                    model: Some(model.clone()),
                    system: system.clone(),
                    code: None,
                    code_file: None,
                    language: None,
                    timeout_secs: None,
                    command: None,
                    command_file: None,
                    prompt: None,
                    steps: None,
                    steps_file: None,
                    template: None,
                    template_file: None,
                    capability: None,
                    action: None,
                    executor_type: None,
                    params: None,
                },
                true,
            )
        }
        ActionImpl::Composite { steps } => {
            let filename = format!("{}.yaml", action_name);
            let path = actions_dir.join(&filename);
            let val = serde_json::to_value(steps).unwrap_or_default();
            let steps_yaml = serde_yaml::to_string(&serde_json::json!({"steps": val}))
                .map_err(|e| format!("序列化 steps: {}", e))?;
            atomic_write_file(&path, &steps_yaml)?;
            (
                RawImplRef {
                    impl_type: "Composite".into(),
                    steps_file: Some(format!("actions/{}", filename)),
                    code: None,
                    code_file: None,
                    language: None,
                    timeout_secs: None,
                    command: None,
                    command_file: None,
                    prompt: None,
                    prompt_file: None,
                    model: None,
                    system: None,
                    steps: None,
                    template: None,
                    template_file: None,
                    capability: None,
                    action: None,
                    executor_type: None,
                    params: None,
                },
                true,
            )
        }
        ActionImpl::Rule { template } => {
            let filename = format!("{}.yaml", action_name);
            let path = actions_dir.join(&filename);
            let rule_yaml = serde_yaml::to_string(&serde_json::json!({"template": template}))
                .map_err(|e| format!("序列化 rule: {}", e))?;
            atomic_write_file(&path, &rule_yaml)?;
            (
                RawImplRef {
                    impl_type: "Rule".into(),
                    template_file: Some(format!("actions/{}", filename)),
                    code: None,
                    code_file: None,
                    language: None,
                    timeout_secs: None,
                    command: None,
                    command_file: None,
                    prompt: None,
                    prompt_file: None,
                    model: None,
                    system: None,
                    steps: None,
                    steps_file: None,
                    template: None,
                    capability: None,
                    action: None,
                    executor_type: None,
                    params: None,
                },
                true,
            )
        }
        ActionImpl::Native {
            capability,
            action: native_action,
        } => (
            RawImplRef {
                impl_type: "Native".into(),
                capability: Some(capability.clone()),
                action: Some(native_action.clone()),
                code: None,
                code_file: None,
                language: None,
                timeout_secs: None,
                command: None,
                command_file: None,
                prompt: None,
                prompt_file: None,
                model: None,
                system: None,
                steps: None,
                steps_file: None,
                template: None,
                template_file: None,
                executor_type: None,
                params: None,
            },
            false,
        ),
        ActionImpl::Custom {
            executor_type,
            params,
        } => (
            RawImplRef {
                impl_type: "Custom".into(),
                executor_type: Some(executor_type.clone()),
                params: Some(params.clone()),
                code: None,
                code_file: None,
                language: None,
                timeout_secs: None,
                command: None,
                command_file: None,
                prompt: None,
                prompt_file: None,
                model: None,
                system: None,
                steps: None,
                steps_file: None,
                template: None,
                template_file: None,
                capability: None,
                action: None,
            },
            false,
        ),
    };

    Ok(ActionYaml {
        name: action.name.clone(),
        description: action.description.clone(),
        input_schema: action.input_schema.clone(),
        implementation: impl_ref,
    })
}

fn safe_filename(name: &str) -> String {
    name.replace('/', "_")
        .replace('\\', "_")
        .replace(':', "_")
        .replace(' ', "_")
}

fn atomic_write_file(path: &Path, content: &str) -> Result<(), String> {
    use std::io::Write;

    let tmp = path.with_extension(format!("tmp-{}", uuid::Uuid::new_v4()));
    let result = (|| -> Result<(), String> {
        let mut file = std::fs::OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&tmp)
            .map_err(|e| format!("创建临时文件 {} 失败: {}", tmp.display(), e))?;
        file.write_all(content.as_bytes())
            .map_err(|e| format!("写入临时文件 {} 失败: {}", tmp.display(), e))?;
        file.sync_all()
            .map_err(|e| format!("同步临时文件 {} 失败: {}", tmp.display(), e))?;
        std::fs::rename(&tmp, path).map_err(|e| {
            format!(
                "原子替换 {} -> {} 失败: {}",
                tmp.display(),
                path.display(),
                e
            )
        })?;
        Ok(())
    })();
    if result.is_err() {
        let _ = std::fs::remove_file(&tmp);
    }
    result
}

// ────────────────────────────────── 辅助 ──────────────────────────────────

/// 检测目录是否为 YAML 多文件格式（有 manifest.yaml 或至少一个 genome.yaml）
pub fn is_yaml_evolution_dir(dir: &Path) -> bool {
    dir.join("manifest.yaml").exists()
        || (dir.is_dir()
            && std::fs::read_dir(dir)
                .map(|mut entries| {
                    entries.any(|e| {
                        e.ok()
                            .map(|entry| {
                                entry.path().is_dir() && entry.path().join("genome.yaml").exists()
                            })
                            .unwrap_or(false)
                    })
                })
                .unwrap_or(false))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_yaml_load_basic() {
        // 用一个最小的内联 genome 测试解析
        let yaml = r#"
name: test_ops
version: "0.1.0"
description: 测试能力
actions:
  - name: hello
    description: say hello
    implementation:
      type: Script
      language: python
      code: |
        print("hello")
      timeout_secs: 10
fitness:
  score: 0.5
  success_rate: 1.0
lineage:
  origin: Native
  mutations: []
test_suite:
  cases: []
"#;
        let gy: GenomeYaml = serde_yaml::from_str(yaml).expect("解析失败");
        assert_eq!(gy.name, "test_ops");
        assert_eq!(gy.actions.len(), 1);
        let raw = &gy.actions[0].implementation;
        assert_eq!(raw.impl_type, "Script");
        // 有 code 而没有 code_file → 应该是内联
        assert!(!raw.is_file_ref(), "应为内联实现，非文件引用");
        assert!(raw.code.as_ref().unwrap().contains("hello"));
        // 旧版 YAML 没有真实/人类信号字段时仍可以加载。
        assert_eq!(gy.fitness.real_validation_failures, 0);
        assert_eq!(
            gy.fitness.strongest_signal,
            crate::validator::SignalStrength::SelfReport
        );
        assert_eq!(gy.fitness.human_signals_count, 0);
        assert_eq!(gy.fitness.human_score, 0.0);
    }

    #[test]
    fn test_yaml_load_file_ref() {
        let yaml = r#"
name: git_ops
version: "0.1.0"
description: Git ops
actions:
  - name: status
    description: git status
    implementation:
      type: Script
      code_file: actions/status.py
      language: python
      timeout_secs: 10
fitness:
  score: 0.9
lineage:
  origin: Mutated
"#;
        let gy: GenomeYaml = serde_yaml::from_str(yaml).expect("解析失败");
        let raw = &gy.actions[0].implementation;
        assert_eq!(raw.impl_type, "Script");
        // 有 code_file → 是文件引用
        assert!(raw.is_file_ref(), "应为文件引用");
        assert_eq!(raw.code_file.as_ref().unwrap(), "actions/status.py");
    }

    #[test]
    fn test_fitness_yaml_roundtrip_preserves_all_statistics() {
        let temp_dir = tempfile::TempDir::new().expect("创建临时目录失败");
        let mut expected = FitnessGene {
            call_count: 20,
            auto_test_count: 3,
            success_count: 16,
            failure_count: 4,
            success_rate: 0.8,
            avg_latency_ms: 123.5,
            score: 0.8765,
            last_evaluated: Some("2026-07-11T12:00:00Z".into()),
            rounds_dormant: 2,
            output_quality: 0.75,
            coverage_score: 0.456,
            dependency_complexity: 0.12,
            non_empty_output_count: 15,
            innovation_score: 0.88,
            utility_score: 0.91,
            llm_evaluated_at: 1_752_230_400,
            real_validation_passes: 7,
            real_validation_failures: 3,
            strongest_signal: crate::validator::SignalStrength::HumanValue,
            strongest_automatic_signal: crate::validator::SignalStrength::TestPass,
            human_signals_count: 4,
            human_score: 0.75,
            total_token_cost: 12_345,
            last_token_cost: 321,
            profit_ratio: 0.001_234,
        };
        expected.recompute_score();

        let mut genome = CapabilityGenome::new("fitness_roundtrip", "fitness 持久化测试");
        genome.fitness = expected.clone();
        let mut genomes = HashMap::new();
        genomes.insert(genome.name.clone(), genome);

        save_genomes_to_yaml_dir(temp_dir.path(), &genomes).expect("保存 YAML 失败");
        let loaded = load_genomes_from_yaml_dir(temp_dir.path()).expect("重新加载 YAML 失败");
        assert_eq!(loaded.len(), 1);

        // 通过 serde 值比较覆盖 FitnessGene 的每个字段，防止未来新增
        // 统计项时只改了内存结构，却遗漏 YAML 的保存或加载映射。
        assert_eq!(
            serde_json::to_value(&loaded[0].fitness).unwrap(),
            serde_json::to_value(&expected).unwrap()
        );
    }
}
