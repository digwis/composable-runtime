//! 进化驱动抽象 — 统一 API 直连与 opencode CLI 两种 LLM 后端
//!
//! 设计目标：让进化循环可以零成本切换 LLM 后端。
//! - `LlmExecutor`：原有 API 直连实现（OpenAI/Anthropic 兼容）
//! - `OpenCodeCliDriver`：通过 `opencode run` CLI 调用内置免费模型
//!
//! 最大化利用免费 CLI 的策略：
//! 1. 模型 fallback 链 — 主模型失败自动切下一个免费模型
//! 2. 多轮对话合并 — execute_conversation 的 follow_ups 合并成单次调用，省冷启动
//! 3. JSON 剥壳 — 自动去除 markdown ```json``` 包裹，保证结构化输出可直接 parse
//! 4. 超时控制 — 单次调用限时，避免进化循环卡死
//! 5. 标签统一映射 — fast/smart/coder 路由标签在免费模型下统一指向主模型（免费模型不分层）

use crate::genome::LlmExecutor;
use async_trait::async_trait;

/// 进化驱动 trait — 所有 LLM 后端的统一抽象
#[async_trait]
pub trait EvolutionDriver: Send + Sync {
    /// 是否有可用的 LLM 后端
    fn has_llm_backend(&self) -> bool;

    /// 单次调用
    async fn execute(
        &self,
        prompt: &str,
        model: &str,
        system: Option<&str>,
    ) -> Result<String, String>;

    /// 多轮深度推理对话
    async fn execute_conversation(
        &self,
        initial_prompt: &str,
        model: &str,
        system: Option<&str>,
        follow_ups: &[&str],
    ) -> Result<String, String>;

    /// 熔断器句柄(None = 该 driver 无熔断器,如 CLI driver)。
    fn breaker(&self) -> Option<std::sync::Arc<crate::llm_health::LlmCircuitBreaker>> {
        None
    }
}

/// 把现有的 LlmExecutor 适配为 EvolutionDriver
#[async_trait]
impl EvolutionDriver for LlmExecutor {
    fn has_llm_backend(&self) -> bool {
        LlmExecutor::has_llm_backend(self)
    }

    async fn execute(
        &self,
        prompt: &str,
        model: &str,
        system: Option<&str>,
    ) -> Result<String, String> {
        LlmExecutor::execute(self, prompt, model, system).await
    }

    async fn execute_conversation(
        &self,
        initial_prompt: &str,
        model: &str,
        system: Option<&str>,
        follow_ups: &[&str],
    ) -> Result<String, String> {
        LlmExecutor::execute_conversation(self, initial_prompt, model, system, follow_ups).await
    }

    fn breaker(&self) -> Option<std::sync::Arc<crate::llm_health::LlmCircuitBreaker>> {
        Some(LlmExecutor::breaker(self))
    }
}

/// opencode CLI 驱动 — 利用 opencode 内置免费模型驱动进化
///
/// 冷启动约 10s/次，适合低频高质量任务。通过 fallback 链和对话合并
/// 最大化可用性与效率。
pub struct OpenCodeCliDriver {
    /// 主模型（默认 opencode/north-mini-code-free — 实测最省 token 且 JSON 直接可用）
    primary_model: String,
    /// fallback 模型链（主模型失败时依次尝试）
    fallback_models: Vec<String>,
    /// 单次调用超时（秒）
    timeout_secs: u64,
    /// opencode 二进制路径
    binary: String,
}

impl OpenCodeCliDriver {
    /// 从环境变量构造，所有参数可配置：
    /// - `ORCH_OPENCODE_MODEL`：主模型（默认 north-mini-code-free）
    /// - `ORCH_OPENCODE_FALLBACK`：逗号分隔的 fallback 链
    /// - `ORCH_OPENCODE_TIMEOUT`：超时秒数（默认 120）
    /// - `ORCH_OPENCODE_BIN`：opencode 路径（默认 opencode）
    pub fn new() -> Self {
        let primary_model = std::env::var("ORCH_OPENCODE_MODEL")
            .unwrap_or_else(|_| "opencode/north-mini-code-free".to_string());
        let fallback = std::env::var("ORCH_OPENCODE_FALLBACK").unwrap_or_else(|_| {
            "opencode/hy3-free,opencode/nemotron-3-ultra-free,opencode/deepseek-v4-flash-free"
                .to_string()
        });
        let fallback_models: Vec<String> = fallback
            .split(',')
            .map(|s| s.trim().to_string())
            .filter(|s| !s.is_empty())
            .collect();
        let timeout_secs = std::env::var("ORCH_OPENCODE_TIMEOUT")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(120);
        let binary = std::env::var("ORCH_OPENCODE_BIN").unwrap_or_else(|_| "opencode".to_string());
        Self {
            primary_model,
            fallback_models,
            timeout_secs,
            binary,
        }
    }

    /// 解析模型路由标签为 fallback 链
    ///
    /// 免费模型不分 fast/smart/coder 层级，所有标签统一用 primary + fallback。
    /// 这样无论进化循环用哪个标签（smart:attribution / coder:novel / fast:testinput），
    /// 都走同一条免费模型链。
    fn resolve_chain(&self, _model: &str) -> Vec<String> {
        let mut chain = vec![self.primary_model.clone()];
        for m in &self.fallback_models {
            if !chain.contains(m) {
                chain.push(m.clone());
            }
        }
        chain
    }

    /// 单次 opencode CLI 调用
    async fn call_once(&self, model: &str, prompt: &str) -> Result<String, String> {
        let timeout = std::time::Duration::from_secs(self.timeout_secs);
        let prompt_owned = prompt.to_string();

        let fut = async {
            let mut cmd = tokio::process::Command::new(&self.binary);
            cmd.args([
                "--print-logs",
                "--log-level",
                "ERROR",
                "run",
                "--format",
                "json",
                "--model",
                model,
                &prompt_owned,
            ]);
            cmd.stdout(std::process::Stdio::piped());
            cmd.stderr(std::process::Stdio::piped());
            cmd.stdin(std::process::Stdio::null());

            let output = cmd
                .output()
                .await
                .map_err(|e| format!("启动 opencode 失败: {}", e))?;

            if !output.status.success() {
                let stderr = String::from_utf8_lossy(&output.stderr);
                return Err(format!(
                    "opencode 退出码 {:?}: {}",
                    output.status.code(),
                    stderr.chars().take(500).collect::<String>()
                ));
            }

            let stdout = String::from_utf8_lossy(&output.stdout);
            let mut texts = Vec::new();
            for line in stdout.lines() {
                let line = line.trim();
                if line.is_empty() {
                    continue;
                }
                let ev: serde_json::Value = match serde_json::from_str(line) {
                    Ok(v) => v,
                    Err(_) => continue,
                };
                match ev.get("type").and_then(|t| t.as_str()) {
                    Some("text") => {
                        if let Some(text) = ev.pointer("/part/text").and_then(|t| t.as_str()) {
                            texts.push(text.to_string());
                        }
                    }
                    Some("error") => {
                        let msg = ev
                            .pointer("/error/data/message")
                            .and_then(|t| t.as_str())
                            .unwrap_or("unknown");
                        return Err(format!("opencode error: {}", msg));
                    }
                    _ => {}
                }
            }

            let text = texts.join("").trim().to_string();
            if text.is_empty() {
                return Err("opencode 返回空文本".to_string());
            }
            Ok(strip_markdown_fence(&text).to_string())
        };

        tokio::time::timeout(timeout, fut)
            .await
            .map_err(|_| format!("opencode 调用超时 ({}s)", self.timeout_secs))?
    }

    /// 遍历 fallback 链执行，返回第一个成功的结果
    async fn call_with_fallback(&self, prompt: &str, model: &str) -> Result<String, String> {
        let chain = self.resolve_chain(model);
        let mut last_err = String::new();
        for m in &chain {
            match self.call_once(m, prompt).await {
                Ok(text) => return Ok(text),
                Err(e) => {
                    tracing::warn!("opencode 模型 {} 失败，尝试 fallback: {}", m, e);
                    if last_err.is_empty() {
                        last_err = format!("[{}]: {}", m, e);
                    } else {
                        last_err.push_str(&format!(" | [{}]: {}", m, e));
                    }
                }
            }
        }
        Err(format!("所有 opencode 模型均失败: {}", last_err))
    }
}

impl Default for OpenCodeCliDriver {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl EvolutionDriver for OpenCodeCliDriver {
    fn has_llm_backend(&self) -> bool {
        // 免费模型不需要 api_key，永远可用
        true
    }

    async fn execute(
        &self,
        prompt: &str,
        model: &str,
        system: Option<&str>,
    ) -> Result<String, String> {
        let full_prompt = match system {
            Some(s) => format!("{}\n\n{}", s, prompt),
            None => prompt.to_string(),
        };
        self.call_with_fallback(&full_prompt, model).await
    }

    async fn execute_conversation(
        &self,
        initial_prompt: &str,
        model: &str,
        system: Option<&str>,
        follow_ups: &[&str],
    ) -> Result<String, String> {
        // CLI 模式下合并多轮为单次调用 — 省去多次冷启动开销
        // opencode 的 agent loop 本身就会多轮思考，合并 prompt 后单次调用即可
        let mut full = match system {
            Some(s) => format!("{}\n\n{}", s, initial_prompt),
            None => initial_prompt.to_string(),
        };
        if !follow_ups.is_empty() {
            full.push_str("\n\n请进一步深入分析以下要点，并最终给出结构化结论：\n");
            for (i, fu) in follow_ups.iter().enumerate() {
                full.push_str(&format!("{}. {}\n", i + 1, fu));
            }
            full.push_str("\n最终请只输出结构化 JSON 结论。");
        }
        self.call_with_fallback(&full, model).await
    }
}

/// 剥除 markdown 代码块包裹（```json ... ``` 或 ``` ... ```）
///
/// 实测部分免费模型（如 deepseek-v4-flash-free）会用 markdown 包裹 JSON，
/// 导致 serde_json::from_str 解析失败。这里统一剥壳。
fn strip_markdown_fence(s: &str) -> &str {
    let s = s.trim();
    if !s.starts_with("```") {
        return s;
    }
    // 跳过第一行（```json 或 ```）
    let after_first = match s.find('\n') {
        Some(idx) => &s[idx + 1..],
        None => return s,
    };
    // 找最后一个 ``` 并截断
    if let Some(end) = after_first.rfind("```") {
        after_first[..end].trim()
    } else {
        after_first.trim()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_strip_fence_json() {
        let input = "```json\n{\"a\": 1}\n```";
        assert_eq!(strip_markdown_fence(input), "{\"a\": 1}");
    }

    #[test]
    fn test_strip_fence_plain() {
        let input = "```\nhello\n```";
        assert_eq!(strip_markdown_fence(input), "hello");
    }

    #[test]
    fn test_strip_no_fence() {
        let input = "{\"a\": 1}";
        assert_eq!(strip_markdown_fence(input), "{\"a\": 1}");
    }

    #[test]
    fn test_resolve_chain_dedup() {
        let driver = OpenCodeCliDriver {
            primary_model: "opencode/a".to_string(),
            fallback_models: vec!["opencode/a".to_string(), "opencode/b".to_string()],
            timeout_secs: 10,
            binary: "opencode".to_string(),
        };
        let chain = driver.resolve_chain("smart:foo");
        assert_eq!(chain, vec!["opencode/a", "opencode/b"]);
    }
}
