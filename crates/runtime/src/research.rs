//! Evidence-first research orchestration.
//!
//! This module owns source selection, provenance, caching and trust scoring.
//! Coding agents may consume the returned evidence, but their prose is never
//! treated as a source by itself.

use futures::StreamExt;
use reqwest::Client;
use serde::{Deserialize, Serialize};
use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, OnceLock};
use std::time::{SystemTime, UNIX_EPOCH};
use tokio::sync::{OwnedSemaphorePermit, Semaphore};

static RESEARCH_WORKER_POOL: OnceLock<Arc<Semaphore>> = OnceLock::new();
static RESEARCH_RUNNING: AtomicUsize = AtomicUsize::new(0);
static RESEARCH_WAITING: AtomicUsize = AtomicUsize::new(0);
static RESEARCH_COMPLETED: AtomicUsize = AtomicUsize::new(0);
static RESEARCH_FAILED: AtomicUsize = AtomicUsize::new(0);

fn research_concurrency() -> usize {
    std::env::var("ORCH_RESEARCH_WORKER_CONCURRENCY")
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .unwrap_or(4)
        .clamp(1, 16)
}

fn research_worker_pool() -> &'static Arc<Semaphore> {
    RESEARCH_WORKER_POOL.get_or_init(|| Arc::new(Semaphore::new(research_concurrency())))
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResearchWorkerPoolStatus {
    pub max_workers: usize,
    pub available_workers: usize,
    pub running: usize,
    pub queued: usize,
    pub completed: usize,
    pub failed: usize,
}

pub fn research_worker_pool_status() -> ResearchWorkerPoolStatus {
    ResearchWorkerPoolStatus {
        max_workers: research_concurrency(),
        available_workers: research_worker_pool().available_permits(),
        running: RESEARCH_RUNNING.load(Ordering::Relaxed),
        queued: RESEARCH_WAITING.load(Ordering::Relaxed),
        completed: RESEARCH_COMPLETED.load(Ordering::Relaxed),
        failed: RESEARCH_FAILED.load(Ordering::Relaxed),
    }
}

struct WaitingResearchWorker;

impl WaitingResearchWorker {
    fn new() -> Self {
        RESEARCH_WAITING.fetch_add(1, Ordering::Relaxed);
        Self
    }
}

impl Drop for WaitingResearchWorker {
    fn drop(&mut self) {
        RESEARCH_WAITING.fetch_sub(1, Ordering::Relaxed);
    }
}

struct ResearchWorkerPermit {
    _permit: OwnedSemaphorePermit,
    succeeded: bool,
}

impl ResearchWorkerPermit {
    fn mark_succeeded(&mut self) {
        self.succeeded = true;
    }
}

impl Drop for ResearchWorkerPermit {
    fn drop(&mut self) {
        RESEARCH_RUNNING.fetch_sub(1, Ordering::Relaxed);
        if self.succeeded {
            RESEARCH_COMPLETED.fetch_add(1, Ordering::Relaxed);
        } else {
            RESEARCH_FAILED.fetch_add(1, Ordering::Relaxed);
        }
    }
}

async fn acquire_research_worker() -> Result<ResearchWorkerPermit, String> {
    let waiting = WaitingResearchWorker::new();
    let permit = research_worker_pool()
        .clone()
        .acquire_owned()
        .await
        .map_err(|_| "Research Worker Pool 已关闭".to_string())?;
    drop(waiting);
    RESEARCH_RUNNING.fetch_add(1, Ordering::Relaxed);
    Ok(ResearchWorkerPermit {
        _permit: permit,
        succeeded: false,
    })
}

#[derive(Debug, Clone, Deserialize)]
pub struct ResearchRequest {
    #[serde(default)]
    pub urls: Vec<String>,
    #[serde(default)]
    pub query: Option<String>,
    #[serde(default)]
    pub max_sources: Option<usize>,
    #[serde(default)]
    pub force_refresh: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EvidenceItem {
    pub url: String,
    pub source_type: String,
    pub title: String,
    pub publisher: String,
    pub status: u16,
    pub excerpt: String,
    pub content_hash: String,
    pub fetched_at: u64,
    pub relevance: f64,
    pub authority: f64,
    pub freshness: f64,
    pub confidence: f64,
    pub cached: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResearchResult {
    pub query: Option<String>,
    pub evidence: Vec<EvidenceItem>,
    pub warnings: Vec<String>,
    pub generated_at: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct CachedEvidence {
    fetched_at: u64,
    item: EvidenceItem,
}

pub struct ResearchEngine {
    client: Client,
    cache_dir: PathBuf,
    ttl_secs: u64,
}

impl ResearchEngine {
    pub fn new(storage_dir: impl AsRef<Path>) -> Self {
        Self {
            client: Client::builder()
                .timeout(std::time::Duration::from_secs(25))
                .user_agent("evolution-runtime-research/0.1")
                .build()
                .expect("构建研究 HTTP 客户端失败"),
            cache_dir: storage_dir.as_ref().join("research_cache"),
            ttl_secs: std::env::var("ORCH_RESEARCH_CACHE_TTL_SECS")
                .ok()
                .and_then(|v| v.parse().ok())
                .unwrap_or(3600),
        }
    }

    pub async fn research(&self, request: ResearchRequest) -> ResearchResult {
        let max_sources = request.max_sources.unwrap_or(8).clamp(1, 20);
        let mut urls = request
            .urls
            .into_iter()
            .filter_map(|url| normalize_url(&url))
            .collect::<Vec<_>>();
        let mut warnings = Vec::new();

        if let Some(query) = request
            .query
            .as_deref()
            .map(str::trim)
            .filter(|q| !q.is_empty())
        {
            match self.github_search_urls(query, max_sources).await {
                Ok(found) => urls.extend(found),
                Err(error) => warnings.push(format!("GitHub 检索失败: {}", error)),
            }
        }
        urls.sort();
        urls.dedup();
        urls.truncate(max_sources);

        let fetches = futures::stream::iter(urls.into_iter().map(|url| async move {
            let result = self.fetch_evidence(&url, request.force_refresh).await;
            (url, result)
        }));
        let fetched = fetches
            .buffer_unordered(research_concurrency())
            .collect::<Vec<_>>()
            .await;
        let mut evidence = Vec::new();
        for (url, result) in fetched {
            match result {
                Ok(item) => evidence.push(item),
                Err(error) => warnings.push(format!("{}: {}", url, error)),
            }
        }
        evidence.sort_by(|a, b| b.confidence.total_cmp(&a.confidence));
        ResearchResult {
            query: request.query,
            evidence,
            warnings,
            generated_at: unix_now(),
        }
    }

    async fn github_search_urls(
        &self,
        query: &str,
        max_sources: usize,
    ) -> Result<Vec<String>, String> {
        let mut worker = acquire_research_worker().await?;
        let endpoint = format!(
            "https://api.github.com/search/repositories?q={}&per_page={}",
            percent_encode(query),
            max_sources.min(10)
        );
        let response = self
            .client
            .get(endpoint)
            .send()
            .await
            .map_err(|e| e.to_string())?;
        if !response.status().is_success() {
            return Err(format!("HTTP {}", response.status()));
        }
        let value: serde_json::Value = response.json().await.map_err(|e| e.to_string())?;
        let urls = value
            .get("items")
            .and_then(|v| v.as_array())
            .into_iter()
            .flatten()
            .filter_map(|item| {
                item.get("html_url")
                    .and_then(|v| v.as_str())
                    .map(str::to_string)
            })
            .collect();
        worker.mark_succeeded();
        Ok(urls)
    }

    async fn fetch_evidence(&self, url: &str, force_refresh: bool) -> Result<EvidenceItem, String> {
        let cache_path = self.cache_dir.join(format!("{}.json", hash(url)));
        if !force_refresh {
            if let Ok(raw) = std::fs::read_to_string(&cache_path) {
                if let Ok(mut cached) = serde_json::from_str::<CachedEvidence>(&raw) {
                    if unix_now().saturating_sub(cached.fetched_at) <= self.ttl_secs {
                        cached.item.cached = true;
                        return Ok(cached.item);
                    }
                }
            }
        }
        std::fs::create_dir_all(&self.cache_dir).map_err(|e| e.to_string())?;
        let mut worker = acquire_research_worker().await?;
        let response = self
            .client
            .get(url)
            .send()
            .await
            .map_err(|e| e.to_string())?;
        let status = response.status().as_u16();
        let final_url = response.url().to_string();
        let body = response.text().await.map_err(|e| e.to_string())?;
        if status >= 400 {
            return Err(format!("HTTP {}", status));
        }
        let excerpt = extract_excerpt(&body);
        let source_type = classify_source(&final_url);
        let publisher = publisher_for(&final_url);
        let title = extract_title(&body).unwrap_or_else(|| final_url.clone());
        let authority = authority_score(&source_type);
        let freshness = 0.7;
        let item = EvidenceItem {
            url: final_url,
            source_type,
            title,
            publisher,
            status,
            content_hash: hash(&body),
            excerpt,
            fetched_at: unix_now(),
            relevance: 0.7,
            authority,
            freshness,
            confidence: (authority * 0.55 + freshness * 0.2 + 0.7 * 0.25).clamp(0.0, 1.0),
            cached: false,
        };
        let cached = CachedEvidence {
            fetched_at: item.fetched_at,
            item: item.clone(),
        };
        let _ = std::fs::write(
            cache_path,
            serde_json::to_vec_pretty(&cached).unwrap_or_default(),
        );
        worker.mark_succeeded();
        Ok(item)
    }
}

fn normalize_url(raw: &str) -> Option<String> {
    let value = raw.trim();
    if !(value.starts_with("https://") || value.starts_with("http://")) {
        return None;
    }
    url_safety_check(value).then(|| value.to_string())
}

fn url_safety_check(url: &str) -> bool {
    !url.contains('@')
        && !url.contains("localhost")
        && !url.contains("127.0.0.1")
        && !url.contains("[::1]")
}

fn classify_source(url: &str) -> String {
    if url.contains("api.github.com") {
        "github_api".into()
    } else if url.contains("github.com") {
        "github".into()
    } else if url.contains("docs.rs")
        || url.contains("developer.mozilla.org")
        || url.contains("docs.python.org")
    {
        "official_docs".into()
    } else {
        "web".into()
    }
}

fn publisher_for(url: &str) -> String {
    url.split('/').nth(2).unwrap_or_default().to_string()
}

fn authority_score(source_type: &str) -> f64 {
    match source_type {
        "official_docs" | "github_api" => 0.95,
        "github" => 0.85,
        _ => 0.45,
    }
}

fn extract_title(body: &str) -> Option<String> {
    let lower = body.to_ascii_lowercase();
    let start = lower.find("<title>")? + 7;
    let end = lower[start..].find("</title>")? + start;
    Some(strip_tags(&body[start..end]).chars().take(200).collect())
}

fn extract_excerpt(body: &str) -> String {
    let text = strip_tags(body);
    text.split_whitespace()
        .collect::<Vec<_>>()
        .join(" ")
        .chars()
        .take(1200)
        .collect()
}

fn strip_tags(value: &str) -> String {
    let mut out = String::with_capacity(value.len());
    let mut inside = false;
    for ch in value.chars() {
        match ch {
            '<' => inside = true,
            '>' => inside = false,
            _ if !inside => out.push(ch),
            _ => {}
        }
    }
    out
}

fn percent_encode(value: &str) -> String {
    value
        .bytes()
        .map(|b| {
            if b.is_ascii_alphanumeric() || b"-_.".contains(&b) {
                (b as char).to_string()
            } else {
                format!("%{:02X}", b)
            }
        })
        .collect()
}
fn hash(value: &str) -> String {
    let mut h = DefaultHasher::new();
    value.hash(&mut h);
    format!("{:016x}", h.finish())
}
fn unix_now() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or_default()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn source_scores_prioritize_first_party_sources() {
        assert!(authority_score("official_docs") > authority_score("web"));
        assert_eq!(
            classify_source("https://github.com/rust-lang/rust"),
            "github"
        );
        assert!(normalize_url("http://localhost:3000").is_none());
    }

    #[test]
    fn html_extraction_is_bounded() {
        assert_eq!(
            extract_title("<html><title>Docs</title></html>"),
            Some("Docs".into())
        );
        assert!(extract_excerpt("<script>bad</script><p>Hello world</p>").contains("Hello"));
    }

    #[test]
    fn worker_pool_status_has_bounded_capacity() {
        let status = research_worker_pool_status();
        assert!((1..=16).contains(&status.max_workers));
        assert!(status.available_workers <= status.max_workers);
        assert!(status.running <= status.max_workers);
    }
}
