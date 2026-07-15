//! User-calibrated value energy for proactive opportunity allocation.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ValueFeedbackStats {
    pub useful: u64,
    pub not_useful: u64,
}

impl ValueFeedbackStats {
    fn weight(&self) -> f64 {
        let ratio = self.posterior_mean();
        let confidence = self.confidence();
        (1.0 + (ratio - 0.5) * 0.70 * confidence).clamp(0.65, 1.35)
    }

    pub fn sample_count(&self) -> u64 {
        self.useful.saturating_add(self.not_useful)
    }

    pub fn posterior_mean(&self) -> f64 {
        (self.useful as f64 + 2.0) / (self.sample_count() as f64 + 4.0)
    }

    pub fn confidence(&self) -> f64 {
        let samples = self.sample_count() as f64;
        samples / (samples + 8.0)
    }

    fn record(&mut self, useful: bool) {
        if useful {
            self.useful = self.useful.saturating_add(1);
        } else {
            self.not_useful = self.not_useful.saturating_add(1);
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UserValueProfile {
    #[serde(default)]
    pub global: ValueFeedbackStats,
    #[serde(default)]
    pub categories: HashMap<String, ValueFeedbackStats>,
    #[serde(default)]
    pub projects: HashMap<String, ProjectValueProfile>,
    #[serde(default)]
    pub updated_at: Option<u64>,
}

impl Default for UserValueProfile {
    fn default() -> Self {
        Self {
            global: ValueFeedbackStats::default(),
            categories: HashMap::new(),
            projects: HashMap::new(),
            updated_at: None,
        }
    }
}

impl UserValueProfile {
    pub fn preference_weight(&self, category: &str) -> f64 {
        let global = self.global.weight();
        let category = normalize_category(category);
        let category = self
            .categories
            .get(&category)
            .map(ValueFeedbackStats::weight)
            .unwrap_or(1.0);
        (category * 0.8 + global * 0.2).clamp(0.65, 1.35)
    }

    pub fn preference_weight_for(&self, project_path: &str, category: &str) -> f64 {
        let category = normalize_category(category);
        let fallback = self.preference_weight(&category);
        let Some(project) = self.projects.get(project_path) else {
            return fallback;
        };
        let project_global = project.global.weight();
        let project_category = project
            .categories
            .get(&category)
            .map(ValueFeedbackStats::weight)
            .unwrap_or(1.0);
        (project_category * 0.60 + project_global * 0.20 + fallback * 0.20).clamp(0.65, 1.35)
    }
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ProjectValueProfile {
    #[serde(default)]
    pub global: ValueFeedbackStats,
    #[serde(default)]
    pub categories: HashMap<String, ValueFeedbackStats>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ValueFeedbackEffect {
    pub category: String,
    #[serde(default)]
    pub project_path: String,
    pub useful: bool,
    pub before_weight: f64,
    pub after_weight: f64,
    pub posterior_mean: f64,
    pub confidence: f64,
    pub sample_count: u64,
}

#[derive(Debug, Clone)]
pub struct ValueFeedbackUpdate {
    pub profile: UserValueProfile,
    pub effect: ValueFeedbackEffect,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ValueEnergyAllocation {
    pub category: String,
    pub base_value: f64,
    pub confidence: f64,
    #[serde(default)]
    pub leverage_score: f64,
    pub preference_weight: f64,
    pub personalized_value: f64,
    pub risk_penalty: f64,
    pub attention_penalty: f64,
    pub resource_penalty: f64,
    pub net_value: f64,
    pub energy_units: u32,
    pub rationale: String,
}

pub struct ValueOpportunityInput<'a> {
    pub category: &'a str,
    pub value: f64,
    pub confidence: f64,
    pub leverage_score: f64,
    pub risk: f64,
    pub attention_cost: f64,
    pub resource_cost: f64,
}

pub fn allocate(
    profile: &UserValueProfile,
    input: ValueOpportunityInput<'_>,
) -> ValueEnergyAllocation {
    let category = normalize_category(input.category);
    let value = input.value.clamp(0.0, 1.0);
    let confidence = input.confidence.clamp(0.0, 1.0);
    let leverage_score = input.leverage_score.clamp(0.0, 1.0);
    let preference_weight = profile.preference_weight(&category);
    let base_value = value * 0.48 + confidence * 0.32 + leverage_score * 0.20;
    let personalized_value = (base_value * preference_weight).clamp(0.0, 1.0);
    let risk_penalty = input.risk.clamp(0.0, 1.0) * 0.32;
    let attention_penalty = input.attention_cost.clamp(0.0, 1.0) * 0.18;
    let resource_penalty = input.resource_cost.clamp(0.0, 1.0) * 0.10;
    let net_value =
        (personalized_value - risk_penalty - attention_penalty - resource_penalty).clamp(0.0, 1.0);
    let energy_units = energy_units(net_value);
    let rationale = format!(
        "个性化价值 {:.2}（含核心杠杆 {:.2}）- 风险 {:.2} - 打扰 {:.2} - 资源 {:.2} = 净价值 {:.2}",
        personalized_value,
        leverage_score,
        risk_penalty,
        attention_penalty,
        resource_penalty,
        net_value
    );
    ValueEnergyAllocation {
        category,
        base_value,
        confidence,
        leverage_score,
        preference_weight,
        personalized_value,
        risk_penalty,
        attention_penalty,
        resource_penalty,
        net_value,
        energy_units,
        rationale,
    }
}

pub fn allocate_for_project(
    profile: &UserValueProfile,
    project_path: &str,
    input: ValueOpportunityInput<'_>,
) -> ValueEnergyAllocation {
    let mut allocation = allocate(profile, input);
    let preference_weight = profile.preference_weight_for(project_path, &allocation.category);
    allocation.preference_weight = preference_weight;
    allocation.personalized_value = (allocation.base_value * preference_weight).clamp(0.0, 1.0);
    allocation.net_value = (allocation.personalized_value
        - allocation.risk_penalty
        - allocation.attention_penalty
        - allocation.resource_penalty)
        .clamp(0.0, 1.0);
    allocation.energy_units = energy_units(allocation.net_value);
    allocation.rationale = format!(
        "项目偏好 {:.2} × 基础价值 {:.2} - 风险 {:.2} - 打扰 {:.2} - 资源 {:.2} = 净价值 {:.2}",
        preference_weight,
        allocation.base_value,
        allocation.risk_penalty,
        allocation.attention_penalty,
        allocation.resource_penalty,
        allocation.net_value
    );
    allocation
}

fn energy_units(net_value: f64) -> u32 {
    match net_value {
        score if score >= 0.75 => 5,
        score if score >= 0.55 => 4,
        score if score >= 0.40 => 3,
        score if score >= 0.25 => 2,
        score if score >= 0.12 => 1,
        _ => 0,
    }
}

pub fn load_profile(storage_dir: &Path) -> UserValueProfile {
    std::fs::read_to_string(profile_path(storage_dir))
        .ok()
        .and_then(|raw| serde_json::from_str(&raw).ok())
        .unwrap_or_default()
}

pub fn record_feedback(
    storage_dir: &Path,
    category: &str,
    useful: bool,
) -> Result<UserValueProfile, String> {
    Ok(record_feedback_for_project(storage_dir, "", category, useful)?.profile)
}

pub fn record_feedback_for_project(
    storage_dir: &Path,
    project_path: &str,
    category: &str,
    useful: bool,
) -> Result<ValueFeedbackUpdate, String> {
    let mut profile = load_profile(storage_dir);
    let category = normalize_category(category);
    let before_weight = if project_path.is_empty() {
        profile.preference_weight(&category)
    } else {
        profile.preference_weight_for(project_path, &category)
    };
    profile.global.record(useful);
    profile
        .categories
        .entry(category.clone())
        .or_default()
        .record(useful);
    if !project_path.is_empty() {
        let project = profile
            .projects
            .entry(project_path.to_string())
            .or_default();
        project.global.record(useful);
        project
            .categories
            .entry(category.clone())
            .or_default()
            .record(useful);
    }
    profile.updated_at = Some(unix_now());
    persist_profile(storage_dir, &profile)?;
    let stats = if project_path.is_empty() {
        profile.categories.get(&category)
    } else {
        profile
            .projects
            .get(project_path)
            .and_then(|project| project.categories.get(&category))
    }
    .cloned()
    .unwrap_or_default();
    let after_weight = if project_path.is_empty() {
        profile.preference_weight(&category)
    } else {
        profile.preference_weight_for(project_path, &category)
    };
    Ok(ValueFeedbackUpdate {
        profile,
        effect: ValueFeedbackEffect {
            category,
            project_path: project_path.to_string(),
            useful,
            before_weight,
            after_weight,
            posterior_mean: stats.posterior_mean(),
            confidence: stats.confidence(),
            sample_count: stats.sample_count(),
        },
    })
}

pub fn infer_category(text: &str) -> String {
    let lower = text.to_lowercase();
    for (category, markers) in [
        (
            "security",
            &["security", "安全", "漏洞", "密钥", "密码"][..],
        ),
        ("growth", &["growth", "增长", "转化", "收入", "seo"][..]),
        ("content", &["content", "内容", "文章", "运营"][..]),
        ("feature", &["feature", "功能", "产品", "体验"][..]),
        ("bug", &["bug", "修复", "错误", "崩溃", "回归"][..]),
        ("test", &["test", "测试", "ci", "验证"][..]),
        ("docs", &["docs", "文档", "readme", "教程"][..]),
        ("dependency", &["dependency", "依赖", "升级"][..]),
    ] {
        if markers.iter().any(|marker| lower.contains(marker)) {
            return category.into();
        }
    }
    "maintenance".into()
}

pub fn normalize_category(category: &str) -> String {
    match category.trim().to_lowercase().as_str() {
        "修复" | "错误" => "bug".into(),
        "功能" | "产品" => "feature".into(),
        "内容" | "运营" => "content".into(),
        "增长" => "growth".into(),
        "测试" | "质量" | "工程化" => "test".into(),
        "文档" => "docs".into(),
        "安全" => "security".into(),
        "依赖" => "dependency".into(),
        "维护" | "" => "maintenance".into(),
        value => value.to_string(),
    }
}

fn persist_profile(storage_dir: &Path, profile: &UserValueProfile) -> Result<(), String> {
    std::fs::create_dir_all(storage_dir).map_err(|error| error.to_string())?;
    let path = profile_path(storage_dir);
    let temporary = path.with_extension("json.tmp");
    std::fs::write(
        &temporary,
        serde_json::to_vec_pretty(profile).map_err(|error| error.to_string())?,
    )
    .map_err(|error| error.to_string())?;
    std::fs::rename(temporary, path).map_err(|error| error.to_string())
}

fn profile_path(storage_dir: &Path) -> PathBuf {
    storage_dir.join("value_profile.json")
}

fn unix_now() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|duration| duration.as_secs())
        .unwrap_or_default()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn higher_value_and_lower_cost_receive_more_energy() {
        let profile = UserValueProfile::default();
        let high = allocate(
            &profile,
            ValueOpportunityInput {
                category: "feature",
                value: 0.92,
                confidence: 0.90,
                leverage_score: 0.90,
                risk: 0.15,
                attention_cost: 0.15,
                resource_cost: 0.30,
            },
        );
        let low = allocate(
            &profile,
            ValueOpportunityInput {
                category: "maintenance",
                value: 0.45,
                confidence: 0.50,
                leverage_score: 0.20,
                risk: 0.55,
                attention_cost: 0.70,
                resource_cost: 0.60,
            },
        );
        assert!(high.net_value > low.net_value);
        assert!(high.energy_units > low.energy_units);
    }

    #[test]
    fn feedback_calibrates_category_preference() {
        let mut profile = UserValueProfile::default();
        for _ in 0..4 {
            profile.global.record(true);
            profile
                .categories
                .entry("growth".into())
                .or_default()
                .record(true);
            profile
                .categories
                .entry("docs".into())
                .or_default()
                .record(false);
        }
        assert!(profile.preference_weight("growth") > profile.preference_weight("docs"));
    }

    #[test]
    fn a_single_feedback_has_low_confidence() {
        let mut stats = ValueFeedbackStats::default();
        stats.record(false);
        assert_eq!(stats.sample_count(), 1);
        assert!(stats.confidence() < 0.2);
        assert!(stats.weight() > 0.9);
    }

    #[test]
    fn project_preference_overrides_category_without_leaking() {
        let mut profile = UserValueProfile::default();
        let project = profile.projects.entry("/project/a".into()).or_default();
        for _ in 0..12 {
            project.global.record(true);
            project
                .categories
                .entry("docs".into())
                .or_default()
                .record(true);
        }
        assert!(profile.preference_weight_for("/project/a", "docs") > 1.0);
        assert_eq!(profile.preference_weight_for("/project/b", "docs"), 1.0);
    }

    #[test]
    fn core_leverage_breaks_otherwise_equal_ties() {
        let profile = UserValueProfile::default();
        let allocate_with = |leverage_score| {
            allocate(
                &profile,
                ValueOpportunityInput {
                    category: "feature",
                    value: 0.7,
                    confidence: 0.7,
                    leverage_score,
                    risk: 0.2,
                    attention_cost: 0.2,
                    resource_cost: 0.3,
                },
            )
        };
        assert!(allocate_with(0.9).net_value > allocate_with(0.2).net_value);
    }
}
