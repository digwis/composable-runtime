//! Initiative policy for deciding when the agent observes, experiments,
//! asks the user, or is allowed to execute automatically.

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum InitiativeAction {
    Observe,
    AskUser,
    Experiment,
    AutoExecute,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InitiativeDecision {
    pub confidence: f64,
    pub value: f64,
    pub risk: f64,
    pub attention_cost: f64,
    pub recommended_action: InitiativeAction,
    pub effective_action: InitiativeAction,
    pub auto_execute_enabled: bool,
    pub rationale: String,
}

impl Default for InitiativeDecision {
    fn default() -> Self {
        decide(0.0, 0.0, 1.0, 1.0)
    }
}

/// A conservative controller. Automatic execution is opt-in through
/// `ORCH_INITIATIVE_AUTO_EXECUTE=1`; the confidence policy remains visible
/// even when the effective action is downgraded to user approval.
pub fn decide(confidence: f64, value: f64, risk: f64, attention_cost: f64) -> InitiativeDecision {
    let confidence = confidence.clamp(0.0, 1.0);
    let value = value.clamp(0.0, 1.0);
    let risk = risk.clamp(0.0, 1.0);
    let attention_cost = attention_cost.clamp(0.0, 1.0);
    let recommended_action =
        if confidence >= 0.95 && value >= 0.75 && risk <= 0.25 && attention_cost <= 0.35 {
            InitiativeAction::AutoExecute
        } else if confidence >= 0.70 && value >= 0.45 && risk <= 0.65 && attention_cost <= 0.75 {
            InitiativeAction::Experiment
        } else if confidence >= 0.40 && value >= 0.20 && attention_cost <= 0.90 {
            InitiativeAction::AskUser
        } else {
            InitiativeAction::Observe
        };
    let auto_execute_enabled = std::env::var("ORCH_INITIATIVE_AUTO_EXECUTE")
        .map(|value| value == "1" || value.eq_ignore_ascii_case("true"))
        .unwrap_or(false);
    let effective_action = match recommended_action {
        InitiativeAction::AutoExecute if auto_execute_enabled => InitiativeAction::AutoExecute,
        InitiativeAction::AutoExecute => InitiativeAction::AskUser,
        action => action,
    };
    let rationale = match recommended_action {
        InitiativeAction::AutoExecute => {
            "高置信度、高价值且低风险；自动执行仍受显式开关保护".into()
        }
        InitiativeAction::Experiment => {
            "净价值足以投入隔离实验，但风险或置信度尚不足以直接修改项目".into()
        }
        InitiativeAction::AskUser => "机会可能有价值，但实验成本或风险需要用户决定方向".into(),
        InitiativeAction::Observe => "净价值不足或打扰成本过高，继续收集信息".into(),
    };
    InitiativeDecision {
        confidence,
        value,
        risk,
        attention_cost,
        recommended_action,
        effective_action,
        auto_execute_enabled,
        rationale,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn confidence_bands_follow_policy() {
        assert_eq!(
            decide(0.2, 0.9, 0.1, 0.1).recommended_action,
            InitiativeAction::Observe
        );
        assert_eq!(
            decide(0.5, 0.9, 0.1, 0.1).recommended_action,
            InitiativeAction::AskUser
        );
        assert_eq!(
            decide(0.8, 0.9, 0.1, 0.1).recommended_action,
            InitiativeAction::Experiment
        );
        assert_eq!(
            decide(0.98, 0.9, 0.1, 0.1).recommended_action,
            InitiativeAction::AutoExecute
        );
    }

    #[test]
    fn automatic_execution_is_guarded_by_opt_in() {
        std::env::remove_var("ORCH_INITIATIVE_AUTO_EXECUTE");
        assert_eq!(
            decide(0.99, 0.9, 0.1, 0.1).effective_action,
            InitiativeAction::AskUser
        );
    }

    #[test]
    fn attention_and_risk_constrain_initiative() {
        assert_eq!(
            decide(0.99, 0.9, 0.1, 0.9).recommended_action,
            InitiativeAction::AskUser
        );
        assert_eq!(
            decide(0.8, 0.9, 0.9, 0.2).recommended_action,
            InitiativeAction::AskUser
        );
    }
}
