//! LLM API 可用性调度 — 时段窗口 + 熔断器

use std::sync::Mutex;
use std::time::Instant;

/// API 开放时段(本地时间)。start>end 表示跨日窗口(如 23:00-09:00)。
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize, PartialEq)]
pub struct TimeWindow {
    /// 开始时间 "HH:MM"(本地)
    pub start: String,
    /// 结束时间 "HH:MM"(本地)
    pub end: String,
}

impl TimeWindow {
    fn parse_hm(s: &str) -> Option<(u32, u32)> {
        let mut parts = s.split(':');
        let h: u32 = parts.next()?.parse().ok()?;
        let m: u32 = parts.next()?.parse().ok()?;
        if h < 24 && m < 60 {
            Some((h, m))
        } else {
            None
        }
    }

    /// 给定 hour/minute(本地)是否落在窗口内。
    /// 边界语义:[start, end),即含 start 不含 end。
    pub fn contains_at(&self, hour: u32, minute: u32) -> bool {
        let start = Self::parse_hm(&self.start);
        let end = Self::parse_hm(&self.end);
        match (start, end) {
            (Some(s), Some(e)) => {
                let now_min = hour * 60 + minute;
                let s_min = s.0 * 60 + s.1;
                let e_min = e.0 * 60 + e.1;
                if s_min <= e_min {
                    now_min >= s_min && now_min < e_min
                } else {
                    now_min >= s_min || now_min < e_min
                }
            }
            _ => false,
        }
    }

    /// 用本地当前时间判断是否在窗口内。
    pub fn contains_now(&self) -> bool {
        let now = chrono::Local::now();
        let h: u32 = now.format("%H").to_string().parse().unwrap_or(0);
        let m: u32 = now.format("%M").to_string().parse().unwrap_or(0);
        self.contains_at(h, m)
    }
}

#[derive(Clone, Debug, PartialEq, serde::Serialize)]
pub enum BreakerState {
    Closed,
    Open,
    HalfOpen,
}

#[derive(Clone, Debug, serde::Serialize)]
pub struct BreakerSnapshot {
    pub state: BreakerState,
    pub consecutive_failures: u32,
    pub opened_at: Option<String>,
}

pub struct LlmCircuitBreaker {
    inner: Mutex<BreakerInner>,
}

#[derive(Clone)]
struct BreakerInner {
    state: BreakerState,
    consecutive_failures: u32,
    opened_at: Option<Instant>,
}

const FAILURE_THRESHOLD: u32 = 3;
const PROBE_INTERVAL_SECS: u64 = 60;

impl LlmCircuitBreaker {
    pub fn new() -> Self {
        Self {
            inner: Mutex::new(BreakerInner {
                state: BreakerState::Closed,
                consecutive_failures: 0,
                opened_at: None,
            }),
        }
    }

    pub fn is_open(&self) -> bool {
        let g = self.inner.lock().unwrap();
        // HalfOpen 让流量通过（允许真实调用验证恢复），仅 Open 状态短路。
        matches!(g.state, BreakerState::Open)
    }

    pub fn should_probe(&self) -> bool {
        let g = self.inner.lock().unwrap();
        match g.state {
            BreakerState::Open => {
                let elapsed = g.opened_at.map(|t| t.elapsed().as_secs()).unwrap_or(0);
                elapsed >= PROBE_INTERVAL_SECS
            }
            BreakerState::HalfOpen => true,
            BreakerState::Closed => false,
        }
    }

    pub fn record_success(&self) {
        let mut g = self.inner.lock().unwrap();
        g.consecutive_failures = 0;
        g.state = BreakerState::Closed;
        g.opened_at = None;
    }

    pub fn record_failure(&self) {
        let mut g = self.inner.lock().unwrap();
        g.consecutive_failures += 1;
        if g.consecutive_failures >= FAILURE_THRESHOLD {
            g.state = BreakerState::Open;
            g.opened_at = Some(Instant::now());
        }
    }

    pub fn probe_succeeded(&self) {
        let mut g = self.inner.lock().unwrap();
        if matches!(g.state, BreakerState::Open) {
            g.state = BreakerState::HalfOpen;
        }
    }

    pub fn snapshot(&self) -> BreakerSnapshot {
        let g = self.inner.lock().unwrap();
        BreakerSnapshot {
            state: g.state.clone(),
            consecutive_failures: g.consecutive_failures,
            opened_at: g.opened_at.map(|_| chrono::Utc::now().to_rfc3339()),
        }
    }
}

impl Default for LlmCircuitBreaker {
    fn default() -> Self {
        Self::new()
    }
}
