pub mod code;
pub mod compute;
pub mod fs;
pub mod greet;
pub mod http;
pub mod shell;
pub mod store;
pub mod web;

pub use code::CodeCapability;
pub use compute::ComputeCapability;
pub use fs::FsCapability;
pub use greet::GreetCapability;
pub use http::HttpCapability;
pub use shell::ShellCapability;
pub use store::StoreCapability;
pub use web::WebCapability;
