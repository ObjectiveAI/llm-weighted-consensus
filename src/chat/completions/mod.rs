#[cfg(not(target_arch = "wasm32"))]
mod client;

mod error;
pub mod request;
pub mod response;

#[cfg(not(target_arch = "wasm32"))]
pub use client::*;

pub use error::*;
