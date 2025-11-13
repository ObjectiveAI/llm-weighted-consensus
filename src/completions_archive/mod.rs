mod fetcher;

pub use fetcher::*;

pub enum Completion {
    Chat(crate::chat::completions::response::unary::ChatCompletion),
    Score(crate::score::completions::response::unary::ChatCompletion),
}
