use dashmap::DashMap;
use futures::Stream;
use std::sync::atomic::AtomicU64;

pub struct ChoiceIndexer {
    counter: AtomicU64,
    indices: DashMap<(usize, u64), u64>,
}

impl ChoiceIndexer {
    pub fn new(initial: u64) -> Self {
        Self {
            counter: AtomicU64::new(initial),
            indices: DashMap::new(),
        }
    }

    pub fn get(
        &self,
        meta_model_index: usize,
        native_choice_index: u64,
    ) -> u64 {
        *self
            .indices
            .entry((meta_model_index, native_choice_index))
            .or_insert_with(|| {
                self.counter
                    .fetch_add(1, std::sync::atomic::Ordering::SeqCst)
            })
    }
}

pub struct StreamOnce<T>(Option<T>);

impl<T> StreamOnce<T> {
    pub fn new(item: T) -> Self {
        Self(Some(item))
    }
}

impl<T> Stream for StreamOnce<T>
where
    T: Unpin,
{
    type Item = T;

    fn poll_next(
        mut self: std::pin::Pin<&mut Self>,
        _cx: &mut std::task::Context<'_>,
    ) -> std::task::Poll<Option<Self::Item>> {
        std::task::Poll::Ready(self.as_mut().get_mut().0.take())
    }
}
