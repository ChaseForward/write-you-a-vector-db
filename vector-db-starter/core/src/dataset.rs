use std::sync::Arc;

use crate::{Metric, Result, VectorError};

#[derive(Debug, Clone)]
pub struct Dataset {
    vectors: Arc<[Vec<f32>]>,
    dimension: usize,
}

impl Dataset {
    pub fn try_new(vectors: Vec<Vec<f32>>) -> Result<Self> {
        if vectors.is_empty() {
            return Err(VectorError::EmptyDataset);
        }
        let dimension = vectors[0].len();
        if dimension == 0 {
            return Err(VectorError::EmptyVector);
        }
        let len = vectors.len();
        for i in 0..len {
            let vector = &vectors[i];
            if vector.len() != dimension {
                return Err(VectorError::DimensionMismatch {
                    expected: dimension,
                    actual: vector.len(),
                });
            }
            for j in 0..vector.len() {
                if !f32::is_finite(vector[j]) {
                    return Err(VectorError::NonFiniteValue {
                        vector: i,
                        dimension: j,
                    });
                }
            }
        }
        Ok(Self {
            vectors: Arc::from(vectors),
            dimension,
        })
    }

    pub fn len(&self) -> usize {
        self.vectors.len()
    }

    pub fn is_empty(&self) -> bool {
        self.vectors.is_empty()
    }

    pub fn dimension(&self) -> usize {
        self.dimension
    }

    pub fn vector(&self, row: usize) -> &[f32] {
        &self.vectors[row]
    }

    pub fn vectors(&self) -> &[Vec<f32>] {
        &self.vectors
    }

    pub(crate) fn validate_for_metric(&self, metric: Metric) -> Result<()> {
        match metric {
            Metric::Cosine => {
                for (row_idx, vector) in self.vectors.iter().enumerate() {
                    if vector.iter().all(|&x| x == 0.0) {
                        return Err(VectorError::ZeroNorm { vector: row_idx });
                    }
                }
            }
            Metric::Dot => {}
            Metric::Euclidean => {}
        };
        Ok(())
    }

    pub(crate) fn validate_query(&self, query: &[f32], metric: Metric) -> Result<()> {
        let len = query.len();
        if len != self.dimension {
            return Err(VectorError::DimensionMismatch {
                expected: self.dimension,
                actual: len,
            });
        }
        for i in 0..len {
            if !f32::is_finite(query[i]) {
                return Err(VectorError::NonFiniteValue {
                    vector: len,
                    dimension: i,
                });
            }
        }
        match metric {
            Metric::Cosine => {
                if query.iter().all(|&x| x == 0.0) {
                    return Err(VectorError::ZeroNorm { vector: len });
                }
            }
            Metric::Dot => {}
            Metric::Euclidean => {}
        };
        Ok(())
    }
}
