use crate::search::{DeterministicRng, TopK};
use crate::{Dataset, Metric, Neighbor, Result, VectorError, VectorIndex};

#[derive(Debug, Clone, Copy)]
pub struct IvfFlatConfig {
    pub partitions: usize,
    pub probes: usize,
    pub iterations: usize,
    pub seed: u64,
}

impl Default for IvfFlatConfig {
    fn default() -> Self {
        Self {
            partitions: 16,
            probes: 4,
            iterations: 12,
            seed: 0x5eed,
        }
    }
}

#[derive(Debug, Clone)]
pub struct IvfFlatIndex {
    dataset: Dataset,
    metric: Metric,
    config: IvfFlatConfig,
    centroids: Vec<Vec<f32>>,
    lists: Vec<Vec<usize>>,
}

impl IvfFlatIndex {
    // 寻找每个数据点最近的质心
    fn nearest_centroid(metric: Metric, query: &[f32], centroids: &[Vec<f32>]) -> usize {
        centroids.iter().enumerate().min_by(|(_, a), (_, b)| {
        metric.distance(query, a).total_cmp(&metric.distance(query, b))
        }).map(|(idx, _)| idx).expect("centroids is non-empty")
    }

    // 当空簇时，将该质心更新为数据集中距离所有质心最远的那个数据点
    fn farthest_row(metric: Metric, dataset: &Dataset, centroids: &[Vec<f32>]) -> usize {
        dataset
        .vectors()
        .iter()
        .enumerate()
        .map(|(row, vector)| {
            let distance = centroids
                .iter()
                .map(|centroid| metric.distance(vector, centroid))
                .min_by(f32::total_cmp)         
                .unwrap_or(f32::INFINITY);
            Neighbor { row, distance }
        })
        .max()                                   
        .expect("dataset is non-empty")
        .row                                    
    }

    pub fn try_new(dataset: Dataset, metric: Metric, config: IvfFlatConfig) -> Result<Self> {
        // 1. 验证输入
        if config.probes < 1 {
            return Err(VectorError::InvalidConfig("probes must be at least 1"));
        } // 查询时，至少选择1个最近的质心，扫描它的倒排链表
        if config.partitions < config.probes {
            return Err(VectorError::InvalidConfig(
                "partitions must be greater than or equal to probes",
            ));
        } // 质心数量应该大于等于查询时选择的质心数量

        dataset.validate_for_metric(metric)?;
        let row = dataset.len();
        if row < config.partitions {
            return Err(VectorError::InvalidConfig(
                "dataset rows must be greater than or equal to partitions",
            ));
        } // 数据行应该大于等于分区数
        if config.iterations < 1 {
            return Err(VectorError::InvalidConfig("iterations must be at least 1"));
        }
        // 2. 初始化
        let mut rng = DeterministicRng::new(config.seed);
        let mut rows = (0..row).collect::<Vec<_>>();
        {
            // 倒序 fisher-yates: shuffle rows
            for end in (1..rows.len()).rev() {
                let selected = rng.index(end + 1); // 在 [0, end] 里随机挑一个下标
                rows.swap(end, selected);
            }
        }
        let mut centroids = 
        rows.iter().take(config.partitions).map(|&idx| dataset.vector(idx).to_vec()).
        collect::<Vec<_>>();
        
        let mut assignments = vec![usize::MAX; dataset.len()];
        // loop
        for _ in 0..config.iterations {
            // 每行找最近质心
            let mut next_assignments = assignments.clone();
            for row in rows.iter() {
                let vector = dataset.vector(*row);
                let best = Self::nearest_centroid(metric, vector, &centroids);
                next_assignments[*row] = best;
            }
            if assignments == next_assignments {
                break;
            } else {
                assignments = next_assignments;
                // update centroids
                let mut sums = vec![vec![0.0_f64; dataset.dimension()]; config.partitions];
                let mut counts = vec![0_usize; config.partitions];
                for (idx, row) in assignments.iter().enumerate() {
                    for (sum, value) in sums[*row].iter_mut().zip(dataset.vector(idx)) {
                        *sum += f64::from(*value);
                    }
                    counts[*row] += 1;
                }
                // compute new centroids
                for idx in 0..config.partitions {
                    // 1. 空簇
                    if counts[idx] == 0 {
                        let row = Self::farthest_row(metric, &dataset, &centroids);
                        centroids[idx].clone_from_slice(dataset.vector(row));
                        continue;
                    }
                    // 2. 正常均值
                    centroids[idx].iter_mut().zip(sums[idx].iter()).for_each(|(a, b)| {
                        *a = (*b / counts[idx] as f64) as f32;
                    });
                    // 3. 计算出的质心全0，没法算余弦
                    if centroids[idx].iter().all(|&a| a == 0.0) { // f32 不是不可以比较吗？
                        if metric == Metric::Cosine {
                            let row = assignments.iter().position(|&a| a == idx).unwrap(); // 任意一个属于该簇的成员向量
                            centroids[idx].clone_from_slice(dataset.vector(row));
                        }
                    }
                }
            }
        }
        Ok(Self {
            dataset: dataset.clone(),
            metric,
            config,
            centroids: centroids.clone(),
            lists: {
                let mut lists = vec![Vec::new(); config.partitions];
                for (row, vector) in dataset.vectors().iter().enumerate() {
                    lists[Self::nearest_centroid(metric, vector, &centroids)].push(row);
                }
                lists
            }
        })
    }

    pub fn centroids(&self) -> &[Vec<f32>] {
        &self.centroids
    }

    pub fn list_sizes(&self) -> Vec<usize> {
        self.lists.iter().map(Vec::len).collect()
    }

    pub fn search_with_probes(
        &self,
        query: &[f32],
        k: usize,
        probes: usize,
    ) -> Result<Vec<Neighbor>> {
        // 验证查询是否符合要求
        if probes < 1 {
            return Err(VectorError::InvalidConfig("probes must be at least 1"));
        } // 查询时，至少选择1个最近的质心，扫描它的倒排链表
        if self.config.partitions < probes {
            return Err(VectorError::InvalidConfig(
                "partitions must be greater than or equal to probes",
            ));
        } // 质心数量应该大于等于查询时选择的质心数量
        self.dataset.validate_query(query, self.metric)?;
        let mut ranked = self.centroids.iter().enumerate().map(|(idx, vector)| {
            Neighbor {
                row: idx,
                distance: self.metric.distance(vector, query),
            }
        }).collect::<Vec<_>>();
        ranked.sort_unstable();
        // probe 前 probes 个质心
        let probes = ranked.iter().take(probes).cloned().collect::<Vec<_>>();
        // 构造  top-k
        let mut top = TopK::new(k);
        for neighbor in probes.iter() {
            let list = &self.lists[neighbor.row];
            for row in list.iter() {
                top.push(Neighbor {
                    row: *row,
                    distance: self.metric.distance(self.dataset.vector(*row), query),
                });
            }
        }
        Ok(top.into_sorted())
    }
}

impl VectorIndex for IvfFlatIndex {
    fn kind(&self) -> &'static str {
        "ivf_flat"
    }

    fn dataset(&self) -> &Dataset {
        &self.dataset
    }

    fn metric(&self) -> Metric {
        self.metric
    }

    fn search(&self, query: &[f32], k: usize) -> Result<Vec<Neighbor>> {
        self.search_with_probes(query, k, self.config.probes)
    }
}
