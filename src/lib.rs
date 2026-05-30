use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use std::collections::HashMap;
use std::collections::HashSet;

// ─── helpers ─────────────────────────────────────────────────────────────────

#[inline]
fn dot_product(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
}

#[inline]
fn magnitude(v: &[f64]) -> f64 {
    v.iter().map(|x| x * x).sum::<f64>().sqrt()
}

#[inline]
fn cosine_sim(a: &[f64], b: &[f64]) -> f64 {
    let mag_a = magnitude(a);
    let mag_b = magnitude(b);
    if mag_a == 0.0 || mag_b == 0.0 {
        return 0.0;
    }
    dot_product(a, b) / (mag_a * mag_b)
}

#[inline]
fn weight(impact_score: f64, usage_count: u64, success_count: u64, failure_count: u64) -> f64 {
    let total = (success_count + failure_count) as f64;
    let success_ratio = if total > 0.0 {
        success_count as f64 / total
    } else {
        0.0
    };
    impact_score * (1.0 + success_ratio) * (1.0 + usage_count as f64).ln()
}

// ─── similarity operations ────────────────────────────────────────────────────

/// Compute cosine similarity between two equal-length vectors.
/// Returns 0.0 if either vector is all-zeros.
#[pyfunction]
fn cosine_similarity(a: Vec<f64>, b: Vec<f64>) -> PyResult<f64> {
    if a.len() != b.len() {
        return Err(PyValueError::new_err(format!(
            "vectors must have equal length: {} != {}",
            a.len(),
            b.len()
        )));
    }
    Ok(cosine_sim(&a, &b))
}

/// Compute cosine similarity between a query vector and each row in matrix.
/// Returns a list of similarity scores, one per row.
#[pyfunction]
fn batch_cosine_similarity(query: Vec<f64>, matrix: Vec<Vec<f64>>) -> PyResult<Vec<f64>> {
    for (i, row) in matrix.iter().enumerate() {
        if row.len() != query.len() {
            return Err(PyValueError::new_err(format!(
                "matrix row {i} length {} != query length {}",
                row.len(),
                query.len()
            )));
        }
    }
    Ok(matrix.iter().map(|row| cosine_sim(&query, row)).collect())
}

// ─── scoring operations ───────────────────────────────────────────────────────

/// Compute the ranking weight for a single memory node.
/// Formula: impact_score * (1 + success_ratio) * ln(1 + usage_count)
#[pyfunction]
fn compute_weight(
    impact_score: f64,
    usage_count: u64,
    success_count: u64,
    failure_count: u64,
) -> f64 {
    weight(impact_score, usage_count, success_count, failure_count)
}

/// Compute weights for a batch of nodes.
/// Each node is (impact_score, usage_count, success_count, failure_count).
/// Returns weights in the same order as input.
#[pyfunction]
fn batch_compute_weights(nodes: Vec<(f64, u64, u64, u64)>) -> Vec<f64> {
    nodes
        .iter()
        .map(|(imp, u, s, f)| weight(*imp, *u, *s, *f))
        .collect()
}

/// Return indices that sort nodes by weight descending.
/// Input: list of (impact_score, usage_count, success_count, failure_count).
/// Output: list of integer indices into the input, highest-weight first.
#[pyfunction]
fn argsort_by_weight(nodes: Vec<(f64, u64, u64, u64)>) -> Vec<usize> {
    let mut weights: Vec<(usize, f64)> = nodes
        .iter()
        .enumerate()
        .map(|(i, (imp, u, s, f))| (i, weight(*imp, *u, *s, *f)))
        .collect();
    weights.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    weights.iter().map(|(i, _)| *i).collect()
}

// ─── traversal operations ─────────────────────────────────────────────────────

/// Walk the causal chain starting from start_id, following id_to_parent links.
/// Returns nodes from oldest ancestor to start_id (inclusive).
/// Raises ValueError if a cycle is detected.
/// Raises ValueError if start_id is not in id_to_parent (key lookup).
#[pyfunction]
fn traverse_chain(
    id_to_parent: HashMap<String, Option<String>>,
    start_id: String,
    max_depth: usize,
) -> PyResult<Vec<String>> {
    let mut chain: Vec<String> = Vec::new();
    let mut seen: HashSet<String> = HashSet::new();
    let mut current = start_id.clone();

    chain.push(current.clone());
    seen.insert(current.clone());

    // max_depth is the maximum number of nodes to include (including start).
    for _ in 0..max_depth.saturating_sub(1) {
        match id_to_parent.get(&current) {
            None => break,
            Some(None) => break,
            Some(Some(parent_id)) => {
                if seen.contains(parent_id) {
                    return Err(PyValueError::new_err(format!(
                        "causal cycle detected at node: {parent_id}"
                    )));
                }
                seen.insert(parent_id.clone());
                chain.push(parent_id.clone());
                current = parent_id.clone();
            }
        }
    }

    chain.reverse();
    Ok(chain)
}

/// Check whether adding an edge from child_id to parent_id would create a cycle.
/// id_to_parent maps each node ID to its current parent ID (or None).
/// Returns true if a cycle would be created.
#[pyfunction]
fn would_create_cycle(
    id_to_parent: HashMap<String, Option<String>>,
    child_id: String,
    parent_id: String,
) -> bool {
    // Walk up from parent; if we reach child_id, linking child→parent creates a cycle.
    let mut seen: HashSet<String> = HashSet::new();
    let mut current = parent_id.clone();
    seen.insert(child_id.clone());

    for _ in 0..1000 {
        if current == child_id {
            return true;
        }
        if seen.contains(&current) {
            return false;
        }
        seen.insert(current.clone());
        match id_to_parent.get(&current) {
            None | Some(None) => return false,
            Some(Some(next)) => {
                current = next.clone();
            }
        }
    }
    false
}

// ─── combined pipeline ────────────────────────────────────────────────────────

/// Find the top-k nodes most similar to a query embedding.
///
/// embeddings: list of (node_id, embedding_vector) pairs.
/// threshold: minimum cosine similarity to include.
/// top_k: maximum number of results.
///
/// Returns node IDs ordered by similarity descending (best first).
#[pyfunction]
fn rank_by_similarity(
    query: Vec<f64>,
    embeddings: Vec<(String, Vec<f64>)>,
    threshold: f64,
    top_k: usize,
) -> PyResult<Vec<String>> {
    let mut scored: Vec<(f64, String)> = Vec::new();
    for (id, emb) in &embeddings {
        if emb.len() != query.len() {
            return Err(PyValueError::new_err(format!(
                "embedding for node {id} has length {} != query length {}",
                emb.len(),
                query.len()
            )));
        }
        let sim = cosine_sim(&query, emb);
        if sim >= threshold {
            scored.push((sim, id.clone()));
        }
    }
    scored.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
    Ok(scored
        .into_iter()
        .take(top_k)
        .map(|(_, id)| id)
        .collect())
}

/// Combined scoring pipeline: for each node, compute a blended score from
/// cosine similarity (if embedding provided) and weight, then return top-k IDs.
///
/// nodes: list of (id, impact_score, usage_count, success_count, failure_count, embedding_or_none)
/// query: the query embedding vector (empty Vec means weight-only ranking)
/// sim_weight: blending factor [0.0, 1.0] — 1.0 = similarity only, 0.0 = weight only
#[pyfunction]
fn rank_blended(
    nodes: Vec<(String, f64, u64, u64, u64, Option<Vec<f64>>)>,
    query: Vec<f64>,
    sim_weight: f64,
    top_k: usize,
) -> Vec<String> {
    let use_sim = !query.is_empty() && sim_weight > 0.0;
    let mut scored: Vec<(f64, String)> = nodes
        .iter()
        .map(|(id, imp, u, s, f, emb)| {
            let w = weight(*imp, *u, *s, *f);
            let sim = if use_sim {
                if let Some(e) = emb {
                    if e.len() == query.len() {
                        cosine_sim(&query, e)
                    } else {
                        0.0
                    }
                } else {
                    0.0
                }
            } else {
                0.0
            };
            let blended = sim_weight * sim + (1.0 - sim_weight) * w;
            (blended, id.clone())
        })
        .collect();
    scored.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
    scored.into_iter().take(top_k).map(|(_, id)| id).collect()
}

// ─── module ───────────────────────────────────────────────────────────────────

#[pymodule]
fn _core(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(cosine_similarity, m)?)?;
    m.add_function(wrap_pyfunction!(batch_cosine_similarity, m)?)?;
    m.add_function(wrap_pyfunction!(compute_weight, m)?)?;
    m.add_function(wrap_pyfunction!(batch_compute_weights, m)?)?;
    m.add_function(wrap_pyfunction!(argsort_by_weight, m)?)?;
    m.add_function(wrap_pyfunction!(traverse_chain, m)?)?;
    m.add_function(wrap_pyfunction!(would_create_cycle, m)?)?;
    m.add_function(wrap_pyfunction!(rank_by_similarity, m)?)?;
    m.add_function(wrap_pyfunction!(rank_blended, m)?)?;
    m.add("__version__", "0.1.0")?;
    Ok(())
}
