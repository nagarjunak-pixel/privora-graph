// Native C++20 (-O3) CSR Personalized PageRank (with Dangling-Node Mass Conservation)
// & Batch Cosine Similarity Kernel for PrivoraGraph.

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <vector>

extern "C" {

// Execute exact CSR Personalized PageRank power iteration:
//   p_{t+1} = (1 - teleport) * (P^T p_t + dangling_mass * seed) + teleport * seed
// Guarantees exact probability simplex conservation sum_i p_i == 1.0 at every step.
int32_t privora_csr_ppr_f64(
    int32_t num_nodes,
    const int32_t* __restrict__ row_ptr,
    const int32_t* __restrict__ col_idx,
    const double* __restrict__ edge_weights,
    const double* __restrict__ seed_dist,
    double teleport_alpha,
    double tol,
    int32_t max_iters,
    double* __restrict__ rank_out
) {
    std::vector<double> row_sums(static_cast<size_t>(num_nodes), 0.0);
    for (int32_t u = 0; u < num_nodes; ++u) {
        double s = 0.0;
        for (int32_t e = row_ptr[u]; e < row_ptr[u + 1]; ++e) {
            s += edge_weights[e];
        }
        row_sums[static_cast<size_t>(u)] = s;
        rank_out[u] = seed_dist[u];
    }

    std::vector<double> next_rank(static_cast<size_t>(num_nodes), 0.0);
    int32_t iters_used = 0;

    for (int32_t it = 0; it < max_iters; ++it) {
        iters_used = it + 1;
        double dangling_mass = 0.0;
        for (int32_t u = 0; u < num_nodes; ++u) {
            if (row_sums[static_cast<size_t>(u)] <= 1e-15) {
                dangling_mass += rank_out[u];
            }
        }

        const double base_seed_scale = teleport_alpha + (1.0 - teleport_alpha) * dangling_mass;
        for (int32_t v = 0; v < num_nodes; ++v) {
            next_rank[static_cast<size_t>(v)] = base_seed_scale * seed_dist[v];
        }

        const double walk_scale = 1.0 - teleport_alpha;
        for (int32_t u = 0; u < num_nodes; ++u) {
            const double rs = row_sums[static_cast<size_t>(u)];
            if (rs > 1e-15) {
                const double contrib = walk_scale * (rank_out[u] / rs);
                for (int32_t e = row_ptr[u]; e < row_ptr[u + 1]; ++e) {
                    const int32_t v = col_idx[e];
                    next_rank[static_cast<size_t>(v)] += contrib * edge_weights[e];
                }
            }
        }

        double l1_diff = 0.0;
        for (int32_t i = 0; i < num_nodes; ++i) {
            l1_diff += std::abs(next_rank[static_cast<size_t>(i)] - rank_out[i]);
            rank_out[i] = next_rank[static_cast<size_t>(i)];
        }
        if (l1_diff <= tol) {
            break;
        }
    }
    return iters_used;
}

// Compute exact L2-normalized cosine similarity scores between query [dim] and corpus [n_vecs, dim].
void privora_batch_cosine_f32(
    const float* __restrict__ query,
    const float* __restrict__ corpus,
    int32_t n_vecs,
    int32_t dim,
    float* __restrict__ scores_out
) {
    double q_norm_sq = 0.0;
    for (int32_t d = 0; d < dim; ++d) {
        const double qv = static_cast<double>(query[d]);
        q_norm_sq += qv * qv;
    }
    const double q_norm = std::sqrt(std::max(q_norm_sq, 1e-24));

    for (int32_t i = 0; i < n_vecs; ++i) {
        const float* row = corpus + static_cast<size_t>(i) * dim;
        double dot = 0.0;
        double r_norm_sq = 0.0;
        for (int32_t d = 0; d < dim; ++d) {
            const double qv = static_cast<double>(query[d]);
            const double rv = static_cast<double>(row[d]);
            dot += qv * rv;
            r_norm_sq += rv * rv;
        }
        const double denom = q_norm * std::sqrt(std::max(r_norm_sq, 1e-24));
        scores_out[i] = static_cast<float>(dot / denom);
    }
}

}  // extern "C"
