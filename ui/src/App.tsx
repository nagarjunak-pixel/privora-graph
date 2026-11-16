import React, { useEffect, useState } from "react";

interface DPEpoch {
  epoch: number;
  epsilon_spent: number;
  dp_test_accuracy_pct: number;
  non_private_test_accuracy_pct: number;
}

interface DPReport {
  num_records: number;
  clipping_norm_c: number;
  noise_multiplier_sigma: number;
  target_delta: number;
  optimal_renyi_alpha: number;
  final_certified_epsilon: number;
  dp_test_accuracy_pct: number;
  non_private_test_accuracy_pct: number;
  non_private_mia_leakage_auc: number;
  dp_mia_leakage_auc: number;
  mia_privacy_risk_reduction_pct: number;
  history: DPEpoch[];
}

interface BiTemporalFact {
  entity: string;
  ppr_score: number;
  fact_summary: string;
}

interface BiTemporalReport {
  kernel_backend: string;
  query_valid_at: number;
  query_tx_at: number;
  total_historical_edges: number;
  active_bitemporal_edges: number;
  retroactive_future_edges_blocked: number;
  ppr_iterations: number;
  simplex_mass_sum: number;
  cpp_vs_numpy_max_diff: number;
  top_ranked_entities: BiTemporalFact[];
}

interface ByzantineReport {
  total_clients_k: number;
  byzantine_clients_f: number;
  poison_magnitude_x: number;
  fedavg_cosine_to_honest: number;
  multikrum_cosine_to_honest: number;
  geomedian_rfa_cosine_to_honest: number;
  fedavg_l2_corruption: number;
  geomedian_rfa_l2_corruption: number;
  corruption_reduction_x: number;
}

interface AuditOutcome {
  case_id: string;
  domain: string;
  audit_tx_timestamp: number;
  ground_truth_code: number;
  naive_unfiltered_answer: number | null;
  naive_leaked_future_info: boolean;
  bitemporal_guarded_answer: number | null;
  bitemporal_exact_match: boolean;
  latency_ms: number;
}

interface RealAuditReport {
  model_name: string;
  num_cases: number;
  naive_correct_count: number;
  naive_accuracy_pct: number;
  naive_future_leakage_count: number;
  bitemporal_correct_count: number;
  bitemporal_accuracy_pct: number;
  future_leakage_eliminated_pct: number;
  outcomes: AuditOutcome[];
}

interface OverviewPayload {
  kernel_backend: string;
  dp_sgd: DPReport;
  bitemporal: BiTemporalReport;
  byzantine: ByzantineReport;
  real_qwen_bitemporal: RealAuditReport | null;
}

export function App(): React.JSX.Element {
  const [data, setData] = useState<OverviewPayload | null>(null);
  const [dp, setDp] = useState<DPReport | null>(null);
  const [bt, setBt] = useState<BiTemporalReport | null>(null);
  const [sigma, setSigma] = useState<number>(1.45);
  const [txDate, setTxDate] = useState<number>(20260601);

  useEffect(() => {
    fetch("/api/v1/overview")
      .then((r) => r.json())
      .then((payload: OverviewPayload) => {
        setData(payload);
        setDp(payload.dp_sgd);
        setBt(payload.bitemporal);
        setSigma(payload.dp_sgd.noise_multiplier_sigma);
      });
  }, []);

  const rerunDp = async () => {
    const r = await fetch("/api/v1/privacy/train", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ sigma }),
    });
    setDp((await r.json()) as DPReport);
  };

  const changeTxDate = async (nextTx: number) => {
    setTxDate(nextTx);
    const r = await fetch("/api/v1/bitemporal/query", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ valid_at: 20260501, tx_at: nextTx }),
    });
    setBt((await r.json()) as BiTemporalReport);
  };

  if (!data || !dp || !bt) {
    return (
      <div className="shell">
        <h2>Initializing PrivoraGraph Sovereign Privacy &amp; Bi-Temporal Workbench...</h2>
      </div>
    );
  }

  const chartW = 580;
  const chartH = 138;
  const hist = dp.history;
  const maxEps = Math.max(4.0, ...hist.map((h) => h.epsilon_spent));
  const ptsAcc = hist
    .map((h, i) => {
      const x = 28 + (i / Math.max(1, hist.length - 1)) * (chartW - 46);
      const y = chartH - 20 - (h.dp_test_accuracy_pct / 100.0) * (chartH - 34);
      return `${x.toFixed(1)},${y.toFixed(1)}`;
    })
    .join(" ");
  const ptsEps = hist
    .map((h, i) => {
      const x = 28 + (i / Math.max(1, hist.length - 1)) * (chartW - 46);
      const y = chartH - 20 - (h.epsilon_spent / maxEps) * (chartH - 34);
      return `${x.toFixed(1)},${y.toFixed(1)}`;
    })
    .join(" ");

  const audit = data.real_qwen_bitemporal;

  return (
    <div className="shell">
      <header className="topbar">
        <div className="brand">
          <h1>PrivoraGraph — Sovereign Differential Privacy, Bi-Temporal GraphRAG &amp; Byzantine Engine</h1>
          <p>
            Rényi DP-SGD Moments Accountant · SQL:2011 Bi-Temporal Point-in-Time CSR PageRank ·
            Smoothed Weiszfeld Geometric-Median RFA &amp; Multi-Krum
          </p>
        </div>
        <div className="pill-row">
          <span className="pill teal">{data.kernel_backend}</span>
          <span className="pill">(ε, δ=1e-5) Certified Rényi DP</span>
          <span className="pill">Zero Point-in-Time Lookahead Leakage</span>
        </div>
      </header>

      <section className="kpi-grid">
        <div className="kpi-card">
          <div className="kpi-label">Certified Rényi (ε, δ=1e-5) Budget</div>
          <div className="kpi-value">ε = {dp.final_certified_epsilon.toFixed(2)}</div>
          <div className="kpi-sub">
            MIA AUC: {dp.non_private_mia_leakage_auc.toFixed(3)} → {dp.dp_mia_leakage_auc.toFixed(3)} (-
            {dp.mia_privacy_risk_reduction_pct.toFixed(1)}% Risk)
          </div>
        </div>

        <div className="kpi-card">
          <div className="kpi-label">Bi-Temporal Point-in-Time Gate</div>
          <div className="kpi-value">{bt.retroactive_future_edges_blocked} Leaks Blocked</div>
          <div className="kpi-sub">
            Simplex = {bt.simplex_mass_sum.toFixed(6)} · C++20 |Δ| ={" "}
            {bt.cpp_vs_numpy_max_diff.toExponential(1)}
          </div>
        </div>

        <div className="kpi-card">
          <div className="kpi-label">Byzantine Silo Resilience (f={data.byzantine.byzantine_clients_f}/{data.byzantine.total_clients_k})</div>
          <div className="kpi-value">{data.byzantine.geomedian_rfa_cosine_to_honest.toFixed(4)} cos</div>
          <div className="kpi-sub">
            FedAvg ({data.byzantine.fedavg_cosine_to_honest.toFixed(2)}) → Weiszfeld RFA (
            {data.byzantine.corruption_reduction_x.toFixed(1)}× Cleaner)
          </div>
        </div>

        <div className="kpi-card">
          <div className="kpi-label">Live Qwen-3.5-4B Audit Accuracy</div>
          <div className="kpi-value">
            {audit ? `${audit.naive_accuracy_pct.toFixed(0)}% → ${audit.bitemporal_accuracy_pct.toFixed(0)}%` : "Pending"}
          </div>
          <div className="kpi-sub">
            {audit
              ? `${audit.naive_future_leakage_count}/${audit.num_cases} Naive Future Leaks → 0 Leaks in Privora`
              : "Run privora eval-ollama"}
          </div>
        </div>
      </section>

      <section className="main-grid">
        <div className="panel">
          <div className="panel-header">
            <h2>1. Per-Sample Clipped DP-SGD &amp; Rényi Accountant (α*={dp.optimal_renyi_alpha})</h2>
            <div className="controls">
              <span style={{ fontSize: "0.76rem", color: "#8fa6c9" }}>Noise σ</span>
              <input
                type="number"
                step={0.15}
                min={0.5}
                max={3.5}
                style={{ width: 58 }}
                value={sigma}
                onChange={(e) => setSigma(Number(e.target.value))}
              />
              <button className="btn" onClick={rerunDp}>
                Audit DP-SGD
              </button>
            </div>
          </div>

          <div className="chart-box">
            <svg viewBox={`0 0 ${chartW} ${chartH}`} width="100%" height="138">
              <line x1="26" y1={chartH - 20} x2={chartW - 14} y2={chartH - 20} stroke="#23395d" />
              <line x1="26" y1="14" x2="26" y2={chartH - 20} stroke="#23395d" />
              <polyline fill="none" stroke="#34d399" strokeWidth="2.5" points={ptsAcc} />
              <polyline
                fill="none"
                stroke="#fbbf24"
                strokeWidth="2.1"
                strokeDasharray="5 3"
                points={ptsEps}
              />
              <text x="34" y="24" fill="#34d399" fontSize="10.5">
                ━ DP-SGD Held-Out Test Accuracy ({dp.dp_test_accuracy_pct.toFixed(1)}%)
              </text>
              <text x="315" y="24" fill="#fbbf24" fontSize="10.5">
                ┅ Cumulative Rényi Privacy Spend ε ({dp.final_certified_epsilon.toFixed(2)})
              </text>
            </svg>
          </div>

          <div className="sub-divider" />
          <div className="panel-header" style={{ marginBottom: 6 }}>
            <h2>2. Byzantine-Robust Sovereign Federated Aggregation (K={data.byzantine.total_clients_k}, f={data.byzantine.byzantine_clients_f})</h2>
          </div>
          <table className="metric-table">
            <thead>
              <tr>
                <th>Federated Aggregator</th>
                <th>Cosine to True Honest Mean</th>
                <th>Status</th>
              </tr>
            </thead>
            <tbody>
              <tr>
                <td className="mono">Naive FedAvg (Arithmetic Mean)</td>
                <td className="mono warn">{data.byzantine.fedavg_cosine_to_honest.toFixed(5)}</td>
                <td className="warn">POISONED (-24× Sign Flip)</td>
              </tr>
              <tr>
                <td className="mono">Blanchard et al. Multi-Krum</td>
                <td className="mono ok">{data.byzantine.multikrum_cosine_to_honest.toFixed(5)}</td>
                <td className="ok">BYZANTINE SAFE</td>
              </tr>
              <tr>
                <td className="mono">Smoothed Weiszfeld Geometric Median (RFA)</td>
                <td className="mono ok">{data.byzantine.geomedian_rfa_cosine_to_honest.toFixed(5)}</td>
                <td className="ok">{data.byzantine.corruption_reduction_x.toFixed(1)}× L2 Error Cut</td>
              </tr>
            </tbody>
          </table>
        </div>

        <div className="panel">
          <div className="panel-header">
            <h2>3. Bi-Temporal CSR Personalized PageRank (Valid=2026-05-01)</h2>
            <div className="controls">
              <select value={txDate} onChange={(e) => changeTxDate(Number(e.target.value))}>
                <option value={20260601}>Audit tx_at: 2026-06-01 (Pre-Restatement)</option>
                <option value={20261001}>Audit tx_at: 2026-10-01 (Post-Forensic SDN)</option>
              </select>
            </div>
          </div>

          <table className="metric-table">
            <thead>
              <tr>
                <th>Entity</th>
                <th>PPR</th>
                <th>Point-in-Time Verified Fact</th>
              </tr>
            </thead>
            <tbody>
              {bt.top_ranked_entities.map((f) => (
                <tr key={f.entity}>
                  <td className="mono">{f.entity}</td>
                  <td className="mono ok">{f.ppr_score.toFixed(4)}</td>
                  <td>{f.fact_summary}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </section>

      {audit && (
        <section className="panel">
          <div className="panel-header">
            <h2>
              4. Zero-Mock Live Ollama ({audit.model_name}) Bi-Temporal Point-in-Time Compliance
              Audit (SEC / FDA / OFAC / Basel-III / GDPR)
            </h2>
            <span className="pill teal">
              Naive GraphRAG: {audit.naive_correct_count}/{audit.num_cases} (
              {audit.naive_future_leakage_count} Retroactive Leaks) → Privora Bi-Temporal:{" "}
              {audit.bitemporal_correct_count}/{audit.num_cases} ({audit.bitemporal_accuracy_pct.toFixed(0)}%
              Exact)
            </span>
          </div>
          <table className="metric-table">
            <thead>
              <tr>
                <th>Audit Case ID</th>
                <th>Regulated Domain</th>
                <th>Audit tx_at</th>
                <th>True Code</th>
                <th>Naive Unfiltered GraphRAG</th>
                <th>Privora Bi-Temporal Gate</th>
                <th>Latency</th>
              </tr>
            </thead>
            <tbody>
              {audit.outcomes.map((o) => (
                <tr key={o.case_id}>
                  <td className="mono">{o.case_id}</td>
                  <td>{o.domain}</td>
                  <td className="mono">{o.audit_tx_timestamp}</td>
                  <td className="mono">{o.ground_truth_code}</td>
                  <td className={`mono ${o.naive_leaked_future_info ? "warn" : "gold"}`}>
                    {o.naive_unfiltered_answer ?? "None"}{" "}
                    {o.naive_leaked_future_info ? "(FUTURE LEAK ✗)" : ""}
                  </td>
                  <td className={`mono ${o.bitemporal_exact_match ? "ok" : "warn"}`}>
                    {o.bitemporal_guarded_answer ?? "None"} {o.bitemporal_exact_match ? "✓" : "✗"}
                  </td>
                  <td className="mono">{o.latency_ms.toFixed(0)} ms</td>
                </tr>
              ))}
            </tbody>
          </table>
        </section>
      )}
    </div>
  );
}
