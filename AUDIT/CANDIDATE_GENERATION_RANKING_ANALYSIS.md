# CANDIDATE GENERATION VS. CANDIDATE RANKING: Error Decomposition & Physical Bottleneck Analysis

**Document Path:** `AUDIT/CANDIDATE_GENERATION_RANKING_ANALYSIS.md`  
**Authoritative Date:** November 21, 2026  
**Auditor:** Senior Wireless Communications Specialist & Multimodal ML Scientist  
**Repository:** [https://github.com/moazzem-hossain-majumder/DeepSenseV2V](https://github.com/moazzem-hossain-majumder/DeepSenseV2V)  

---

## 1. Executive Summary

A central finding of this research is the **orthogonal decomposition of mmWave beam management errors into Candidate Generation Failure (Exclusion) versus Candidate Ranking Failure (Selection)**.

Across 16,643 zero-leakage validation windows across Scenarios 36–39, we prove that **84.7% of severe tracking failures (>10 dB power drop)** under a standard 20-candidate geometric shortlist are caused by the candidate generator failing to include the optimal beam, while only **15.3%** are caused by the neural ranking model selecting suboptimal beams from within the available set.

**Critical Methodological Boundary:**  
This decomposition is an **offline oracle diagnostic** evaluated against saved ground-truth beam power vectors. It quantifies the theoretical performance ceiling of a candidate generator, but must **never be described as a deployable real-time capability**, because a physical transceiver only observes channel powers for the $K$ beams it actively probes.

---

## 2. Mathematical Formalization of the Decomposition

Let $\mathcal{B} = \{1, \dots, 256\}$ be the full codebook of directional beams.  
For a given prediction window $w$, let:
- $\mathbf{P}_w = [P_{w, 1}, \dots, P_{w, 256}]$ be the true received power vector in dBm.
- $P_{w}^* = \max_{b \in \mathcal{B}} P_{w, b}$ be the global maximum received power.
- $y_w^* = \arg\max_{b \in \mathcal{B}} P_{w, b}$ be the globally optimal beam index.
- $\mathcal{C}_{w, M} \subset \mathcal{B}$ be the candidate set of size $M$ produced by the candidate generator (e.g., $M = 20$ or $40$ from analytic geometry).
- $\mathcal{S}_{w, K} \subset \mathcal{C}_{w, M}$ be the final shortlist of size $K$ selected by the ranking model (e.g., $K = 5$).

### 2.1 Error Definitions
1. **Total Achieved Power Loss:**
   $$\Delta P_{\text{total}}(w) = P_w^* - \max_{s \in \mathcal{S}_{w, K}} P_{w, s} \ge 0\text{ dB}$$
2. **Oracle Candidate Exclusion Loss (Generation Error):**
   The minimum unavoidable power loss given the candidate set $\mathcal{C}_{w, M}$:
   $$\Delta P_{\text{excl}}(w) = P_w^* - \max_{c \in \mathcal{C}_{w, M}} P_{w, c} \ge 0\text{ dB}$$
   *Candidate Exclusion Failure* occurs when $\Delta P_{\text{excl}}(w) > \tau$, where $\tau$ is the operational threshold (e.g., $\tau = 3.0\text{ dB}$ for near-optimal coverage, or $\tau = 10.0\text{ dB}$ for severe outage).
3. **Shortlist Ranking Loss (Ranking Error):**
   The additional power lost because the ranking model failed to select the best beam available in $\mathcal{C}_{w, M}$:
   $$\Delta P_{\text{rank}}(w) = \max_{c \in \mathcal{C}_{w, M}} P_{w, c} - \max_{s \in \mathcal{S}_{w, K}} P_{w, s} \ge 0\text{ dB}$$

### 2.2 Exact Additive Identity
Because $\mathcal{S}_{w, K} \subseteq \mathcal{C}_{w, M} \subset \mathcal{B}$, the total power loss decomposes additively without approximation:
$$\Delta P_{\text{total}}(w) = \Delta P_{\text{excl}}(w) + \Delta P_{\text{rank}}(w)$$

---

## 3. Quantitative Analysis Across Scenarios and Candidate Budgets

### 3.1 Severe Failure Decomposition (Threshold $\tau = 10.0\text{ dB}$)
Under the primary 20-candidate geometry-constrained transfer protocol ($M=20$, $K=5$, 3 seeds, equal scenario weighting):
- Total severe outage rate ($\Delta P_{\text{total}} > 10\text{ dB}$): **16.80% of validation windows**.
- Oracle exclusion outage rate ($\Delta P_{\text{excl}} > 10\text{ dB}$): **14.23% of validation windows**.
- Ranking-only severe outage rate ($\Delta P_{\text{excl}} \le 10\text{ dB}$ but $\Delta P_{\text{total}} > 10\text{ dB}$): **2.57% of validation windows**.

$$\text{Candidate Exclusion Share } E_{10\text{dB}} = \frac{14.23\%}{16.80\%} = \mathbf{84.70\%}$$
$$\text{Ranking Error Share } R_{10\text{dB}} = \frac{2.57\%}{16.80\%} = \mathbf{15.30\%}$$

### 3.2 Primary Near-Optimal Decomposition (Threshold $\tau = 3.0\text{ dB}$)
- Total 3 dB outage rate ($\Delta P_{\text{total}} > 3\text{ dB}$): **21.97% of validation windows** (78.03% success).
- Oracle exclusion rate ($\Delta P_{\text{excl}} > 3\text{ dB}$): **15.02% of validation windows**.
- Ranking error rate ($\Delta P_{\text{excl}} \le 3\text{ dB}$ but $\Delta P_{\text{total}} > 3\text{ dB}$): **6.95% of validation windows**.

$$\text{Candidate Exclusion Share } E_{3\text{dB}} = \frac{15.02\%}{21.97\%} = \mathbf{68.37\%}$$
$$\text{Ranking Error Share } R_{3\text{dB}} = \frac{6.95\%}{21.97\%} = \mathbf{31.63\%}$$

---

### 3.3 Scenario-by-Scenario Breakdown ($M=20, K=5$)

```
+-------------------------------------------------------------------------------------------------------------------------+
|                                  CANDIDATE EXCLUSION VS. RANKING ERROR BY SCENARIO                                      |
+--------------+------------------+---------------------+---------------------+---------------------+---------------------+
| Scenario     | Driving Setting  | Total Severe Outage | Oracle Exclusion    | Ranking Failure     | Exclusion Share (%) |
|              |                  | (>10 dB Miss Rate)  | (>10 dB in M=20)    | (Within Top 20)     | (E_10dB)            |
+--------------+------------------+---------------------+---------------------+---------------------+---------------------+
| Scenario 36  | Day Urban LOS    | 12.14%              | 10.51%              | 1.63%               | 86.57%              |
| Scenario 37  | Night Following  | 18.92%              | 16.34%              | 2.58%               | 86.36%              |
| Scenario 38  | Day Corridor     | 13.45%              | 11.20%              | 2.25%               | 83.27%              |
| Scenario 39  | Highway Overtake | 22.69%              | 18.87%              | 3.82%               | 83.16%              |
+--------------+------------------+---------------------+---------------------+---------------------+---------------------+
| Macro Mean   | Balanced 4-Scen  | 16.80%              | 14.23%              | 2.57%               | 84.70%              |
+--------------+------------------+---------------------+---------------------+---------------------+---------------------+
```

---

### 3.4 Impact of Shortlist Width Widening ($M = 5, 10, 20, 40, 64$)

```
+-------------------------------------------------------------------------------------------------------------------------+
|                                    CANDIDATE SET SCALING & THE OVERHEAD TRADE-OFF                                       |
+-----+--------------+-------------------+--------------------+------------------+--------------------+-------------------+
| M   | Probed K     | Oracle Exclusion  | Deployed 3 dB      | Probing Overhead | Probing Time       | Net Effective     |
|     | (Shortlist)  | Outage (>3 dB)    | Success Rate       | (% of 20 ms MAC) | (T_probe = 0.1 ms) | Rate (Gbps)       |
+-----+--------------+-------------------+--------------------+------------------+--------------------+-------------------+
| 5   | 5 (Geo Base) | 21.96%            | 78.04%             | 2.5%             | 0.5 ms             | 5.06 Gbps (PEAK)  |
| 10  | 5            | 18.42%            | 78.03%             | 5.0%             | 1.0 ms             | 4.98 Gbps         |
| 20  | 5 (Rerank)   | 15.02%            | 78.03%             | 10.0%            | 2.0 ms             | 4.66 Gbps         |
| 40  | 5 (Wide)     | 12.15%            | 78.04%             | 20.0%            | 4.0 ms             | 4.13 Gbps         |
| 64  | 5            | 10.40%            | 78.05%             | 32.0%            | 6.4 ms             | 3.51 Gbps         |
| 256 | 256 (Exhaust)| 0.00% (Oracle)    | 100.0% (Oracle)    | 128.0% (OVERFLOW)| 25.6 ms            | 0.00 Gbps (STALL) |
+-----+--------------+-------------------+--------------------+------------------+--------------------+-------------------+
```

#### Key Physical Takeaways:
1. **The Diminishing Return of Candidate Widening:** Expanding $M$ from 20 to 40 reduces the oracle exclusion rate from 15.02% to 12.15% (a 2.87 pp theoretical gain). However, in realistic evaluation (Stage 267), the visual reranker realized only **+0.011 pp of empirical gain**, because the neural model struggled to correctly rank candidates in the 21–40 range.
2. **The Probing Overhead Penalty:** While larger candidate sets reduce exclusion, probing those candidates eats into user data payload time. Net effective throughput peaks sharply at $K = 5$ (5.06 Gbps) and collapses as candidate sets widen.

---

## 4. Verification of No Information Leakage

We audited the calculation code (`src/shortlist_transfer_audit.py` and `src/width_attribution.py`):
1. **Strictly Causal Inputs:** Candidate generation $\mathcal{C}_{w, M}$ is computed strictly from historical GPS observations up to time $t$ ($t-8, t-6, t-4, t-2, t$).
2. **Post-Hoc Oracle Separation:** True received powers at $t+5$ are accessed **only** to evaluate the diagnostic ground-truth labels. No target power or target beam information is leaked to the candidate generator or ranking network during training or inference.
3. **Disjoint Partitioning:** All statistics are computed on validation windows belonging to distinct trajectory runs that share zero sequence keys with the training partition.

---

## 5. Architectural Recommendations for IEEE Manuscript

In Section V of the revised manuscript:
- Present Candidate Exclusion as the **primary structural bottleneck** of vehicular beam tracking.
- Conclude that future mmWave wireless research should prioritize **kinematically adaptive candidate-set dimensioning** rather than increasingly deep neural ranking architectures.
