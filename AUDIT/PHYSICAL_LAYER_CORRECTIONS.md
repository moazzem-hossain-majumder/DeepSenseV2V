# PHYSICAL-LAYER CORRECTIONS & COMMUNICATION-COST AUDIT

**Document Path:** `AUDIT/PHYSICAL_LAYER_CORRECTIONS.md`  
**Authoritative Date:** November 21, 2026  
**Auditor:** Senior IEEE Vehicular Technology & mmWave Communications Specialist  
**Repository:** [https://github.com/moazzem-hossain-majumder/DeepSenseV2V](https://github.com/moazzem-hossain-majumder/DeepSenseV2V)  

---

## 1. Executive Summary

This report establishes the mandatory physical-layer mathematical corrections for mmWave beam management at 60 GHz in vehicle-to-vehicle (V2V) environments. 

It explicitly corrects the **arithmetic error in the Doppler coherence-time calculation** previously reported in `COMMUNICATION_COST_AUDIT.md` and `IEEE_REVIEWER_REPORT.md` ($4.16\text{ ms}$ vs. the true arithmetic value of $83.33\,\mu\text{s}$), rigorously defines and distinguishes five distinct physical time scales, and formalizes an IEEE 802.11ad/ay-compliant analytical communication-cost model.

---

## 2. Derivation and Correction of Physical Time Scales

### 2.1 The Coherence-Time Arithmetic Error & Rectification
The previous report stated:
$$T_c \approx \frac{c}{2 f_c v} = 4.16\text{ ms}$$
for parameters:
- Carrier frequency: $f_c = 60\text{ GHz} = 60 \times 10^9\text{ Hz}$
- Relative vehicular velocity: $v = 30\text{ m/s}$ ($\approx 108\text{ km/h}$)
- Speed of light: $c = 3 \times 10^8\text{ m/s}$

#### Exact Recalculation:
1. Maximum Doppler Frequency Shift:
   $$f_{d,\text{max}} = \frac{v}{\lambda} = \frac{v \cdot f_c}{c} = \frac{30\text{ m/s} \times 60 \times 10^9\text{ Hz}}{3 \times 10^8\text{ m/s}} = 6000\text{ Hz} = 6.0\text{ kHz}$$
2. Small-Scale Doppler Phase Coherence Time (Clarke's / Jakes' model at 50% correlation):
   $$T_{c,\text{small}} \approx \sqrt{\frac{9}{16 \pi f_d^2}} \approx \frac{1}{2 f_{d,\text{max}}} = \frac{3 \times 10^8}{2 \times 60 \times 10^9 \times 30} = \frac{3 \times 10^8}{3.6 \times 10^{12}} = 8.333 \times 10^{-5}\text{ s} = \mathbf{83.33\,\mu\text{s}}$$

**The value $4.16\text{ ms}$ was off by a factor of exactly 50.**  
($4.16\text{ ms}$ corresponds to a carrier frequency of $1.2\text{ GHz}$ at 30 m/s, or $60\text{ GHz}$ at walking speed $v = 0.6\text{ m/s}$.)

---

### 2.2 Distinguishing Physical Time Scales in Directional mmWave V2V
In mmWave communications, **small-scale channel phase coherence** must never be conflated with **spatial beam-alignment coherence**. They operate on vastly different orders of magnitude:

```
+-------------------------------------------------------------------------------------------------------------------------+
|                                          PHYSICAL TIME SCALES IN 60 GHz V2V                                             |
+-----+--------------------------------------+---------------------+------------------------------------------------------+
| No. | Physical Quantity                    | Typical Value       | Physical Definition & Operational Meaning            |
+-----+--------------------------------------+---------------------+------------------------------------------------------+
| 1   | Maximum Doppler Frequency (f_d)      | 6.0 kHz             | Maximum frequency shift at 30 m/s and 60 GHz.        |
+-----+--------------------------------------+---------------------+------------------------------------------------------+
| 2   | Small-Scale Phase Coherence Time     | 83.3 μs             | Duration over which multipath channel phase remains  |
|     | (T_c,small)                          |                     | correlated. Handled by OFDM pilot symbols/equalizer. |
+-----+--------------------------------------+---------------------+------------------------------------------------------+
| 3   | Directional Array Doppler Dispersion | ~370 Hz             | Doppler spread inside a 5° narrow beamwidth:         |
|     | (Δf_d,beam)                          |                     | Δf_d = f_d · |sin θ| · HPBW. Lengthens in-beam phase |
|     |                                      |                     | coherence to T_c,dir ≈ 1.35 ms.                      |
+-----+--------------------------------------+---------------------+------------------------------------------------------+
| 4   | Spatial Beam-Alignment Coherence     | 29 ms to 175 ms     | Duration before vehicle motion causes LOS direction  |
|     | (Δt_beam)                            |                     | to exit the 5° Half-Power Beamwidth (HPBW).          |
+-----+--------------------------------------+---------------------+------------------------------------------------------+
| 5   | MAC Management Frame Duration        | 20.0 ms             | Standard IEEE 802.11ad/ay Beacon Interval / Data     |
|     | (T_frame)                            |                     | Transmission Interval (DTI) allocation block.        |
+-----+--------------------------------------+---------------------+------------------------------------------------------+
| 6   | Beam Probing Duration (T_probe)      | 100 μs per beam     | Duration of one IEEE 802.11ad Sector-Level Sweep     |
|     |                                      |                     | (SLS) short packet transmission.                     |
+-----+--------------------------------------+---------------------+------------------------------------------------------+
| 7   | Feedback & Processing Latency        | 350 μs              | DSP processing (150 μs) + SSW-Feedback frame (200 μs)|
|     | (T_proc + T_fb)                      |                     | transmission over omnidirectional / quasi-omni link. |
+-----+--------------------------------------+---------------------+------------------------------------------------------+
| 8   | Forecast Horizon (Δt_pred)           | 500 ms (0.500 s)    | Time gap between last observation t and prediction.  |
+-----+--------------------------------------+---------------------+------------------------------------------------------+
```

#### Mathematical Derivation of Spatial Beam Coherence ($\Delta t_{\text{beam}}$):
Consider two vehicles moving on parallel highway lanes at separation distance $d = 20\text{ m}$ with relative transverse velocity $v_\perp = 10\text{ m/s}$. The angular velocity is:
$$\dot{\theta} = \frac{v_\perp}{d} = \frac{10}{20} = 0.5\text{ rad/s} \approx 28.65^\circ/\text{s}$$
For a $16 \times 16$ planar array with half-power beamwidth $\text{HPBW} \approx 5.0^\circ = 0.08727\text{ rad}$, the time required for the link to drift out of alignment is:
$$\Delta t_{\text{beam}} = \frac{\text{HPBW}}{\dot{\theta}} = \frac{0.08727\text{ rad}}{0.5\text{ rad/s}} = \mathbf{174.5\text{ ms}}$$
Under extreme worst-case maneuvering ($v_\perp = 30\text{ m/s}$ at close range $d = 10\text{ m}$):
$$\dot{\theta}_{\text{worst}} = \frac{30}{10} = 3.0\text{ rad/s} \approx 171.9^\circ/\text{s} \implies \Delta t_{\text{beam,worst}} = \frac{0.08727}{3.0} = \mathbf{29.1\text{ ms}}$$
**Physical Insight:**  
Because spatial beam coherence ($29\text{--}175\text{ ms}$) is strictly greater than the 20 ms MAC transmission frame, a beam chosen at the start of a 20 ms frame remains spatially valid throughout the frame! However, an exhaustive 256-beam sweep requires $25.6\text{ ms}$, which directly violates the 20 ms frame boundary and approaches the worst-case beam coherence limit.

---

## 3. Grounding the 60 GHz Analytical Communication-Cost Model

### 3.1 Assumptions & Standards Provenance
All communication-cost calculations in this repository must be clearly identified as **analytical evaluations based on IEEE 802.11ad/ay PHY/MAC specifications**, evaluated via **offline replay of empirical DeepSense 6G beam powers**:
- **Frame Duration ($T_{\text{frame}}$):** 20 ms (standard Beacon Interval / DTI in IEEE 802.11ad Clause 9.3.2).
- **Probing Time ($T_{\text{probe}}$):** $100\,\mu\text{s}$ per beam (duration of a Sector Sweep [SSW] frame comprising preamble, Golay complementary sequences, and SSW payload at 27.5 Mbps control PHY rate).
- **Processing Time ($T_{\text{proc}}$):** $150\,\mu\text{s}$ (baseband DSP maximum-power selection).
- **Feedback Time ($T_{\text{feedback}}$):** $200\,\mu\text{s}$ (SSW-Feedback and SSW-ACK exchange).
- **Peak Physical Data Rate ($R_{\text{max}}$):** 6.776 Gbps (IEEE 802.11ad MCS 12, 16-QAM, code rate 13/16).
- **Effective Throughput Equation:**
  $$\eta_{\text{eff}}(K) = \left(1 - \frac{K \cdot T_{\text{probe}} + T_{\text{proc}} + T_{\text{feedback}}}{T_{\text{frame}}}\right) \cdot R_{\text{max}} \cdot \mathbb{P}\left(\Delta P(K) \le 3\text{ dB}\right)$$

---

### 3.2 Corrected Analytical Communication-Cost Table

```
+---------------------------------------------------------------------------------------------------------------------------------------------------+
|                                        60 GHz VEHICULAR BEAM MANAGEMENT OVERHEAD & THROUGHPUT TABLE                                               |
+--------------------------+------------+------------+---------------+-------------------+-----------------+-------------------+--------------------+
| Strategy                 | Candidates | Probing    | Total Align   | Frame Overhead    | Coherence Safety| 3 dB Power Success| Effective Data     |
|                          | Budget (K) | Time (ms)  | Latency (ms)  | Penalty (%)       | Margin (T < Δt) | Rate (Empirical)  | Throughput (Gbps)  |
+--------------------------+------------+------------+---------------+-------------------+-----------------+-------------------+--------------------+
| Exhaustive Codebook Sweep| 256        | 25.6 ms    | 25.95 ms      | 129.8% (OVERFLOW) | VIOLATED (<1.0x)| 100.0% (Oracle)   | 0.00 Gbps (OUTAGE) |
| Broad Sector Sweep       | 64         | 6.4 ms     | 6.75 ms       | 33.8%             | MARGINAL (4.3x) | 95.8% (Empirical) | 4.31 Gbps          |
| Wide Multimodal Rerank   | 40         | 4.0 ms     | 4.35 ms       | 21.8%             | SAFE (6.7x)     | 78.03% (Stage 267)| 4.13 Gbps          |
| Multimodal Rerank (Ours) | 20         | 2.0 ms     | 2.35 ms       | 11.8%             | SAFE (12.4x)    | 78.02% (Stage 267)| 4.66 Gbps          |
| Analytic Geometry (Ours) | 5          | 0.5 ms     | 0.85 ms       | 4.25%             | OPTIMAL (34.2x) | 78.04% (Stage 267)| 5.06 Gbps (PEAK)   |
| Single-Beam Prediction   | 1          | 0.1 ms     | 0.45 ms       | 2.25%             | OPTIMAL (64.7x) | 48.33% (Top-1)    | 3.19 Gbps          |
+--------------------------+------------+------------+---------------+-------------------+-----------------+-------------------+--------------------+
Notes: Evaluated analytically within a 20 ms MAC transmission frame. Effective throughput assumes 6.776 Gbps raw MCS 12 rate.
Exhaustive sweeping exceeds the 20 ms frame duration (129.8% overhead), causing MAC transmission stall and link blackout.
```

---

## 4. Manuscript Sections Requiring Revision

The following documents contain the superseded $4.16\text{ ms}$ calculation or conflate small-scale Doppler coherence with spatial beam coherence and must be updated:
1. `COMMUNICATION_COST_AUDIT.md`: Replace Section 1 equation and derivations with the corrected formulas in Section 2 of this report.
2. `IEEE_REVIEWER_REPORT.md`: Correct lines 35–40 in Reviewer 1's section to reflect $T_{c,\text{small}} = 83.33\,\mu\text{s}$ and $\Delta t_{\text{beam}} \approx 175\text{ ms}$.
3. `paper/TVT/manuscript.tex` & `paper/V2V_BEAM_FORECASTING_MANUSCRIPT.md`: Ensure Section 1 and Section 6 explicitly state that candidate shortlisting overhead is an analytical evaluation based on IEEE 802.11ad specifications, avoiding any claim that probing latency was physically measured in hardware.
