<template>
  <div class="dashboard-page">
    <header class="dashboard-header">
      <div>
        <div class="brand-mark">NEURAL BRIDGE</div>
        <p class="eyebrow">VEATIC-124 evidence dashboard</p>
        <h1>From Raw Losses To Bridge Wins</h1>
      </div>
      <nav class="header-actions" aria-label="Dashboard navigation">
        <RouterLink to="/neuro-viewer">Neural response monitor</RouterLink>
      </nav>
    </header>

    <main class="dashboard-grid">
      <section class="hero-panel">
        <div>
          <p class="eyebrow">Current claim</p>
          <h2>Event and spike ranking survived the controls. Exact continuous arousal forecasting did not become the headline.</h2>
        </div>
        <div class="hero-metrics">
          <div v-for="metric in headlineMetrics" :key="metric.label" class="metric-tile">
            <span>{{ metric.value }}</span>
            <label>{{ metric.label }}</label>
          </div>
        </div>
      </section>

      <section class="timeline-panel">
        <div class="section-heading">
          <p class="eyebrow">Progression</p>
          <h2>The bridge made TRIBE useful</h2>
        </div>
        <div class="timeline">
          <article v-for="item in progression" :key="item.title" class="timeline-item" :class="item.tone">
            <div class="timeline-dot"></div>
            <div class="timeline-copy">
              <span>{{ item.stage }}</span>
              <h3>{{ item.title }}</h3>
              <p>{{ item.detail }}</p>
            </div>
            <div class="timeline-score">
              <strong>{{ item.primary }}</strong>
              <small>{{ item.secondary }}</small>
            </div>
          </article>
        </div>
      </section>

      <section class="comparison-panel">
        <div class="section-heading">
          <p class="eyebrow">Blocked spike row</p>
          <h2>VEATIC-124 v2 baseline</h2>
        </div>
        <div class="bar-chart" aria-label="Blocked spike PR-AUC comparison">
          <div v-for="bar in baselineBars" :key="bar.label" class="bar-row">
            <span>{{ bar.label }}</span>
            <div class="bar-track">
              <i :style="{ width: `${bar.value * 100}%` }"></i>
            </div>
            <strong>{{ formatPercent(bar.value) }}</strong>
          </div>
        </div>
      </section>

      <section class="comparison-panel">
        <div class="section-heading">
          <p class="eyebrow">Raw audit</p>
          <h2>Compression beat raw width</h2>
        </div>
        <div class="lane-list">
          <article v-for="lane in rawAuditLanes" :key="lane.label" class="lane-card" :class="{ winner: lane.winner }">
            <div>
              <h3>{{ lane.label }}</h3>
              <p>{{ lane.note }}</p>
            </div>
            <strong>{{ formatPercent(lane.prAuc) }}</strong>
          </article>
        </div>
      </section>

      <section class="wide-panel">
        <div class="section-heading">
          <p class="eyebrow">Post-v2 head layer</p>
          <h2>AR plus compressed cortical features is the honest next layer</h2>
        </div>
        <div class="head-grid">
          <article v-for="head in trainedHeadCards" :key="head.label" class="head-card" :class="head.tone">
            <span>{{ head.label }}</span>
            <strong>{{ formatPercent(head.value) }}</strong>
            <p>{{ head.note }}</p>
          </article>
        </div>
      </section>

      <section class="wide-panel source-panel">
        <div class="section-heading">
          <p class="eyebrow">Evidence sources</p>
          <h2>Tracked lightweight artifacts</h2>
        </div>
        <ul>
          <li v-for="source in sources" :key="source">{{ source }}</li>
        </ul>
      </section>
    </main>
  </div>
</template>

<script setup>
import { RouterLink } from 'vue-router'

const headlineMetrics = [
  { label: 'videos in frozen VEATIC run', value: '124/124' },
  { label: 'accepted rows', value: '10,357' },
  { label: 'dense AGAIN substrate', value: '995/995' }
]

const progression = [
  {
    stage: 'Initial raw/global signal',
    title: 'Weak raw summary lanes barely beat AR',
    detail: 'The 6-feature fast/default row did not pass controls at the main blocked spike threshold.',
    primary: '+0.83 pp',
    secondary: 'fast/default vs AR',
    tone: 'warning'
  },
  {
    stage: 'VEATIC-124 v2',
    title: 'PCA64-delta became the frozen baseline',
    detail: 'Compressed cortical dynamics beat AR, shuffled, and random controls on the strongest blocked spike row.',
    primary: '+5.67 pp',
    secondary: 'PCA64-delta vs AR',
    tone: 'success'
  },
  {
    stage: 'Raw representation audit',
    title: 'Raw uncompressed ridge lost to compressed lanes',
    detail: 'Full-width raw cortical predictions were valid, but did not beat the PCA64-delta comparator on primary grouped targets.',
    primary: '-0.03 pp',
    secondary: 'raw vs PCA64-delta',
    tone: 'warning'
  },
  {
    stage: 'Post-v2 heads',
    title: 'AR + PCA128 passed the incremental gate',
    detail: 'The trained-head layer recomputed AR and controls fresh, then showed incremental spike-ranking signal from compressed cortical features.',
    primary: '+1.44 pp',
    secondary: 'AR+PCA128 vs AR',
    tone: 'success'
  }
]

const baselineBars = [
  { label: 'AR', value: 0.1969 },
  { label: 'Shuffled', value: 0.1840 },
  { label: 'Random', value: 0.1944 },
  { label: 'Fast/default raw summary', value: 0.2052 },
  { label: 'Global delta', value: 0.2196 },
  { label: 'PCA64', value: 0.2455 },
  { label: 'PCA64-delta', value: 0.2536 }
]

const rawAuditLanes = [
  {
    label: 'Raw current ridge',
    prAuc: 0.385608,
    note: 'Full 20,484-width raw cortical lane; valid but not promoted.',
    winner: false
  },
  {
    label: 'PCA64-delta comparator',
    prAuc: 0.385886,
    note: 'Frozen v2 comparator retained after the audit.',
    winner: false
  },
  {
    label: 'PCA128 causal 2s mean',
    prAuc: 0.425140,
    note: 'Best event/spike candidate for the model-ready tensor contract.',
    winner: true
  },
  {
    label: 'ROI parcel features',
    prAuc: 0.403874,
    note: 'Compact side branch with future-change value.',
    winner: false
  }
]

const trainedHeadCards = [
  {
    label: 'AR only',
    value: 0.417017,
    note: 'Fresh grouped mean baseline from the trained-head run.',
    tone: 'neutral'
  },
  {
    label: 'PCA128 only',
    value: 0.298886,
    note: 'Did not stably beat AR by itself.',
    tone: 'warning'
  },
  {
    label: 'AR + PCA128',
    value: 0.431465,
    note: 'Passed the primary grouped incremental neural-value gate.',
    tone: 'success'
  },
  {
    label: 'Residualized AR + PCA128',
    value: 0.429976,
    note: 'Also passed, supporting incremental compressed cortical signal.',
    tone: 'success'
  }
]

const sources = [
  'benchmarks/veatic/veatic_124_confirmatory_benchmark_report_20260616.md',
  'benchmarks/veatic/veatic_124_retest_event_spike_core_20260616.md',
  'docs/veatic_raw_representation_audit.md',
  'outputs/veatic_124_raw_representation_audit_primary_20260620_152411/raw_vs_compressed_leaderboard.csv',
  'outputs/veatic_124_frozen_tensor_trained_heads_mps_20260620_full/trained_head_report.md'
]

const formatPercent = value => `${(Number(value) * 100).toFixed(2)}%`
</script>

<style scoped>
.dashboard-page {
  min-height: 100vh;
  background:
    radial-gradient(circle at 14% 10%, rgba(69, 199, 177, 0.18), transparent 28%),
    radial-gradient(circle at 82% 12%, rgba(255, 138, 76, 0.16), transparent 30%),
    linear-gradient(145deg, #06100e 0%, #101716 48%, #170f0b 100%);
  color: #ecf8f2;
  font-family: 'Space Grotesk', 'JetBrains Mono', sans-serif;
  padding: 28px;
}

.dashboard-header,
.hero-panel,
.timeline-panel,
.comparison-panel,
.wide-panel {
  border: 1px solid rgba(236, 248, 242, 0.14);
  background: rgba(6, 14, 13, 0.78);
  box-shadow: 0 22px 70px rgba(0, 0, 0, 0.28);
}

.dashboard-header {
  display: flex;
  justify-content: space-between;
  gap: 22px;
  align-items: flex-start;
  padding: 22px;
  margin-bottom: 18px;
}

.brand-mark,
.header-actions a {
  display: inline-flex;
  border: 1px solid rgba(255, 205, 123, 0.34);
  background: rgba(255, 143, 75, 0.12);
  color: #fff0d7;
  padding: 9px 13px;
  text-transform: uppercase;
  letter-spacing: 0.08em;
  text-decoration: none;
  font-weight: 800;
}

.eyebrow {
  color: #88d7bd;
  font-size: 0.75rem;
  letter-spacing: 0.14em;
  margin: 12px 0 7px;
  text-transform: uppercase;
}

h1,
h2,
h3,
p {
  margin: 0;
}

h1 {
  max-width: 920px;
  font-size: clamp(2.6rem, 6vw, 6.5rem);
  line-height: 0.92;
}

h2 {
  font-size: clamp(1.35rem, 2.3vw, 2.4rem);
  line-height: 1.05;
}

h3 {
  font-size: 1rem;
}

.dashboard-grid {
  display: grid;
  grid-template-columns: minmax(0, 1.08fr) minmax(360px, 0.92fr);
  gap: 18px;
}

.hero-panel,
.timeline-panel,
.comparison-panel,
.wide-panel {
  padding: 20px;
}

.hero-panel {
  grid-column: 1 / -1;
  display: grid;
  grid-template-columns: minmax(0, 1fr) auto;
  gap: 24px;
  align-items: end;
}

.hero-metrics {
  display: flex;
  gap: 12px;
}

.metric-tile {
  min-width: 155px;
  padding: 15px;
  background: rgba(255, 255, 255, 0.045);
  border: 1px solid rgba(255, 255, 255, 0.09);
}

.metric-tile span {
  display: block;
  font-size: 1.55rem;
  font-weight: 900;
}

.metric-tile label,
.timeline-copy span,
.timeline-score small,
.head-card span {
  color: #9fc9bb;
  font-size: 0.72rem;
  text-transform: uppercase;
  letter-spacing: 0.08em;
}

.section-heading {
  margin-bottom: 18px;
}

.timeline {
  display: grid;
  gap: 12px;
}

.timeline-item {
  display: grid;
  grid-template-columns: 16px minmax(0, 1fr) 110px;
  gap: 13px;
  align-items: center;
  padding: 14px;
  border: 1px solid rgba(255, 255, 255, 0.1);
  background: rgba(255, 255, 255, 0.04);
}

.timeline-dot {
  width: 12px;
  height: 12px;
  border-radius: 999px;
  background: #ffb75f;
  box-shadow: 0 0 20px rgba(255, 183, 95, 0.4);
}

.timeline-item.success .timeline-dot {
  background: #42e0bd;
  box-shadow: 0 0 20px rgba(66, 224, 189, 0.4);
}

.timeline-copy {
  display: grid;
  gap: 6px;
}

.timeline-copy p,
.head-card p,
.lane-card p,
.source-panel li {
  color: #bdd5cd;
  line-height: 1.48;
}

.timeline-score {
  text-align: right;
}

.timeline-score strong {
  display: block;
  color: #fff0d7;
  font-size: 1.35rem;
}

.bar-chart,
.lane-list,
.head-grid {
  display: grid;
  gap: 10px;
}

.bar-row {
  display: grid;
  grid-template-columns: 160px minmax(0, 1fr) 70px;
  gap: 10px;
  align-items: center;
}

.bar-row span {
  color: #dceee8;
}

.bar-row strong {
  text-align: right;
  color: #fff0d7;
}

.bar-track {
  height: 13px;
  background: rgba(255, 255, 255, 0.075);
  overflow: hidden;
}

.bar-track i {
  display: block;
  height: 100%;
  background: linear-gradient(90deg, #3fd1b2, #ffb35a);
}

.lane-card,
.head-card {
  display: flex;
  justify-content: space-between;
  gap: 16px;
  padding: 14px;
  border: 1px solid rgba(255, 255, 255, 0.1);
  background: rgba(255, 255, 255, 0.04);
}

.lane-card.winner,
.head-card.success {
  border-color: rgba(66, 224, 189, 0.42);
  background: rgba(66, 224, 189, 0.08);
}

.head-card.warning {
  border-color: rgba(255, 183, 95, 0.36);
  background: rgba(255, 183, 95, 0.07);
}

.lane-card strong,
.head-card strong {
  flex: 0 0 auto;
  color: #fff0d7;
  font-size: 1.35rem;
}

.wide-panel {
  grid-column: 1 / -1;
}

.head-grid {
  grid-template-columns: repeat(4, minmax(0, 1fr));
}

.head-card {
  display: grid;
}

.source-panel ul {
  margin: 0;
  padding-left: 18px;
}

@media (max-width: 980px) {
  .dashboard-grid,
  .hero-panel,
  .dashboard-header,
  .head-grid {
    grid-template-columns: 1fr;
  }

  .dashboard-header,
  .hero-panel {
    display: grid;
  }

  .hero-metrics {
    flex-wrap: wrap;
  }
}
</style>
