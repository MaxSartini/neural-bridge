<template>
  <a class="skip-link" href="#analysis-workspace">Skip to analysis workspace</a>

  <div class="analyst-page">
    <header class="app-header">
      <div class="brand-block">
        <RouterLink class="brand-mark" to="/evidence-dashboard">NEURAL BRIDGE</RouterLink>
        <div>
          <p class="eyebrow">Playback-synchronized response intelligence</p>
          <h1>Analyst workspace</h1>
        </div>
      </div>

      <div class="header-actions" aria-label="Analysis exports">
        <span class="mode-badge">Zero-label inference</span>
        <a
          v-if="selectedAnalysisId"
          class="action-link"
          :href="jsonReportUrl"
          download
        >
          Export JSON
        </a>
        <a
          v-if="selectedAnalysisId"
          class="action-link"
          :href="predictionsCsvUrl"
          download
        >
          Export CSV
        </a>
      </div>
    </header>

    <p class="live-status" aria-live="polite">{{ liveStatus }}</p>

    <section v-if="inventoryLoading" class="state-panel" aria-busy="true">
      <span class="loading-orbit" aria-hidden="true"></span>
      <div>
        <h2>Loading analyses</h2>
        <p>Reading the Neural Bridge v1 analysis inventory.</p>
      </div>
    </section>

    <section v-else-if="inventoryError" class="state-panel error-state" role="alert">
      <div>
        <p class="eyebrow">Inventory unavailable</p>
        <h2>Analyses could not be loaded</h2>
        <p>{{ inventoryError }}</p>
      </div>
      <button type="button" @click="loadAnalyses">Retry</button>
    </section>

    <section v-else-if="!analyses.length" class="state-panel empty-state">
      <div>
        <p class="eyebrow">No completed runs</p>
        <h2>No Neural Bridge analyses are available yet</h2>
        <p>The workspace will populate when the v1 API publishes an analysis resource.</p>
      </div>
      <button type="button" @click="loadAnalyses">Refresh inventory</button>
    </section>

    <div v-else class="analyst-layout">
      <aside class="asset-rail" aria-labelledby="asset-heading">
        <div class="rail-heading">
          <div>
            <p class="eyebrow">Analysis library</p>
            <h2 id="asset-heading">Assets</h2>
          </div>
          <button type="button" class="quiet-button" @click="loadAnalyses" aria-label="Refresh analysis library">
            Refresh
          </button>
        </div>

        <label class="search-field">
          <span>Search analyses</span>
          <input
            v-model.trim="searchQuery"
            type="search"
            placeholder="ID, file, or dataset"
            autocomplete="off"
          />
        </label>

        <div class="asset-count">{{ filteredAnalyses.length }} of {{ analyses.length }}</div>

        <div v-if="filteredAnalyses.length" class="asset-list">
          <button
            v-for="analysis in filteredAnalyses"
            :key="analysis.analysis_id"
            type="button"
            class="asset-item"
            :class="{ active: selectedAnalysisId === analysis.analysis_id }"
            :aria-pressed="selectedAnalysisId === analysis.analysis_id"
            @click="selectAnalysis(analysis.analysis_id)"
          >
            <span class="asset-status">
              <i aria-hidden="true"></i>
              {{ analysis.status || 'ready' }}
            </span>
            <strong>{{ analysisTitle(analysis) }}</strong>
            <span>{{ analysisSubtitle(analysis) }}</span>
            <small>{{ formatDuration(analysis.video?.duration_seconds) }} · {{ analysis.analysis_id }}</small>
          </button>
        </div>

        <div v-else class="rail-empty">
          <strong>No matches</strong>
          <span>Try a broader search term.</span>
        </div>
      </aside>

      <main id="analysis-workspace" class="workspace" tabindex="-1">
        <section v-if="!selectedAnalysisId" class="state-panel compact-state">
          <div>
            <p class="eyebrow">Ready</p>
            <h2>Select an analysis</h2>
            <p>Choose an asset to open synchronized playback and future-movement ranking.</p>
          </div>
        </section>

        <template v-else>
          <section class="playback-panel" aria-labelledby="playback-heading">
            <div class="section-heading playback-heading">
              <div>
                <p class="eyebrow">Selected analysis</p>
                <h2 id="playback-heading">{{ selectedAnalysisTitle }}</h2>
                <p class="analysis-id">{{ selectedAnalysisId }}</p>
              </div>
              <div class="contract-badges">
            <span>{{ timeline?.schema_version || 'neural_bridge.timeline.v1' }}</span>
                <span>+2s to +5s target</span>
              </div>
            </div>

            <div class="video-shell" :class="{ loading: timelineLoading || !videoReady }">
              <video
                :key="selectedAnalysisId"
                ref="stimulusVideo"
                :src="mediaUrl"
                controls
                playsinline
                preload="metadata"
                :aria-label="`Stimulus playback for ${selectedAnalysisTitle}`"
                @loadedmetadata="onVideoLoaded"
                @canplay="onVideoCanPlay"
                @timeupdate="syncFromVideo"
                @seeked="syncFromVideo"
                @play="onVideoPlay"
                @pause="onVideoPause"
                @ended="onVideoPause"
                @error="onVideoError"
              >
                Your browser does not support HTML video playback.
              </video>

              <div v-if="timelineLoading" class="video-overlay" aria-busy="true">
                <span class="loading-orbit" aria-hidden="true"></span>
                <strong>Loading synchronized predictions</strong>
              </div>
              <div v-else-if="videoError" class="video-overlay error-overlay" role="alert">
                <strong>Media unavailable</strong>
                <span>{{ videoError }}</span>
              </div>
            </div>

            <div class="transport" aria-label="Playback controls">
              <button type="button" class="primary-control" :disabled="!videoReady" @click="togglePlayback">
                {{ playing ? 'Pause' : 'Play' }}
              </button>
              <button type="button" :disabled="!timestamps.length" @click="stepIndex(-1)">Previous row</button>
              <button type="button" :disabled="!timestamps.length" @click="stepIndex(1)">Next row</button>
              <span class="transport-time">{{ formatTime(currentTime) }} / {{ formatTime(mediaDuration) }}</span>
              <label class="rate-control">
                <span>Playback rate</span>
                <select v-model.number="playbackRate" :disabled="!videoReady" @change="applyPlaybackRate">
                  <option :value="0.5">0.5×</option>
                  <option :value="0.75">0.75×</option>
                  <option :value="1">1×</option>
                  <option :value="1.25">1.25×</option>
                  <option :value="1.5">1.5×</option>
                  <option :value="2">2×</option>
                </select>
              </label>
            </div>

            <p class="keyboard-hint">
              Keyboard: <kbd>Space</kbd> play/pause, <kbd>←</kbd>/<kbd>→</kbd> step prediction rows.
            </p>
          </section>

          <section v-if="timelineError" class="state-panel error-state compact-state" role="alert">
            <div>
              <p class="eyebrow">Timeline unavailable</p>
              <h2>Synchronized predictions could not be loaded</h2>
              <p>{{ timelineError }}</p>
            </div>
            <button type="button" @click="loadTimeline(selectedAnalysisId)">Retry</button>
          </section>

          <template v-else-if="timeline">
            <section class="moment-strip" aria-label="Current prediction row">
              <article class="moment-card primary-moment">
                <span>Future-movement rank</span>
                <strong>{{ formatPercentile(currentPoint.percentile) }}</strong>
                <small>Within this video, not an exact arousal level</small>
              </article>
              <article class="moment-card">
                <span>Model score</span>
                <strong>{{ formatScore(currentPoint.score) }}</strong>
                <small>Ranking signal at {{ formatTime(currentPoint.timestamp) }}</small>
              </article>
              <article class="moment-card">
                <span>Forecast window</span>
                <strong>{{ formatTime(forecastWindow.start) }}–{{ formatTime(forecastWindow.end) }}</strong>
                <small>Current row +2s through +5s</small>
              </article>
              <article class="moment-card quality-card" :class="{ warning: currentPoint.coldStart || currentPoint.tailHorizon }">
                <span>Row quality</span>
                <strong>{{ currentQualityLabel }}</strong>
                <small>{{ currentQualityDetail }}</small>
              </article>
            </section>

            <div v-if="currentPoint.tailHorizon" class="tail-warning" role="status">
              <strong>Tail-horizon warning.</strong>
              This row’s +5s target extends beyond available media; treat the ranking as incomplete.
            </div>

            <section class="timeline-panel" aria-labelledby="timeline-heading">
              <div class="section-heading timeline-heading">
                <div>
                  <p class="eyebrow">Playback-synchronized output</p>
                  <h2 id="timeline-heading">Ranked future arousal movement</h2>
                </div>
                <div class="timeline-legend" aria-label="Timeline legend">
                  <span><i class="legend-line"></i> Within-video percentile</span>
                  <span><i class="legend-band"></i> Member-rank spread</span>
                  <span><i class="legend-event"></i> Provisional event</span>
                </div>
              </div>

              <p class="chart-note">
                Higher percentiles identify stronger relative future movement within this asset. The orange member band is diagnostic only and is <strong>not calibrated uncertainty</strong>.
              </p>

              <div class="chart-wrap">
                <svg
                  class="rank-chart"
                  viewBox="0 0 1000 280"
                  role="img"
                  aria-labelledby="rank-chart-title rank-chart-desc"
                  @click="seekFromChart"
                >
                  <title id="rank-chart-title">Within-video future-movement percentile over playback time</title>
                  <desc id="rank-chart-desc">
                    A synchronized percentile line with upstream cold-start shading from zero to four seconds, provisional event markers, the active plus-two-to-plus-five-second forecast window, a tail-horizon warning region, and diagnostic member-rank spread.
                  </desc>
                  <defs>
                    <linearGradient id="rankFill" x1="0" y1="0" x2="0" y2="1">
                      <stop offset="0%" stop-color="#ffb347" stop-opacity="0.34" />
                      <stop offset="100%" stop-color="#ff7130" stop-opacity="0.02" />
                    </linearGradient>
                    <pattern id="coldHatch" width="9" height="9" patternUnits="userSpaceOnUse" patternTransform="rotate(35)">
                      <rect width="9" height="9" fill="#0a2323" fill-opacity="0.85" />
                      <line x1="0" y1="0" x2="0" y2="9" stroke="#59d7ca" stroke-opacity="0.3" stroke-width="3" />
                    </pattern>
                    <pattern id="tailHatch" width="9" height="9" patternUnits="userSpaceOnUse" patternTransform="rotate(-35)">
                      <rect width="9" height="9" fill="#2b1510" fill-opacity="0.8" />
                      <line x1="0" y1="0" x2="0" y2="9" stroke="#ff8154" stroke-opacity="0.32" stroke-width="3" />
                    </pattern>
                  </defs>

                  <g class="chart-grid" aria-hidden="true">
                    <template v-for="tick in percentileTicks" :key="tick">
                      <line :x1="chart.left" :x2="chart.right" :y1="chartY(tick)" :y2="chartY(tick)" />
                      <text :x="chart.left - 10" :y="chartY(tick) + 4">{{ tick }}</text>
                    </template>
                  </g>

                  <rect
                    class="cold-region"
                    :x="chartX(chartStart)"
                    :y="chart.top"
                    :width="Math.max(0, chartX(Math.min(4, chartEnd)) - chartX(chartStart))"
                    :height="chart.plotHeight"
                    fill="url(#coldHatch)"
                  />
                  <text v-if="chartEnd > 1" class="region-label" :x="chartX(chartStart) + 8" :y="chart.top + 16">0–4s upstream cold start</text>

                  <rect
                    class="tail-region"
                    :x="chartX(tailRegionStart)"
                    :y="chart.top"
                    :width="Math.max(0, chartX(chartEnd) - chartX(tailRegionStart))"
                    :height="chart.plotHeight"
                    fill="url(#tailHatch)"
                  />

                  <rect
                    class="forecast-region"
                    :x="chartX(clampedForecastWindow.start)"
                    :y="chart.top"
                    :width="Math.max(2, chartX(clampedForecastWindow.end) - chartX(clampedForecastWindow.start))"
                    :height="chart.plotHeight"
                  />

                  <path v-if="memberBandPath" class="member-band" :d="memberBandPath" />
                  <path v-if="percentileAreaPath" class="percentile-area" :d="percentileAreaPath" />
                  <path v-if="percentileLinePath" class="percentile-line" :d="percentileLinePath" />

                  <g class="event-markers" aria-hidden="true">
                    <g v-for="event in eventItems" :key="event.id">
                      <line :x1="chartX(event.timestamp)" :x2="chartX(event.timestamp)" :y1="chart.top" :y2="chart.bottom" />
                      <path :d="eventTriangle(event.timestamp)" />
                    </g>
                  </g>

                  <line
                    class="playhead"
                    :x1="chartX(currentPoint.timestamp)"
                    :x2="chartX(currentPoint.timestamp)"
                    :y1="chart.top - 4"
                    :y2="chart.bottom + 7"
                  />
                  <circle
                    class="current-dot"
                    :cx="chartX(currentPoint.timestamp)"
                    :cy="chartY(currentPoint.percentile)"
                    r="6"
                  />

                  <g class="time-axis" aria-hidden="true">
                    <template v-for="tick in timeTicks" :key="tick">
                      <line :x1="chartX(tick)" :x2="chartX(tick)" :y1="chart.bottom" :y2="chart.bottom + 6" />
                      <text :x="chartX(tick)" :y="chart.bottom + 24">{{ formatTime(tick) }}</text>
                    </template>
                  </g>
                </svg>
              </div>

              <label class="timeline-scrubber">
                <span>Prediction row {{ currentIndex + 1 }} of {{ timestamps.length }}</span>
                <input
                  type="range"
                  min="0"
                  :max="Math.max(0, timestamps.length - 1)"
                  step="1"
                  :value="currentIndex"
                  :disabled="!timestamps.length"
                  @input="scrubToIndex"
                />
                <output>{{ formatTime(currentPoint.timestamp) }}</output>
              </label>

              <div v-if="eventItems.length" class="event-list" aria-label="Provisional event markers">
                <span class="event-policy">{{ eventPolicyText }}</span>
                <button
                  v-for="event in eventItems"
                  :key="`button-${event.id}`"
                  type="button"
                  @click="seekToTime(event.timestamp)"
                >
                  Provisional · {{ formatTime(event.timestamp) }}
                </button>
              </div>
            </section>

            <section class="evidence-panel" aria-labelledby="evidence-heading">
              <div class="section-heading">
                <div>
                  <p class="eyebrow">Evidence boundary</p>
                  <h2 id="evidence-heading">Reference evidence is not run validation</h2>
                </div>
              </div>

              <div class="evidence-grid">
                <article>
                  <span class="scope-label validated">Model-validation reference</span>
                  <h3>{{ evidenceTitle(modelValidationEvidence, 'Locked model evidence') }}</h3>
                  <p>{{ evidenceSummary(modelValidationEvidence, 'The model-level reference describes validation performed elsewhere under its stored protocol.') }}</p>
                </article>
                <article>
                  <span class="scope-label unvalidated">This analysis run · unvalidated</span>
                  <h3>{{ evidenceTitle(runLevelEvidence, 'Run-level inference output') }}</h3>
                  <p>{{ evidenceSummary(runLevelEvidence, 'This individual asset is an inference result, not a new held-out validation or client-outcome study.') }}</p>
                </article>
                <article>
                  <span class="scope-label implementation">Implementation</span>
                  <h3>{{ evidenceTitle(implementationEvidence, 'Runtime provenance') }}</h3>
                  <p>{{ evidenceSummary(implementationEvidence, 'Implementation metadata identifies the producing pipeline; it does not expand the validated claim.') }}</p>
                </article>
              </div>
            </section>

            <section class="unsupported-panel" aria-labelledby="unsupported-heading">
              <div>
                <p class="eyebrow">Unsupported outputs</p>
                <h2 id="unsupported-heading">Do not infer beyond the contract</h2>
              </div>
              <ul>
                <li v-for="item in unsupportedOutputItems" :key="item.key">
                  <strong>{{ item.label }}</strong>
                  <span>Unsupported</span>
                  <small>{{ item.reason }}</small>
                </li>
              </ul>
            </section>
          </template>
        </template>
      </main>

      <aside v-if="timeline" class="inspector" aria-labelledby="inspector-heading">
        <div class="inspector-heading">
          <p class="eyebrow">Moment inspector</p>
          <h2 id="inspector-heading">{{ formatTime(currentPoint.timestamp) }}</h2>
        </div>

        <section class="inspector-section current-diagnostic">
          <h3>Member diagnostic</h3>
          <template v-if="currentMemberDiagnostic.available">
            <div class="diagnostic-value">
              <span>{{ formatScore(currentMemberDiagnostic.min) }}</span>
              <i aria-hidden="true"></i>
              <span>{{ formatScore(currentMemberDiagnostic.max) }}</span>
            </div>
            <p>Member score range {{ formatScore(currentMemberDiagnostic.spread) }}.</p>
          </template>
          <p v-else>No member diagnostics were returned.</p>
          <small>Diagnostic only · not calibrated uncertainty.</small>
        </section>

        <section class="inspector-section">
          <h3>Top ranked moments</h3>
          <ol v-if="topMoments.length" class="top-moments">
            <li v-for="moment in topMoments" :key="moment.index">
              <button type="button" @click="seekToIndex(moment.index)">
                <span>
                  <strong>{{ formatTime(moment.timestamp) }}</strong>
                  <small>Forecast {{ formatTime(moment.timestamp + 2) }}–{{ formatTime(moment.timestamp + 5) }}</small>
                </span>
                <span class="moment-rank">{{ formatPercentile(moment.percentile) }}</span>
                <span v-if="moment.coldStart" class="flag cold-flag">Cold start · provisional</span>
                <span v-if="moment.tailHorizon" class="flag tail-flag">Tail horizon</span>
              </button>
            </li>
          </ol>
          <p v-else>No ranked moments were returned.</p>
        </section>

        <section class="inspector-section target-contract">
          <h3>Target contract</h3>
          <dl>
            <div>
              <dt>Output</dt>
              <dd>{{ targetLabel }}</dd>
            </div>
            <div>
              <dt>Forecast</dt>
              <dd>+2s through +5s</dd>
            </div>
            <div>
              <dt>Ranking</dt>
              <dd>Within-video percentile</dd>
            </div>
            <div>
              <dt>Cold start</dt>
              <dd>0–4.0s upstream context</dd>
            </div>
          </dl>
        </section>
      </aside>
    </div>
  </div>
</template>

<script setup>
import { computed, nextTick, onMounted, onUnmounted, ref } from 'vue'
import { RouterLink } from 'vue-router'
import {
  getNeuralBridgeTimeline,
  listNeuralBridgeAnalyses,
  neuralBridgeJsonReportUrl,
  neuralBridgeMediaUrl,
  neuralBridgePredictionsCsvUrl
} from '../api/neuralBridge'

const FORECAST_MIN_SECONDS = 2
const FORECAST_MAX_SECONDS = 5
const UPSTREAM_COLD_START_SECONDS = 4

const chart = Object.freeze({
  width: 1000,
  height: 280,
  left: 52,
  right: 982,
  top: 24,
  bottom: 240,
  plotWidth: 930,
  plotHeight: 216
})

const percentileTicks = [0, 25, 50, 75, 100]

const analyses = ref([])
const inventoryLoading = ref(true)
const inventoryError = ref('')
const searchQuery = ref('')
const selectedAnalysisId = ref('')
const timeline = ref(null)
const timelineLoading = ref(false)
const timelineError = ref('')
const stimulusVideo = ref(null)
const videoReady = ref(false)
const videoError = ref('')
const playbackNotice = ref('')
const currentIndex = ref(0)
const currentTime = ref(0)
const mediaDurationFromVideo = ref(0)
const playing = ref(false)
const playbackRate = ref(1)

let timelineRequest = null
let timelineRequestId = 0
let videoFrameId = null
let animationFrameId = null

const unwrapResponse = response => response?.data ?? response

const filteredAnalyses = computed(() => {
  const query = searchQuery.value.toLowerCase()
  if (!query) return analyses.value
  return analyses.value.filter(analysis => JSON.stringify(analysis).toLowerCase().includes(query))
})

const selectedAnalysis = computed(() => {
  return analyses.value.find(analysis => analysis.analysis_id === selectedAnalysisId.value) || null
})

const selectedAnalysisTitle = computed(() => {
  return selectedAnalysis.value ? analysisTitle(selectedAnalysis.value) : 'Analysis'
})

const mediaUrl = computed(() => {
  return selectedAnalysisId.value ? neuralBridgeMediaUrl(selectedAnalysisId.value) : ''
})

const jsonReportUrl = computed(() => {
  return selectedAnalysisId.value ? neuralBridgeJsonReportUrl(selectedAnalysisId.value) : ''
})

const predictionsCsvUrl = computed(() => {
  return selectedAnalysisId.value ? neuralBridgePredictionsCsvUrl(selectedAnalysisId.value) : ''
})

const timestamps = computed(() => {
  const values = timeline.value?.grid?.timestamps_seconds
  return Array.isArray(values) ? values.map(Number).filter(Number.isFinite) : []
})

const movementScores = computed(() => {
  return numericSeries(timeline.value?.series?.future_arousal_movement_score?.values)
})

const percentileValues = computed(() => {
  return numericSeries(timeline.value?.series?.within_video_percentile?.values).map(value => {
    const percent = value <= 1 ? value * 100 : value
    return clamp(percent, 0, 100)
  })
})

const memberSeries = computed(() => {
  const members = timeline.value?.diagnostics?.member_scores
  if (!members || typeof members !== 'object' || Array.isArray(members)) return []
  return Object.entries(members)
    .map(([memberId, values]) => ({
      id: memberId,
      values: numericSeries(values)
    }))
    .filter(member => member.values.length)
})

const memberRankBand = computed(() => {
  if (!memberSeries.value.length || !timestamps.value.length) return []
  const ranks = memberSeries.value.map(member => rankAsPercentiles(member.values))
  return timestamps.value.map((timestamp, index) => {
    const values = ranks.map(series => series[index]).filter(Number.isFinite)
    return {
      timestamp,
      min: values.length ? Math.min(...values) : 0,
      max: values.length ? Math.max(...values) : 0
    }
  })
})

const chartStart = computed(() => 0)
const chartEnd = computed(() => {
  const lastTimestamp = timestamps.value[timestamps.value.length - 1] ?? 0
  return Math.max(lastTimestamp, mediaDuration.value, chartStart.value + 1)
})

const mediaDuration = computed(() => {
  const fromAnalysis = Number(selectedAnalysis.value?.video?.duration_seconds)
  if (Number.isFinite(mediaDurationFromVideo.value) && mediaDurationFromVideo.value > 0) return mediaDurationFromVideo.value
  if (Number.isFinite(fromAnalysis) && fromAnalysis > 0) return fromAnalysis
  return timestamps.value[timestamps.value.length - 1] ?? 0
})

const forecastWindow = computed(() => ({
  start: currentPoint.value.timestamp + FORECAST_MIN_SECONDS,
  end: currentPoint.value.timestamp + FORECAST_MAX_SECONDS
}))

const clampedForecastWindow = computed(() => ({
  start: clamp(forecastWindow.value.start, chartStart.value, chartEnd.value),
  end: clamp(forecastWindow.value.end, chartStart.value, chartEnd.value)
}))

const tailRegionStart = computed(() => Math.max(chartStart.value, chartEnd.value - FORECAST_MAX_SECONDS))

const fullUpstreamWindowContext = computed(() => {
  return qualityArray('full_upstream_window_context')
})

const fullForecastWindowInMedia = computed(() => {
  return qualityArray('full_forecast_window_in_media')
})

const currentPoint = computed(() => {
  const index = clamp(currentIndex.value, 0, Math.max(0, timestamps.value.length - 1))
  const timestamp = timestamps.value[index] ?? 0
  return {
    index,
    timestamp,
    score: movementScores.value[index],
    percentile: percentileValues.value[index] ?? 0,
    coldStart: isColdStart(index, timestamp),
    tailHorizon: isTailHorizon(index, timestamp)
  }
})

const currentQualityLabel = computed(() => {
  if (currentPoint.value.coldStart) return 'Cold start · provisional'
  if (currentPoint.value.tailHorizon) return 'Incomplete horizon'
  return 'Full context'
})

const currentQualityDetail = computed(() => {
  if (currentPoint.value.coldStart) return 'Upstream V-JEPA window is not yet fully populated.'
  if (currentPoint.value.tailHorizon) return 'The target horizon extends past the media tail.'
  return 'Full upstream context and forecast horizon are available.'
})

const currentMemberDiagnostic = computed(() => {
  const values = memberSeries.value
    .map(member => member.values[currentPoint.value.index])
    .filter(Number.isFinite)
  if (!values.length) return { available: false, min: 0, max: 0, spread: 0 }
  const min = Math.min(...values)
  const max = Math.max(...values)
  return { available: true, min, max, spread: max - min }
})

const percentileLinePath = computed(() => {
  return linePath(percentileValues.value, value => chartY(value))
})

const percentileAreaPath = computed(() => {
  if (!percentileLinePath.value) return ''
  const lastIndex = Math.min(percentileValues.value.length, timestamps.value.length) - 1
  if (lastIndex < 0) return ''
  return `${percentileLinePath.value} L ${chartX(timestamps.value[lastIndex])} ${chart.bottom} L ${chartX(timestamps.value[0])} ${chart.bottom} Z`
})

const memberBandPath = computed(() => {
  if (!memberRankBand.value.length) return ''
  const upper = memberRankBand.value
    .map((point, index) => `${index ? 'L' : 'M'} ${chartX(point.timestamp)} ${chartY(point.max)}`)
    .join(' ')
  const lower = [...memberRankBand.value]
    .reverse()
    .map(point => `L ${chartX(point.timestamp)} ${chartY(point.min)}`)
    .join(' ')
  return `${upper} ${lower} Z`
})

const eventItems = computed(() => {
  const items = timeline.value?.events?.items
  if (!Array.isArray(items)) return []
  return items
    .map((item, index) => {
      const itemIndex = Number(item?.row_index)
      const timestamp = Number.isFinite(Number(item?.anchor_time_seconds))
        ? Number(item.anchor_time_seconds)
        : timestamps.value[itemIndex]
      return {
        ...item,
        id: item?.event_id || `${index}-${timestamp}`,
        timestamp: Number(timestamp)
      }
    })
    .filter(item => Number.isFinite(item.timestamp))
})

const eventPolicyText = computed(() => {
  const policy = timeline.value?.events?.policy
  if (typeof policy === 'string') return policy
  if (policy?.provisional) {
    const threshold = Number(policy.percentile)
    const label = Number.isFinite(threshold) ? `top ${((1 - threshold) * 100).toFixed(0)}%` : 'high-rank'
    return `Provisional ${label} within-video markers; not observed outcomes.`
  }
  return 'Provisional markers from the stored event policy; not observed outcomes.'
})

const topMoments = computed(() => {
  const candidates = percentileValues.value
    .map((percentile, index) => ({
      index,
      percentile,
      timestamp: timestamps.value[index]
    }))
    .filter(item => Number.isFinite(item.percentile) && Number.isFinite(item.timestamp))
    .sort((a, b) => b.percentile - a.percentile)

  const selected = []
  for (const candidate of candidates) {
    if (selected.some(item => Math.abs(item.timestamp - candidate.timestamp) < FORECAST_MIN_SECONDS)) continue
    selected.push({
      ...candidate,
      coldStart: isColdStart(candidate.index, candidate.timestamp),
      tailHorizon: isTailHorizon(candidate.index, candidate.timestamp)
    })
    if (selected.length === 5) break
  }
  return selected
})

const timeTicks = computed(() => {
  const count = 5
  const span = chartEnd.value - chartStart.value
  return Array.from({ length: count }, (_, index) => chartStart.value + (span * index) / (count - 1))
})

const targetLabel = computed(() => {
  const target = timeline.value?.target
  if (typeof target === 'string') return target
  return target?.id || 'Future arousal movement ranking'
})

const modelValidationEvidence = computed(() => timeline.value?.evidence_scopes?.model_validation_reference)
const runLevelEvidence = computed(() => timeline.value?.evidence_scopes?.run_level_validation)
const implementationEvidence = computed(() => timeline.value?.evidence_scopes?.implementation_reproduction)

const unsupportedOutputItems = computed(() => {
  const outputs = timeline.value?.unsupported_outputs || {}
  return [
    {
      key: 'arousal_dropoff',
      label: 'Drop-off prediction',
      reason: unsupportedReason(outputs, 'arousal_dropoff', 'No validated drop-off output is produced by this contract.')
    },
    {
      key: 'valence',
      label: 'Valence',
      reason: unsupportedReason(outputs, 'valence', 'Positive-versus-negative emotional valence is not inferred.')
    },
    {
      key: 'exact_arousal_level',
      label: 'Exact arousal level',
      reason: unsupportedReason(outputs, 'exact_arousal_level', 'The output is a relative movement ranking, not a calibrated exact level.')
    }
  ]
})

const liveStatus = computed(() => {
  if (inventoryLoading.value) return 'Loading analysis inventory.'
  if (timelineLoading.value) return `Loading analysis ${selectedAnalysisId.value}.`
  if (timelineError.value) return 'Timeline loading failed.'
  if (playbackNotice.value) return playbackNotice.value
  if (timeline.value) return `Analysis ${selectedAnalysisId.value} ready at ${formatTime(currentPoint.value.timestamp)}.`
  return `${analyses.value.length} analyses available.`
})

function clamp(value, minimum, maximum) {
  return Math.min(maximum, Math.max(minimum, Number(value) || 0))
}

function numericSeries(values) {
  if (!Array.isArray(values)) return []
  return values.map(value => Number(value))
}

function analysisTitle(analysis) {
  return analysis.video?.display_name || analysis.analysis_id
}

function analysisSubtitle(analysis) {
  const dataset = analysis.source?.dataset_id || 'Unknown dataset'
  const modalities = Array.isArray(analysis.inference?.modalities_used)
    ? analysis.inference.modalities_used.join(' + ')
    : 'modalities not recorded'
  return `${dataset} · ${modalities}`
}

function formatDuration(value) {
  const seconds = Number(value)
  return Number.isFinite(seconds) ? formatTime(seconds) : 'duration unknown'
}

function formatTime(value) {
  const seconds = Math.max(0, Number(value) || 0)
  const minutes = Math.floor(seconds / 60)
  const remainder = seconds - minutes * 60
  return `${minutes}:${remainder.toFixed(1).padStart(4, '0')}`
}

function formatPercentile(value) {
  const number = Number(value)
  return Number.isFinite(number) ? `P${number.toFixed(1)}` : 'n/a'
}

function formatScore(value) {
  const number = Number(value)
  if (!Number.isFinite(number)) return 'n/a'
  return `${number >= 0 ? '+' : ''}${number.toFixed(4)}`
}

function errorMessage(error, fallback) {
  return error?.response?.data?.error?.message || error?.message || fallback
}

function qualityArray(key) {
  const value = timeline.value?.row_quality?.[key]
  if (Array.isArray(value)) return value
  return Array.isArray(value?.values) ? value.values : []
}

function isColdStart(index, timestamp) {
  if (Number(timestamp) < UPSTREAM_COLD_START_SECONDS) return true
  const quality = fullUpstreamWindowContext.value
  return quality.length > index ? quality[index] !== true : false
}

function isTailHorizon(_index, timestamp) {
  const index = Number(_index)
  const quality = fullForecastWindowInMedia.value
  if (quality.length > index) return quality[index] !== true
  return Number(timestamp) + FORECAST_MAX_SECONDS > chartEnd.value + 1e-6
}

function rankAsPercentiles(values) {
  const finite = values.filter(Number.isFinite).sort((a, b) => a - b)
  if (finite.length < 2) return values.map(() => 50)
  return values.map(value => {
    if (!Number.isFinite(value)) return NaN
    let low = 0
    let high = finite.length
    while (low < high) {
      const middle = Math.floor((low + high) / 2)
      if (finite[middle] <= value) low = middle + 1
      else high = middle
    }
    return ((low - 1) / (finite.length - 1)) * 100
  })
}

function chartX(timestamp) {
  const span = Math.max(0.001, chartEnd.value - chartStart.value)
  const ratio = clamp((Number(timestamp) - chartStart.value) / span, 0, 1)
  return chart.left + ratio * chart.plotWidth
}

function chartY(percentile) {
  const value = clamp(percentile, 0, 100)
  return chart.top + (1 - value / 100) * chart.plotHeight
}

function linePath(values, yForValue) {
  const points = []
  const count = Math.min(values.length, timestamps.value.length)
  for (let index = 0; index < count; index += 1) {
    const value = Number(values[index])
    if (!Number.isFinite(value)) continue
    points.push(`${points.length ? 'L' : 'M'} ${chartX(timestamps.value[index])} ${yForValue(value)}`)
  }
  return points.join(' ')
}

function eventTriangle(timestamp) {
  const x = chartX(timestamp)
  return `M ${x - 5} ${chart.top - 2} L ${x + 5} ${chart.top - 2} L ${x} ${chart.top + 7} Z`
}

function evidenceTitle(scope, fallback) {
  if (!scope) return fallback
  if (typeof scope === 'string') return fallback
  return scope.evidence_id || scope.validation_status || scope.status || fallback
}

function evidenceSummary(scope, fallback) {
  if (!scope) return fallback
  if (typeof scope === 'string') return scope
  if (scope.applicability_note) return scope.applicability_note
  if (scope.evidence_scope === 'this_analysis') {
    return `External validity: ${scope.external_validity || 'not recorded'}. Controls run: ${scope.controls_run ? 'yes' : 'no'}.`
  }
  if (scope.validates) {
    return `${scope.validates}. This does not validate external predictive validity or new-video correctness.`
  }
  return fallback
}

function unsupportedReason(outputs, key, fallback) {
  const value = Array.isArray(outputs)
    ? outputs.find(item => item?.key === key)
    : outputs?.[key]
  if (typeof value === 'string') return value
  return value?.reason || value?.description || fallback
}

async function loadAnalyses() {
  inventoryLoading.value = true
  inventoryError.value = ''
  try {
    const items = []
    let cursor = null
    do {
      const response = await listNeuralBridgeAnalyses({ params: cursor ? { cursor } : {} })
      const payload = unwrapResponse(response)
      if (!Array.isArray(payload?.items)) throw new Error('The analyses endpoint did not return an items array.')
      items.push(...payload.items)
      cursor = payload.next_cursor || null
    } while (cursor)
    analyses.value = items.filter(item => item?.analysis_id)
    if (selectedAnalysisId.value && analyses.value.some(item => item.analysis_id === selectedAnalysisId.value)) return
    const first = analyses.value[0]
    if (first) await selectAnalysis(first.analysis_id)
  } catch (error) {
    inventoryError.value = errorMessage(error, 'Unknown inventory error.')
  } finally {
    inventoryLoading.value = false
  }
}

async function selectAnalysis(analysisId) {
  if (!analysisId) return
  stopVideoClock()
  stimulusVideo.value?.pause()
  selectedAnalysisId.value = String(analysisId)
  currentIndex.value = 0
  currentTime.value = 0
  mediaDurationFromVideo.value = 0
  videoReady.value = false
  videoError.value = ''
  playbackNotice.value = ''
  await loadTimeline(selectedAnalysisId.value)
}

async function loadTimeline(analysisId) {
  timelineRequest?.abort()
  timelineRequest = new AbortController()
  const requestId = ++timelineRequestId
  timelineLoading.value = true
  timelineError.value = ''
  timeline.value = null
  try {
    const response = await getNeuralBridgeTimeline(analysisId, { signal: timelineRequest.signal })
    if (requestId !== timelineRequestId) return
    const payload = unwrapResponse(response)
    const grid = payload?.grid?.timestamps_seconds
    const percentile = payload?.series?.within_video_percentile?.values
    const score = payload?.series?.future_arousal_movement_score?.values
    if (!Array.isArray(grid) || !Array.isArray(percentile) || !Array.isArray(score)) {
      throw new Error('Timeline does not satisfy the neural_bridge.timeline.v1 series contract.')
    }
    timeline.value = {
      ...payload,
      schema_version: response?.schema_version || 'neural_bridge.timeline.v1'
    }
    await nextTick()
    stimulusVideo.value?.load()
  } catch (error) {
    if (error?.name === 'CanceledError' || error?.name === 'AbortError') return
    timelineError.value = errorMessage(error, 'Unknown timeline error.')
  } finally {
    if (requestId === timelineRequestId) timelineLoading.value = false
  }
}

function onVideoLoaded() {
  const video = stimulusVideo.value
  if (!video) return
  mediaDurationFromVideo.value = Number(video.duration) || 0
  video.playbackRate = playbackRate.value
  syncFromVideo()
}

function onVideoCanPlay() {
  videoReady.value = true
  videoError.value = ''
  applyPlaybackRate()
}

function onVideoError() {
  videoReady.value = false
  videoError.value = 'The analysis media endpoint could not provide playable video.'
  stopVideoClock()
}

function applyPlaybackRate() {
  if (stimulusVideo.value) stimulusVideo.value.playbackRate = Number(playbackRate.value)
}

function togglePlayback() {
  const video = stimulusVideo.value
  if (!video || !videoReady.value) return
  if (video.paused) {
    video.play().catch(error => {
      playing.value = false
      playbackNotice.value = error?.name === 'NotAllowedError'
        ? 'Playback was blocked until a direct media gesture; use the native video control to start.'
        : 'Playback could not start. The media source remains available.'
    })
  } else {
    video.pause()
  }
}

function onVideoPlay() {
  playing.value = true
  playbackNotice.value = ''
  startVideoClock()
}

function onVideoPause() {
  playing.value = false
  syncFromVideo()
  stopVideoClock()
}

function startVideoClock() {
  stopVideoClock()
  const video = stimulusVideo.value
  if (!video || video.paused) return

  if (typeof video.requestVideoFrameCallback === 'function') {
    const tick = (_now, metadata) => {
      syncToTime(metadata?.mediaTime ?? video.currentTime)
      if (!video.paused) videoFrameId = video.requestVideoFrameCallback(tick)
    }
    videoFrameId = video.requestVideoFrameCallback(tick)
    return
  }

  const tick = () => {
    syncFromVideo()
    if (!video.paused) animationFrameId = requestAnimationFrame(tick)
  }
  animationFrameId = requestAnimationFrame(tick)
}

function stopVideoClock() {
  const video = stimulusVideo.value
  if (videoFrameId !== null && video && typeof video.cancelVideoFrameCallback === 'function') {
    video.cancelVideoFrameCallback(videoFrameId)
  }
  if (animationFrameId !== null) cancelAnimationFrame(animationFrameId)
  videoFrameId = null
  animationFrameId = null
}

function syncFromVideo() {
  if (!stimulusVideo.value) return
  syncToTime(stimulusVideo.value.currentTime)
}

function syncToTime(seconds) {
  currentTime.value = Number(seconds) || 0
  currentIndex.value = nearestTimestampIndex(currentTime.value)
}

function nearestTimestampIndex(seconds) {
  const values = timestamps.value
  if (!values.length) return 0
  if (seconds <= values[0]) return 0
  if (seconds >= values[values.length - 1]) return values.length - 1
  let low = 0
  let high = values.length - 1
  while (low <= high) {
    const middle = Math.floor((low + high) / 2)
    if (values[middle] < seconds) low = middle + 1
    else high = middle - 1
  }
  const before = Math.max(0, low - 1)
  const after = Math.min(values.length - 1, low)
  return Math.abs(values[before] - seconds) <= Math.abs(values[after] - seconds) ? before : after
}

function seekToIndex(index) {
  const safeIndex = clamp(index, 0, Math.max(0, timestamps.value.length - 1))
  seekToTime(timestamps.value[safeIndex] ?? 0)
}

function seekToTime(seconds) {
  const time = clamp(seconds, 0, Math.max(mediaDuration.value, chartEnd.value))
  currentTime.value = time
  currentIndex.value = nearestTimestampIndex(time)
  if (stimulusVideo.value && Number.isFinite(time)) stimulusVideo.value.currentTime = time
}

function stepIndex(direction) {
  seekToIndex(currentIndex.value + Number(direction))
}

function scrubToIndex(event) {
  seekToIndex(Number(event.target.value))
}

function seekFromChart(event) {
  const bounds = event.currentTarget.getBoundingClientRect()
  const viewX = ((event.clientX - bounds.left) / Math.max(1, bounds.width)) * chart.width
  const ratio = clamp((viewX - chart.left) / chart.plotWidth, 0, 1)
  seekToTime(chartStart.value + ratio * (chartEnd.value - chartStart.value))
}

function handleKeyboardShortcut(event) {
  if (event.defaultPrevented || event.metaKey || event.ctrlKey || event.altKey) return
  const target = event.target
  const tag = target?.tagName?.toLowerCase()
  if (target?.isContentEditable || ['input', 'select', 'textarea', 'button', 'a', 'video'].includes(tag)) return
  if (event.code === 'Space') {
    event.preventDefault()
    togglePlayback()
  } else if (event.key === 'ArrowLeft') {
    event.preventDefault()
    stepIndex(-1)
  } else if (event.key === 'ArrowRight') {
    event.preventDefault()
    stepIndex(1)
  }
}

onMounted(() => {
  window.addEventListener('keydown', handleKeyboardShortcut)
  loadAnalyses()
})

onUnmounted(() => {
  timelineRequest?.abort()
  stopVideoClock()
  window.removeEventListener('keydown', handleKeyboardShortcut)
})
</script>

<style scoped>
.analyst-page {
  --bg-deep: #030908;
  --bg-panel: rgba(7, 19, 18, 0.92);
  --bg-raised: rgba(16, 31, 29, 0.9);
  --border: rgba(220, 247, 235, 0.14);
  --border-strong: rgba(255, 190, 104, 0.42);
  --text: #edf8f2;
  --muted: #a9c9bd;
  --teal: #67dfc7;
  --teal-deep: #123c38;
  --orange: #ffad55;
  --orange-hot: #ff7130;
  --danger: #ff8d6e;
  min-height: 100vh;
  padding: 20px;
  color: var(--text);
  font-family: 'Space Grotesk', 'JetBrains Mono', sans-serif;
  background:
    radial-gradient(circle at 9% 3%, rgba(37, 198, 177, 0.16), transparent 24%),
    radial-gradient(circle at 92% 0%, rgba(255, 121, 47, 0.17), transparent 23%),
    linear-gradient(145deg, #030908 0%, #081311 48%, #160c08 100%);
}

.skip-link {
  position: fixed;
  z-index: 100;
  top: 10px;
  left: 10px;
  padding: 10px 14px;
  color: #07110f;
  background: #fff2d7;
  transform: translateY(-160%);
  transition: transform 120ms ease;
}

.skip-link:focus {
  transform: translateY(0);
}

.analyst-page :focus-visible {
  outline: 3px solid #fff0c8;
  outline-offset: 3px;
}

.app-header {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 20px;
  max-width: 1880px;
  margin: 0 auto 12px;
  padding: 14px 16px;
  border: 1px solid var(--border);
  background: rgba(4, 13, 12, 0.78);
  box-shadow: 0 18px 60px rgba(0, 0, 0, 0.28);
  backdrop-filter: blur(18px);
}

.brand-block,
.header-actions,
.rail-heading,
.section-heading,
.transport,
.timeline-legend,
.contract-badges {
  display: flex;
  align-items: center;
}

.brand-block {
  gap: 14px;
}

.brand-mark,
.action-link,
.mode-badge,
.contract-badges span {
  display: inline-flex;
  align-items: center;
  min-height: 38px;
  border: 1px solid var(--border-strong);
  color: #fff0d8;
  background: rgba(255, 122, 48, 0.1);
  padding: 8px 11px;
  font-weight: 800;
  font-size: 0.74rem;
  letter-spacing: 0.08em;
  text-decoration: none;
  text-transform: uppercase;
}

.mode-badge {
  border-color: rgba(103, 223, 199, 0.35);
  color: #aff6e8;
  background: rgba(35, 188, 164, 0.1);
}

.header-actions {
  justify-content: flex-end;
  gap: 8px;
  flex-wrap: wrap;
}

.eyebrow {
  margin: 0 0 4px;
  color: #83dbc7;
  font-size: 0.7rem;
  font-weight: 700;
  letter-spacing: 0.13em;
  text-transform: uppercase;
}

h1,
h2,
h3,
p {
  margin-top: 0;
}

h1 {
  margin-bottom: 0;
  font-size: clamp(1.35rem, 2.4vw, 2.15rem);
  line-height: 1;
}

h2 {
  margin-bottom: 0;
  font-size: clamp(1.12rem, 1.8vw, 1.55rem);
  line-height: 1.1;
}

h3 {
  margin-bottom: 8px;
  font-size: 0.95rem;
}

.live-status {
  width: 1px;
  height: 1px;
  margin: -1px;
  overflow: hidden;
  position: absolute;
  clip: rect(0, 0, 0, 0);
  white-space: nowrap;
}

.analyst-layout {
  display: grid;
  grid-template-columns: 250px minmax(560px, 1fr) 310px;
  gap: 14px;
  max-width: 1880px;
  margin: 0 auto;
  align-items: start;
}

.asset-rail,
.playback-panel,
.timeline-panel,
.evidence-panel,
.unsupported-panel,
.inspector,
.state-panel {
  border: 1px solid var(--border);
  background:
    linear-gradient(145deg, rgba(6, 18, 17, 0.94), rgba(18, 12, 9, 0.88)),
    radial-gradient(circle at top left, rgba(65, 230, 198, 0.06), transparent 42%);
  box-shadow: 0 20px 64px rgba(0, 0, 0, 0.3);
}

.asset-rail,
.inspector {
  position: sticky;
  top: 14px;
  max-height: calc(100vh - 28px);
  overflow: auto;
}

.asset-rail {
  padding: 14px;
}

.rail-heading,
.section-heading {
  justify-content: space-between;
  gap: 14px;
}

.quiet-button,
.transport button,
.event-list button,
.state-panel button {
  min-height: 40px;
  border: 1px solid rgba(255, 196, 112, 0.28);
  color: var(--text);
  background: rgba(255, 132, 66, 0.08);
  padding: 8px 12px;
  cursor: pointer;
}

button:disabled,
select:disabled {
  opacity: 0.48;
  cursor: not-allowed;
}

.search-field {
  display: grid;
  gap: 6px;
  margin: 16px 0 7px;
  color: var(--muted);
  font-size: 0.75rem;
}

.search-field input,
.rate-control select {
  min-height: 42px;
  border: 1px solid rgba(212, 241, 229, 0.2);
  border-radius: 0;
  color: var(--text);
  background: rgba(0, 0, 0, 0.27);
  padding: 9px 10px;
}

.asset-count {
  margin-bottom: 10px;
  color: #8eb6aa;
  font-size: 0.7rem;
}

.asset-list {
  display: grid;
  gap: 8px;
}

.asset-item {
  display: grid;
  gap: 5px;
  width: 100%;
  min-height: 104px;
  padding: 11px;
  border: 1px solid rgba(224, 247, 238, 0.1);
  color: var(--text);
  background: rgba(255, 255, 255, 0.025);
  text-align: left;
  cursor: pointer;
}

.asset-item:hover {
  border-color: rgba(103, 223, 199, 0.32);
  background: rgba(103, 223, 199, 0.06);
}

.asset-item.active {
  border-color: rgba(255, 179, 86, 0.62);
  background: linear-gradient(100deg, rgba(255, 113, 48, 0.16), rgba(65, 218, 197, 0.06));
  box-shadow: inset 3px 0 #ffae55;
}

.asset-item > span,
.asset-item small {
  color: var(--muted);
  font-size: 0.7rem;
  overflow-wrap: anywhere;
}

.asset-item strong {
  min-width: 0;
  overflow-wrap: anywhere;
  word-break: break-word;
}

.asset-status {
  display: inline-flex;
  align-items: center;
  gap: 6px;
  text-transform: uppercase;
  letter-spacing: 0.08em;
}

.asset-status i {
  width: 7px;
  height: 7px;
  border-radius: 50%;
  background: var(--teal);
  box-shadow: 0 0 10px rgba(103, 223, 199, 0.55);
}

.rail-empty {
  display: grid;
  gap: 5px;
  padding: 20px 10px;
  color: var(--muted);
  text-align: center;
}

.workspace {
  display: grid;
  gap: 14px;
  min-width: 0;
}

.playback-panel,
.timeline-panel,
.evidence-panel,
.unsupported-panel {
  padding: 16px;
}

.playback-heading {
  align-items: flex-start;
  margin-bottom: 13px;
}

.analysis-id {
  margin: 5px 0 0;
  color: var(--muted);
  font-family: 'JetBrains Mono', monospace;
  font-size: 0.7rem;
  overflow-wrap: anywhere;
}

.contract-badges {
  justify-content: flex-end;
  gap: 6px;
  flex-wrap: wrap;
}

.contract-badges span {
  min-height: 30px;
  padding: 5px 8px;
  font-size: 0.62rem;
}

.video-shell {
  position: relative;
  width: 100%;
  overflow: hidden;
  aspect-ratio: 16 / 9;
  border: 1px solid rgba(230, 250, 241, 0.12);
  background: #010403;
}

.video-shell video {
  display: block;
  width: 100%;
  height: 100%;
  object-fit: contain;
  background: #010403;
}

.video-overlay {
  position: absolute;
  inset: 0;
  display: grid;
  place-content: center;
  justify-items: center;
  gap: 12px;
  color: #dff9ee;
  background: rgba(2, 8, 7, 0.78);
  backdrop-filter: blur(8px);
}

.error-overlay {
  color: #ffd7cc;
}

.loading-orbit {
  display: inline-block;
  width: 34px;
  height: 34px;
  border: 3px solid rgba(103, 223, 199, 0.2);
  border-top-color: var(--teal);
  border-radius: 50%;
  animation: spin 780ms linear infinite;
}

@keyframes spin {
  to { transform: rotate(360deg); }
}

.transport {
  gap: 8px;
  margin-top: 11px;
  flex-wrap: wrap;
}

.transport .primary-control {
  border-color: rgba(255, 182, 89, 0.58);
  color: #17100a;
  background: linear-gradient(135deg, #ffc46f, #ff8d43);
  font-weight: 900;
}

.transport-time {
  margin-left: auto;
  color: #d7eee5;
  font-family: 'JetBrains Mono', monospace;
  font-size: 0.78rem;
}

.rate-control {
  display: flex;
  align-items: center;
  gap: 7px;
  color: var(--muted);
  font-size: 0.7rem;
}

.rate-control select {
  min-height: 40px;
}

.keyboard-hint {
  margin: 9px 0 0;
  color: #91b8ad;
  font-size: 0.68rem;
}

kbd {
  display: inline-block;
  min-width: 24px;
  padding: 2px 5px;
  border: 1px solid rgba(255, 255, 255, 0.2);
  color: #f5eee2;
  background: rgba(255, 255, 255, 0.06);
  text-align: center;
}

.moment-strip {
  display: grid;
  grid-template-columns: repeat(4, minmax(0, 1fr));
  gap: 9px;
}

.moment-card {
  min-width: 0;
  padding: 12px;
  border: 1px solid var(--border);
  color: var(--text);
  background: var(--bg-panel);
}

.moment-card > span,
.moment-card small {
  display: block;
  color: var(--muted);
  font-size: 0.68rem;
}

.moment-card > span {
  margin-bottom: 6px;
  letter-spacing: 0.08em;
  text-transform: uppercase;
}

.moment-card strong {
  display: block;
  margin-bottom: 5px;
  color: #fff2df;
  font-size: 1.25rem;
}

.primary-moment {
  border-color: rgba(255, 179, 86, 0.5);
  background: linear-gradient(145deg, rgba(255, 117, 44, 0.16), rgba(13, 35, 31, 0.92));
}

.quality-card.warning {
  border-color: rgba(255, 126, 83, 0.52);
  background: rgba(111, 40, 21, 0.18);
}

.tail-warning {
  padding: 11px 14px;
  border: 1px solid rgba(255, 126, 83, 0.48);
  color: #ffd8cb;
  background: rgba(104, 32, 17, 0.22);
  font-size: 0.82rem;
}

.timeline-heading {
  align-items: flex-start;
  margin-bottom: 8px;
}

.timeline-legend {
  justify-content: flex-end;
  gap: 12px;
  flex-wrap: wrap;
  color: var(--muted);
  font-size: 0.66rem;
}

.timeline-legend span {
  display: inline-flex;
  align-items: center;
  gap: 5px;
}

.timeline-legend i {
  display: inline-block;
  width: 22px;
}

.legend-line {
  height: 3px;
  background: #ffae55;
}

.legend-band {
  height: 9px;
  background: rgba(75, 221, 200, 0.25);
  border: 1px solid rgba(75, 221, 200, 0.5);
}

.legend-event {
  height: 11px;
  border-left: 2px dashed #ff8154;
}

.chart-note {
  margin-bottom: 8px;
  color: #b7d2c8;
  font-size: 0.75rem;
  line-height: 1.45;
}

.chart-wrap {
  overflow-x: auto;
  border: 1px solid rgba(255, 255, 255, 0.08);
  background: rgba(0, 5, 5, 0.42);
}

.rank-chart {
  display: block;
  width: 100%;
  min-width: 620px;
  height: auto;
  cursor: crosshair;
}

.chart-grid line {
  stroke: rgba(223, 247, 237, 0.1);
  stroke-width: 1;
}

.chart-grid text,
.time-axis text,
.region-label {
  fill: #98bdb2;
  font-family: 'JetBrains Mono', monospace;
  font-size: 11px;
}

.chart-grid text {
  text-anchor: end;
}

.time-axis line {
  stroke: rgba(223, 247, 237, 0.28);
}

.time-axis text {
  text-anchor: middle;
}

.region-label {
  fill: #b5eee3;
  font-size: 10px;
  letter-spacing: 0.05em;
  text-transform: uppercase;
}

.forecast-region {
  fill: rgba(255, 181, 86, 0.09);
  stroke: rgba(255, 181, 86, 0.5);
  stroke-width: 1;
  stroke-dasharray: 5 4;
}

.member-band {
  fill: rgba(82, 220, 199, 0.18);
  stroke: rgba(82, 220, 199, 0.34);
  stroke-width: 1;
}

.percentile-area {
  fill: url(#rankFill);
}

.percentile-line {
  fill: none;
  stroke: #ffb25b;
  stroke-width: 3;
  stroke-linejoin: round;
  stroke-linecap: round;
  filter: drop-shadow(0 0 5px rgba(255, 141, 64, 0.36));
}

.event-markers line {
  stroke: #ff8154;
  stroke-width: 1.5;
  stroke-dasharray: 5 4;
}

.event-markers path {
  fill: #ff8154;
}

.playhead {
  stroke: #f4fff9;
  stroke-width: 2;
  filter: drop-shadow(0 0 4px rgba(255, 255, 255, 0.7));
}

.current-dot {
  fill: #fff8df;
  stroke: #ff7a35;
  stroke-width: 3;
}

.timeline-scrubber {
  display: grid;
  grid-template-columns: auto minmax(120px, 1fr) auto;
  gap: 12px;
  align-items: center;
  margin-top: 11px;
  color: var(--muted);
  font-size: 0.72rem;
}

.timeline-scrubber input {
  width: 100%;
  accent-color: #ff9b4a;
}

.timeline-scrubber output {
  color: #fff0d8;
  font-family: 'JetBrains Mono', monospace;
}

.event-list {
  display: flex;
  gap: 7px;
  align-items: center;
  margin-top: 12px;
  overflow-x: auto;
  padding-bottom: 4px;
}

.event-list button {
  flex: 0 0 auto;
  min-height: 34px;
  padding: 6px 9px;
  color: #ffd9cb;
  border-color: rgba(255, 129, 84, 0.32);
}

.event-policy {
  flex: 0 0 180px;
  color: var(--muted);
  font-size: 0.65rem;
  line-height: 1.35;
}

.evidence-grid {
  display: grid;
  grid-template-columns: repeat(3, minmax(0, 1fr));
  gap: 10px;
  margin-top: 14px;
}

.evidence-grid article {
  min-width: 0;
  padding: 13px;
  border: 1px solid rgba(255, 255, 255, 0.1);
  background: rgba(255, 255, 255, 0.025);
}

.evidence-grid h3 {
  min-width: 0;
  overflow-wrap: anywhere;
  word-break: break-word;
}

.evidence-grid p {
  margin-bottom: 0;
  color: var(--muted);
  font-size: 0.76rem;
  line-height: 1.5;
}

.scope-label {
  display: inline-block;
  margin-bottom: 10px;
  padding: 4px 6px;
  border: 1px solid currentColor;
  font-size: 0.6rem;
  letter-spacing: 0.08em;
  text-transform: uppercase;
}

.scope-label.validated { color: #72e4cd; }
.scope-label.unvalidated { color: #ffb066; }
.scope-label.implementation { color: #b8cfc6; }

.unsupported-panel {
  display: grid;
  grid-template-columns: minmax(190px, 0.38fr) 1fr;
  gap: 20px;
}

.unsupported-panel ul {
  display: grid;
  grid-template-columns: repeat(3, minmax(0, 1fr));
  gap: 8px;
  margin: 0;
  padding: 0;
  list-style: none;
}

.unsupported-panel li {
  display: grid;
  gap: 5px;
  padding: 11px;
  border: 1px solid rgba(255, 139, 103, 0.24);
  background: rgba(100, 35, 20, 0.12);
}

.unsupported-panel li span {
  color: #ffb397;
  font-size: 0.68rem;
  text-transform: uppercase;
}

.unsupported-panel li small {
  color: var(--muted);
  line-height: 1.4;
}

.inspector {
  display: grid;
  align-content: start;
  gap: 0;
}

.inspector-heading,
.inspector-section {
  padding: 14px;
  border-bottom: 1px solid var(--border);
}

.inspector-section:last-child {
  border-bottom: 0;
}

.inspector-section > p {
  color: var(--muted);
  font-size: 0.75rem;
  line-height: 1.45;
}

.current-diagnostic small {
  display: block;
  margin-top: 7px;
  color: #ffcb91;
  font-size: 0.66rem;
}

.diagnostic-value {
  display: grid;
  grid-template-columns: auto 1fr auto;
  gap: 7px;
  align-items: center;
  color: #eafff8;
  font-family: 'JetBrains Mono', monospace;
  font-size: 0.75rem;
}

.diagnostic-value i {
  height: 7px;
  background: linear-gradient(90deg, rgba(78, 218, 197, 0.4), #ffab58);
}

.top-moments {
  display: grid;
  gap: 7px;
  margin: 0;
  padding: 0;
  list-style: none;
}

.top-moments button {
  display: grid;
  grid-template-columns: 1fr auto;
  gap: 4px 8px;
  width: 100%;
  min-height: 56px;
  padding: 9px;
  border: 1px solid rgba(255, 255, 255, 0.09);
  color: var(--text);
  background: rgba(255, 255, 255, 0.025);
  text-align: left;
  cursor: pointer;
}

.top-moments button:hover {
  border-color: rgba(255, 179, 86, 0.4);
  background: rgba(255, 133, 61, 0.07);
}

.top-moments small {
  display: block;
  margin-top: 3px;
  color: var(--muted);
}

.moment-rank {
  color: #ffd59b;
  font-weight: 900;
}

.flag {
  grid-column: 1 / -1;
  display: inline-flex;
  justify-self: start;
  padding: 3px 5px;
  border: 1px solid currentColor;
  font-size: 0.58rem;
  letter-spacing: 0.06em;
  text-transform: uppercase;
}

.cold-flag { color: #85e0d1; }
.tail-flag { color: #ff9c7b; }

.target-contract dl {
  display: grid;
  gap: 8px;
  margin: 0;
}

.target-contract dl div {
  display: flex;
  justify-content: space-between;
  gap: 12px;
  padding-bottom: 7px;
  border-bottom: 1px solid rgba(255, 255, 255, 0.07);
}

.target-contract dt {
  color: var(--muted);
}

.target-contract dd {
  margin: 0;
  text-align: right;
}

.state-panel {
  display: flex;
  align-items: center;
  justify-content: center;
  gap: 18px;
  max-width: 1880px;
  min-height: 54vh;
  margin: 0 auto;
  padding: 28px;
  text-align: center;
}

.state-panel p {
  max-width: 620px;
  margin: 8px auto 0;
  color: var(--muted);
}

.compact-state {
  min-height: 220px;
}

.error-state {
  border-color: rgba(255, 126, 83, 0.4);
}

@media (max-width: 1280px) {
  .analyst-layout {
    grid-template-columns: 220px minmax(520px, 1fr);
  }

  .inspector {
    position: static;
    grid-column: 1 / -1;
    grid-template-columns: repeat(3, minmax(0, 1fr));
    max-height: none;
  }

  .inspector-heading,
  .inspector-section {
    border-right: 1px solid var(--border);
    border-bottom: 0;
  }
}

@media (max-width: 900px) {
  .analyst-page {
    padding: 10px;
  }

  .app-header,
  .brand-block {
    align-items: flex-start;
  }

  .app-header {
    display: grid;
  }

  .header-actions {
    justify-content: flex-start;
  }

  .analyst-layout {
    grid-template-columns: 1fr;
  }

  .asset-rail {
    position: static;
    max-height: none;
  }

  .asset-list {
    grid-template-columns: repeat(auto-fit, minmax(210px, 1fr));
  }

  .moment-strip,
  .evidence-grid,
  .unsupported-panel ul,
  .inspector {
    grid-template-columns: repeat(2, minmax(0, 1fr));
  }

  .unsupported-panel {
    grid-template-columns: 1fr;
  }
}

@media (max-width: 620px) {
  .brand-block {
    display: grid;
  }

  .playback-heading,
  .timeline-heading {
    display: grid;
  }

  .contract-badges,
  .timeline-legend {
    justify-content: flex-start;
  }

  .moment-strip,
  .evidence-grid,
  .unsupported-panel ul,
  .inspector {
    grid-template-columns: 1fr;
  }

  .transport-time {
    width: 100%;
    margin-left: 0;
  }

  .timeline-scrubber {
    grid-template-columns: 1fr auto;
  }

  .timeline-scrubber input {
    grid-column: 1 / -1;
    grid-row: 2;
  }

  .inspector-heading,
  .inspector-section {
    border-right: 0;
    border-bottom: 1px solid var(--border);
  }
}

@media (prefers-reduced-motion: reduce) {
  *,
  *::before,
  *::after {
    animation-duration: 0.01ms !important;
    animation-iteration-count: 1 !important;
    transition-duration: 0.01ms !important;
    scroll-behavior: auto !important;
  }
}
</style>
