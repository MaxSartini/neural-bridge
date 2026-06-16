<template>
  <div class="neuro-page">
    <header class="neuro-header">
      <div>
        <button class="back-button" @click="router.push('/')">NEURAL BRIDGE</button>
        <p class="eyebrow">TRIBE v2 predicted BOLD proxy viewer</p>
        <h1>Neural Response Monitor</h1>
      </div>
      <div class="header-stats">
        <div class="stat-card">
          <span>{{ progress.complete }}</span>
          <label>cached</label>
        </div>
        <div class="stat-card">
          <span>{{ progress.total_seen }}</span>
          <label>seen</label>
        </div>
        <div class="stat-card">
          <span>{{ selectedVideoId || 'none' }}</span>
          <label>video</label>
        </div>
      </div>
    </header>

    <main class="neuro-grid">
      <aside class="video-list">
        <div class="panel-title">
          <span>Cache</span>
          <button @click="refreshAll">refresh</button>
        </div>
        <button
          v-for="video in videos"
          :key="video.video_id"
          class="video-item"
          :class="{ active: selectedVideoId === video.video_id, disabled: !video.complete }"
          :disabled="!video.complete"
          @click="selectVideo(video.video_id)"
        >
          <span class="video-id">#{{ video.video_id }}</span>
          <span class="video-meta">{{ formatSeconds(video.duration_seconds) }} · {{ video.manifest_rows || '?' }} windows</span>
          <span class="video-state">{{ video.complete ? 'ready' : 'running' }}</span>
        </button>
      </aside>

      <section class="brain-stage">
        <div class="stage-toolbar">
          <div>
            <p class="eyebrow">window {{ currentIndex + 1 }} / {{ timepoints }}</p>
            <h2>{{ selectedVideoId ? `Stimulus ${selectedVideoId}` : 'Select a cached stimulus' }}</h2>
          </div>
          <div class="controls">
            <button @click="togglePlayback">{{ playing ? 'pause' : 'play' }}</button>
            <button @click="step(-1)">-1</button>
            <button @click="step(1)">+1</button>
            <select v-model="speedMs">
              <option :value="900">slow</option>
              <option :value="450">normal</option>
              <option :value="180">fast</option>
            </select>
          </div>
        </div>

        <div class="brain-card">
          <div class="activation-legend">
            <span>deactivation</span>
            <i></i>
            <span>high predicted response</span>
          </div>
          <canvas ref="meshCanvas" class="mesh-canvas" :class="{ visible: surfaceReady }" />
          <div v-if="surfaceLoading" class="mesh-loading">loading fsaverage5 mesh...</div>
          <svg
            v-if="!surfaceReady"
            class="brain-svg"
            viewBox="0 0 100 100"
            role="img"
            aria-label="Animated cortical and subcortical activation map"
          >
            <defs>
              <radialGradient id="hemiGlow" cx="50%" cy="50%" r="65%">
                <stop offset="0%" stop-color="#1d3d3a" />
                <stop offset="100%" stop-color="#07110f" />
              </radialGradient>
              <filter id="softGlow">
                <feGaussianBlur stdDeviation="1.8" result="blur" />
                <feMerge>
                  <feMergeNode in="blur" />
                  <feMergeNode in="SourceGraphic" />
                </feMerge>
              </filter>
            </defs>

            <path class="hemi" d="M47 11 C25 10 11 27 12 50 C13 75 29 90 48 88 C43 75 42 64 45 51 C42 38 42 25 47 11Z" />
            <path class="hemi" d="M53 11 C75 10 89 27 88 50 C87 75 71 90 52 88 C57 75 58 64 55 51 C58 38 58 25 53 11Z" />
            <path class="fold" d="M24 31 C33 26 39 31 45 37" />
            <path class="fold" d="M18 53 C29 48 38 52 45 61" />
            <path class="fold" d="M76 31 C67 26 61 31 55 37" />
            <path class="fold" d="M82 53 C71 48 62 52 55 61" />
            <ellipse class="sub-core" cx="50" cy="54" rx="13" ry="25" />

            <circle
              v-for="region in visibleCorticalRegions"
              :key="region.id"
              class="region-dot cortical-dot"
              :cx="region.position.x"
              :cy="region.position.y"
              :r="dotRadius(region)"
              :fill="regionColor(region)"
              :opacity="dotOpacity(region)"
            >
              <title>{{ region.label }} · {{ activationLabel(region) }}</title>
            </circle>

            <circle
              v-for="region in subcorticalRegions"
              :key="region.id"
              class="region-dot sub-dot"
              :cx="region.position.x"
              :cy="region.position.y"
              :r="dotRadius(region) + 0.4"
              :fill="regionColor(region)"
              :opacity="region.interpretation_eligible ? dotOpacity(region) : 0.2"
            >
              <title>{{ region.label }} · {{ activationLabel(region) }}</title>
            </circle>
          </svg>

          <div class="trace-strip">
            <div
              v-for="(value, index) in globalTrace"
              :key="index"
              class="trace-bar"
              :class="{ current: index === currentIndex }"
              :style="{ height: `${12 + value * 88}%` }"
              @click="seekToIndex(index)"
            />
          </div>
        </div>
      </section>

      <aside class="side-panel">
        <div class="stimulus-card">
          <video
            v-if="selectedVideoId"
            ref="stimulusVideo"
            :src="mediaUrl"
            controls
            muted
            playsinline
            @timeupdate="syncIndexFromVideo"
            @seeked="syncIndexFromVideo"
            @play="playing = true"
            @pause="playing = false"
          />
          <div v-if="selectedVideoId" class="sync-chip">video-synced · {{ currentTimestampLabel }}</div>
          <div v-else class="empty-video">No video selected</div>
        </div>

        <div class="metric-card">
          <h3>Global response</h3>
          <div class="metric-row" v-for="metric in globalMetrics" :key="metric.label">
            <span>{{ metric.label }}</span>
            <strong>{{ metric.value }}</strong>
          </div>
        </div>

        <div class="roi-card">
          <h3>Top live regions</h3>
          <div v-for="region in topLiveRegions" :key="region.id" class="roi-row">
            <span>{{ cleanLabel(region.label) }}</span>
            <div class="roi-meter">
              <i :style="{ width: `${regionMagnitude(region) * 100}%`, background: regionColor(region) }" />
            </div>
          </div>
        </div>
      </aside>
    </main>
  </div>
</template>

<script setup>
import { computed, nextTick, onMounted, onUnmounted, ref, watch } from 'vue'
import { useRouter } from 'vue-router'
import * as THREE from 'three'
import {
  getNeuroViewerProgress,
  getNeuroViewerSurface,
  getNeuroViewerTimeline,
  listNeuroViewerVideos,
  neuroViewerMediaUrl
} from '../api/neuroViewer'

const router = useRouter()
const progress = ref({ complete: 0, total_seen: 0, videos: [] })
const videos = ref([])
const selectedVideoId = ref(null)
const timeline = ref(null)
const surface = ref(null)
const surfaceReady = ref(false)
const surfaceLoading = ref(false)
const meshCanvas = ref(null)
const stimulusVideo = ref(null)
const currentIndex = ref(0)
const playing = ref(false)
const speedMs = ref(450)
let playTimer = null
let refreshTimer = null
let renderer = null
let scene = null
let camera = null
let brainMesh = null
let animationFrame = null
let dragState = null

const timepoints = computed(() => timeline.value?.timepoints || 0)
const corticalRegions = computed(() => timeline.value?.regions?.cortical || [])
const subcorticalRegions = computed(() => timeline.value?.regions?.subcortical || [])
const visibleCorticalRegions = computed(() => corticalRegions.value.slice(0, 36))
const globalTrace = computed(() => timeline.value?.global_traces?.mean_abs || [])
const mediaUrl = computed(() => selectedVideoId.value ? neuroViewerMediaUrl(selectedVideoId.value) : '')
const currentTimestampLabel = computed(() => {
  const timestamps = timeline.value?.timestamps_seconds || surface.value?.timestamps_seconds || []
  const seconds = timestamps[currentIndex.value] ?? currentIndex.value
  return `${Number(seconds || 0).toFixed(1)}s`
})

const globalMetrics = computed(() => {
  const summary = timeline.value?.summary || {}
  return [
    ['mean activation', summary.mean_activation_proxy],
    ['temporal variance', summary.temporal_variance_proxy],
    ['peak response', summary.peak_response_proxy],
    ['retention', summary.segment_quality?.retention_ratio]
  ].map(([label, value]) => ({ label, value: value === undefined || value === null ? 'n/a' : Number(value).toFixed(3) }))
})

const topLiveRegions = computed(() => {
  return [...corticalRegions.value, ...subcorticalRegions.value]
    .map(region => ({ ...region, live: regionMagnitude(region) }))
    .sort((a, b) => b.live - a.live)
    .slice(0, 10)
})

const refreshAll = async () => {
  const [progressRes, videosRes] = await Promise.all([
    getNeuroViewerProgress(),
    listNeuroViewerVideos()
  ])
  progress.value = progressRes.data || progressRes
  videos.value = videosRes.data || videosRes
  if (!selectedVideoId.value) {
    const firstReady = videos.value.find(video => video.complete)
    if (firstReady) await selectVideo(firstReady.video_id)
  }
}

const selectVideo = async (videoId) => {
  selectedVideoId.value = String(videoId)
  currentIndex.value = 0
  playing.value = false
  surfaceReady.value = false
  surface.value = null
  const response = await getNeuroViewerTimeline(videoId, 44)
  timeline.value = response.data || response
  await loadSurface(videoId)
  await nextTick()
  seekToIndex(0, false)
}

const regionValue = (region) => {
  return Number(region.trace?.[currentIndex.value] || 0)
}

const regionMagnitude = (region) => {
  return Math.abs(regionValue(region))
}

const regionColor = (region) => {
  const [r, g, b] = activationToColor(regionValue(region), 0.45).map(channel => Math.round(channel * 255))
  return `rgb(${r}, ${g}, ${b})`
}

const dotRadius = (region) => {
  return 1.1 + regionMagnitude(region) * (region.kind === 'subcortical' ? 4.8 : 4.0)
}

const dotOpacity = (region) => {
  return 0.28 + regionMagnitude(region) * 0.72
}

const activationLabel = (region) => {
  return regionValue(region).toFixed(3)
}

const cleanLabel = (label) => {
  return String(label || '').replace(/^L:/, 'L ').replace(/^R:/, 'R ').replace(/_/g, ' ')
}

const loadSurface = async (videoId) => {
  surfaceLoading.value = true
  try {
    const response = await getNeuroViewerSurface(videoId, 1)
    surface.value = response.data || response
    await nextTick()
    buildSurfaceScene()
  } finally {
    surfaceLoading.value = false
  }
}

const lerp = (a, b, t) => a + (b - a) * t

const colorRamp = (stops, t) => {
  const value = Math.max(0, Math.min(1, t))
  const scaled = value * (stops.length - 1)
  const index = Math.min(stops.length - 2, Math.floor(scaled))
  const local = scaled - index
  return [
    lerp(stops[index][0], stops[index + 1][0], local),
    lerp(stops[index][1], stops[index + 1][1], local),
    lerp(stops[index][2], stops[index + 1][2], local)
  ]
}

const mixColor = (a, b, t) => {
  const value = Math.max(0, Math.min(1, t))
  return [
    lerp(a[0], b[0], value),
    lerp(a[1], b[1], value),
    lerp(a[2], b[2], value)
  ]
}

const activationToColor = (value, background = 0.5) => {
  const signed = Math.max(-1, Math.min(1, Number(value || 0)))
  const magnitude = Math.abs(signed)
  const sulcal = 0.13 + Number(background || 0) * 0.16
  const cortexBase = [sulcal * 0.62, sulcal * 0.92, sulcal * 0.82]
  const hot = colorRamp(
    [
      [0.17, 0.04, 0.02],
      [0.58, 0.02, 0.01],
      [1.0, 0.22, 0.01],
      [1.0, 0.73, 0.06],
      [1.0, 0.98, 0.76]
    ],
    magnitude
  )
  const cold = colorRamp(
    [
      [0.03, 0.06, 0.14],
      [0.04, 0.16, 0.46],
      [0.0, 0.55, 0.95],
      [0.46, 0.92, 1.0]
    ],
    magnitude
  )
  const target = signed >= 0 ? hot : cold
  const boost = Math.min(1, 0.18 + magnitude * 1.12)
  return mixColor(cortexBase, target, boost)
}

const disposeSurfaceScene = () => {
  if (animationFrame) cancelAnimationFrame(animationFrame)
  animationFrame = null
  if (brainMesh) {
    brainMesh.geometry?.dispose()
    brainMesh.material?.dispose()
  }
  if (renderer) renderer.dispose()
  renderer = null
  scene = null
  camera = null
  brainMesh = null
  surfaceReady.value = false
}

const buildSurfaceScene = () => {
  if (!meshCanvas.value || !surface.value?.surface?.coords?.length) return
  disposeSurfaceScene()

  const canvas = meshCanvas.value
  renderer = new THREE.WebGLRenderer({ canvas, antialias: true, alpha: true })
  renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2))
  scene = new THREE.Scene()
  scene.fog = new THREE.FogExp2(0x06100f, 0.035)
  camera = new THREE.PerspectiveCamera(34, 1, 0.01, 100)
  camera.position.set(0, 0, 4.2)
  scene.add(new THREE.AmbientLight(0x8fffe1, 0.62))
  const keyLight = new THREE.DirectionalLight(0xffc46b, 1.35)
  keyLight.position.set(2.2, 1.6, 3.5)
  scene.add(keyLight)
  const rimLight = new THREE.DirectionalLight(0x4de2ff, 1.15)
  rimLight.position.set(-2.5, -1.2, 2.8)
  scene.add(rimLight)

  const coords = surface.value.surface.coords
  const faces = surface.value.surface.faces
  const background = surface.value.surface.background || []
  const positions = new Float32Array(coords.length * 3)
  const colors = new Float32Array(coords.length * 3)
  for (let i = 0; i < coords.length; i++) {
    positions[i * 3] = coords[i][0]
    positions[i * 3 + 1] = coords[i][1]
    positions[i * 3 + 2] = coords[i][2]
    const color = activationToColor(0, background[i])
    colors[i * 3] = color[0]
    colors[i * 3 + 1] = color[1]
    colors[i * 3 + 2] = color[2]
  }

  const indices = new Uint32Array(faces.length * 3)
  for (let i = 0; i < faces.length; i++) {
    indices[i * 3] = faces[i][0]
    indices[i * 3 + 1] = faces[i][1]
    indices[i * 3 + 2] = faces[i][2]
  }

  const geometry = new THREE.BufferGeometry()
  geometry.setAttribute('position', new THREE.BufferAttribute(positions, 3))
  geometry.setAttribute('color', new THREE.BufferAttribute(colors, 3))
  geometry.setIndex(new THREE.BufferAttribute(indices, 1))
  geometry.computeVertexNormals()

  const material = new THREE.MeshStandardMaterial({
    vertexColors: true,
    side: THREE.DoubleSide,
    transparent: true,
    opacity: 0.98,
    roughness: 0.42,
    metalness: 0.05,
    emissive: new THREE.Color(0x061d18),
    emissiveIntensity: 0.18
  })
  brainMesh = new THREE.Mesh(geometry, material)
  brainMesh.rotation.x = -0.18
  brainMesh.rotation.y = 0.22
  scene.add(brainMesh)

  canvas.onpointerdown = event => {
    dragState = { x: event.clientX, y: event.clientY, rx: brainMesh.rotation.x, ry: brainMesh.rotation.y }
    canvas.setPointerCapture(event.pointerId)
  }
  canvas.onpointermove = event => {
    if (!dragState) return
    brainMesh.rotation.y = dragState.ry + (event.clientX - dragState.x) * 0.008
    brainMesh.rotation.x = dragState.rx + (event.clientY - dragState.y) * 0.008
  }
  canvas.onpointerup = () => {
    dragState = null
  }

  surfaceReady.value = true
  resizeSurface()
  updateSurfaceColors()
  animateSurface()
}

const resizeSurface = () => {
  if (!renderer || !camera || !meshCanvas.value) return
  const rect = meshCanvas.value.getBoundingClientRect()
  const width = Math.max(320, Math.floor(rect.width))
  const height = Math.max(320, Math.floor(rect.height))
  renderer.setSize(width, height, false)
  camera.aspect = width / height
  camera.updateProjectionMatrix()
}

const updateSurfaceColors = () => {
  if (!brainMesh || !surface.value?.activity?.length) return
  const activity = surface.value.activity[Math.min(currentIndex.value, surface.value.activity.length - 1)] || []
  const background = surface.value.surface.background || []
  const colorAttr = brainMesh.geometry.getAttribute('color')
  for (let i = 0; i < colorAttr.count; i++) {
    const color = activationToColor(activity[i] || 0, background[i])
    colorAttr.setXYZ(i, color[0], color[1], color[2])
  }
  colorAttr.needsUpdate = true
}

const animateSurface = () => {
  if (!renderer || !scene || !camera) return
  if (brainMesh && !dragState) brainMesh.rotation.y += 0.002
  renderer.render(scene, camera)
  animationFrame = requestAnimationFrame(animateSurface)
}

const formatSeconds = (value) => {
  if (!value && value !== 0) return '?'
  return `${Number(value).toFixed(1)}s`
}

const indexFromSeconds = (seconds) => {
  if (!timepoints.value) return 0
  const timestamps = timeline.value?.timestamps_seconds || surface.value?.timestamps_seconds || []
  if (!timestamps.length) {
    return Math.max(0, Math.min(timepoints.value - 1, Math.floor(Number(seconds || 0))))
  }
  const value = Number(seconds || 0)
  let bestIndex = 0
  let bestDistance = Infinity
  timestamps.forEach((timestamp, index) => {
    const distance = Math.abs(Number(timestamp) - value)
    if (distance < bestDistance) {
      bestDistance = distance
      bestIndex = index
    }
  })
  return Math.max(0, Math.min(timepoints.value - 1, bestIndex))
}

const seekToIndex = (index, keepPlaying = playing.value) => {
  if (!timepoints.value) return
  const targetIndex = Math.max(0, Math.min(timepoints.value - 1, index))
  currentIndex.value = targetIndex
  const video = stimulusVideo.value
  if (!video) return
  const timestamps = timeline.value?.timestamps_seconds || surface.value?.timestamps_seconds || []
  const targetTime = Number(timestamps[targetIndex] ?? targetIndex)
  if (Number.isFinite(targetTime) && Math.abs(video.currentTime - targetTime) > 0.25) {
    video.currentTime = targetTime
  }
  if (keepPlaying && video.paused) {
    video.play().catch(() => {
      playing.value = false
    })
  }
}

const syncIndexFromVideo = () => {
  const video = stimulusVideo.value
  if (!video || !timepoints.value) return
  currentIndex.value = indexFromSeconds(video.currentTime)
}

const step = (direction) => {
  if (!timepoints.value) return
  seekToIndex((currentIndex.value + direction + timepoints.value) % timepoints.value, false)
}

const togglePlayback = () => {
  const video = stimulusVideo.value
  if (video) {
    if (video.paused) {
      video.play().catch(() => {
        playing.value = false
      })
    } else {
      video.pause()
    }
    return
  }
  playing.value = !playing.value
}

const restartPlayback = () => {
  if (playTimer) clearInterval(playTimer)
  playTimer = setInterval(() => {
    const video = stimulusVideo.value
    if (video && !video.paused) {
      syncIndexFromVideo()
      return
    }
    if (!video && playing.value && timepoints.value > 0) step(1)
  }, Number(speedMs.value))
}

watch(speedMs, restartPlayback)
watch(currentIndex, updateSurfaceColors)

onMounted(async () => {
  await refreshAll()
  restartPlayback()
  refreshTimer = setInterval(refreshAll, 8000)
})

onUnmounted(() => {
  if (playTimer) clearInterval(playTimer)
  if (refreshTimer) clearInterval(refreshTimer)
  window.removeEventListener('resize', resizeSurface)
  disposeSurfaceScene()
})

window.addEventListener('resize', resizeSurface)
</script>

<style scoped>
.neuro-page {
  min-height: 100vh;
  background:
    radial-gradient(circle at 18% 10%, rgba(0, 238, 255, 0.22), transparent 25%),
    radial-gradient(circle at 78% 14%, rgba(255, 103, 0, 0.25), transparent 28%),
    radial-gradient(circle at 50% 95%, rgba(35, 255, 151, 0.14), transparent 38%),
    linear-gradient(135deg, #030908 0%, #081715 45%, #160b06 100%);
  color: #e8f7ef;
  font-family: 'Space Grotesk', 'JetBrains Mono', sans-serif;
  padding: 26px;
}

.neuro-header {
  display: flex;
  justify-content: space-between;
  gap: 24px;
  align-items: flex-start;
  margin-bottom: 24px;
}

.back-button,
.controls button,
.controls select,
.panel-title button {
  border: 1px solid rgba(255, 198, 107, 0.32);
  background: linear-gradient(135deg, rgba(255, 117, 31, 0.16), rgba(54, 224, 255, 0.08));
  color: #e8f7ef;
  padding: 9px 13px;
  text-transform: uppercase;
  letter-spacing: 0.08em;
  cursor: pointer;
}

.eyebrow {
  color: #83d6b7;
  font-size: 0.78rem;
  letter-spacing: 0.14em;
  text-transform: uppercase;
  margin: 12px 0 5px;
}

h1,
h2,
h3 {
  margin: 0;
  letter-spacing: -0.04em;
}

h1 {
  font-size: clamp(2.4rem, 5vw, 5.4rem);
  line-height: 0.92;
}

.header-stats {
  display: flex;
  gap: 12px;
}

.stat-card,
.video-list,
.brain-card,
.stimulus-card,
.metric-card,
.roi-card {
  background:
    linear-gradient(145deg, rgba(6, 18, 17, 0.82), rgba(20, 10, 6, 0.62)),
    radial-gradient(circle at top left, rgba(255, 126, 33, 0.1), transparent 42%);
  border: 1px solid rgba(238, 255, 232, 0.15);
  box-shadow:
    0 20px 70px rgba(0, 0, 0, 0.34),
    inset 0 0 60px rgba(30, 255, 190, 0.035);
  backdrop-filter: blur(18px);
}

.stat-card {
  min-width: 98px;
  padding: 14px;
}

.stat-card span {
  display: block;
  font-size: 1.5rem;
  font-weight: 800;
}

.stat-card label {
  color: #83d6b7;
  font-size: 0.72rem;
  text-transform: uppercase;
}

.neuro-grid {
  display: grid;
  grid-template-columns: 250px minmax(420px, 1fr) 330px;
  gap: 18px;
  align-items: stretch;
}

.video-list,
.side-panel {
  min-height: 72vh;
}

.video-list {
  padding: 14px;
  overflow: auto;
}

.panel-title {
  display: flex;
  align-items: center;
  justify-content: space-between;
  margin-bottom: 12px;
  color: #83d6b7;
  text-transform: uppercase;
  letter-spacing: 0.1em;
}

.video-item {
  width: 100%;
  border: 1px solid rgba(255, 255, 255, 0.08);
  background: rgba(255, 255, 255, 0.04);
  color: inherit;
  padding: 12px;
  margin-bottom: 8px;
  text-align: left;
  display: grid;
  gap: 5px;
  cursor: pointer;
}

.video-item.active {
  border-color: #ffb84c;
  background:
    linear-gradient(90deg, rgba(255, 116, 22, 0.22), rgba(0, 229, 255, 0.08));
  box-shadow: 0 0 22px rgba(255, 110, 24, 0.14);
}

.video-item.disabled {
  opacity: 0.42;
  cursor: not-allowed;
}

.video-id {
  font-weight: 900;
}

.video-meta,
.video-state {
  color: #aac7bc;
  font-size: 0.76rem;
}

.brain-stage {
  min-width: 0;
}

.stage-toolbar {
  display: flex;
  justify-content: space-between;
  gap: 16px;
  align-items: flex-end;
  margin-bottom: 14px;
}

.controls {
  display: flex;
  gap: 8px;
}

.brain-card {
  position: relative;
  min-height: 72vh;
  padding: 16px;
  display: grid;
  grid-template-rows: minmax(420px, 1fr) 70px;
  overflow: hidden;
}

.brain-card::before {
  content: '';
  position: absolute;
  inset: 5%;
  background:
    radial-gradient(circle at 50% 42%, rgba(255, 205, 84, 0.22), transparent 20%),
    radial-gradient(circle at 36% 50%, rgba(0, 229, 255, 0.16), transparent 34%),
    radial-gradient(circle at 64% 52%, rgba(255, 70, 8, 0.18), transparent 38%);
  filter: blur(34px);
}

.brain-card::after {
  content: '';
  position: absolute;
  inset: 0;
  pointer-events: none;
  background:
    linear-gradient(rgba(255, 255, 255, 0.035) 1px, transparent 1px),
    linear-gradient(90deg, rgba(255, 255, 255, 0.025) 1px, transparent 1px);
  background-size: 42px 42px;
  mask-image: radial-gradient(circle at center, black 0%, transparent 78%);
  opacity: 0.45;
}

.activation-legend {
  position: absolute;
  z-index: 5;
  right: 18px;
  top: 18px;
  display: grid;
  grid-template-columns: auto 130px auto;
  gap: 9px;
  align-items: center;
  padding: 9px 11px;
  background: rgba(2, 8, 8, 0.58);
  border: 1px solid rgba(255, 255, 255, 0.14);
  color: #dffbef;
  font-size: 0.66rem;
  letter-spacing: 0.08em;
  text-transform: uppercase;
  backdrop-filter: blur(12px);
}

.activation-legend i {
  display: block;
  height: 9px;
  border-radius: 99px;
  background: linear-gradient(90deg, #05173f 0%, #0099ff 20%, #241008 43%, #e21a00 60%, #ff9b00 78%, #fff4b8 100%);
  box-shadow: 0 0 16px rgba(255, 132, 0, 0.32);
}

.mesh-canvas {
  position: relative;
  z-index: 2;
  width: 100%;
  height: 100%;
  min-height: 420px;
  opacity: 0;
  transition: opacity 220ms ease;
  cursor: grab;
  filter:
    drop-shadow(0 0 22px rgba(255, 111, 0, 0.2))
    drop-shadow(0 0 34px rgba(0, 219, 255, 0.12));
}

.mesh-canvas.visible {
  opacity: 1;
}

.mesh-canvas:active {
  cursor: grabbing;
}

.mesh-loading {
  position: absolute;
  z-index: 4;
  top: 22px;
  left: 22px;
  color: #ffcf7d;
  text-transform: uppercase;
  letter-spacing: 0.12em;
  font-size: 0.72rem;
}

.brain-svg {
  position: relative;
  width: 100%;
  height: 100%;
  z-index: 1;
}

.hemi {
  fill: url(#hemiGlow);
  stroke: rgba(255, 197, 103, 0.3);
  stroke-width: 0.7;
}

.fold {
  fill: none;
  stroke: rgba(191, 240, 218, 0.2);
  stroke-width: 0.5;
}

.sub-core {
  fill: rgba(255, 120, 10, 0.12);
  stroke: rgba(255, 199, 92, 0.3);
}

.region-dot {
  filter: url(#softGlow);
  transition: r 160ms linear, opacity 160ms linear, fill 160ms linear;
  mix-blend-mode: screen;
}

.sub-dot {
  stroke: rgba(255, 255, 255, 0.28);
  stroke-width: 0.25;
}

.trace-strip {
  position: relative;
  z-index: 2;
  display: flex;
  gap: 2px;
  align-items: end;
  border-top: 1px solid rgba(255, 205, 112, 0.16);
  padding-top: 14px;
}

.trace-bar {
  flex: 1;
  min-width: 3px;
  background: linear-gradient(to top, #04265b 0%, #00a6ff 20%, #eb2500 62%, #ffc847 100%);
  opacity: 0.52;
  cursor: pointer;
  box-shadow: 0 0 10px rgba(255, 96, 14, 0.12);
}

.trace-bar.current {
  opacity: 1;
  box-shadow:
    0 0 16px #ff8c1a,
    0 0 28px rgba(255, 210, 104, 0.45);
}

.side-panel {
  display: grid;
  grid-template-rows: auto auto 1fr;
  gap: 14px;
}

.stimulus-card,
.metric-card,
.roi-card {
  padding: 14px;
}

.stimulus-card video,
.empty-video {
  width: 100%;
  aspect-ratio: 16 / 9;
  background: #020605;
  object-fit: contain;
}

.sync-chip {
  margin-top: 10px;
  display: inline-flex;
  padding: 7px 9px;
  border: 1px solid rgba(255, 205, 112, 0.22);
  background: rgba(255, 132, 0, 0.1);
  color: #ffd58c;
  font-size: 0.68rem;
  letter-spacing: 0.1em;
  text-transform: uppercase;
}

.empty-video {
  display: grid;
  place-items: center;
  color: #83d6b7;
}

.metric-card h3,
.roi-card h3 {
  margin-bottom: 13px;
}

.metric-row {
  display: flex;
  justify-content: space-between;
  padding: 8px 0;
  border-bottom: 1px solid rgba(255, 255, 255, 0.08);
  color: #b8d4ca;
}

.metric-row strong {
  color: #fff3df;
}

.roi-row {
  display: grid;
  grid-template-columns: 1fr 92px;
  gap: 10px;
  align-items: center;
  padding: 7px 0;
  font-size: 0.82rem;
  color: #d7eee5;
}

.roi-meter {
  height: 7px;
  background: linear-gradient(90deg, rgba(0, 118, 255, 0.18), rgba(255, 89, 0, 0.12));
  overflow: hidden;
}

.roi-meter i {
  display: block;
  height: 100%;
  box-shadow: 0 0 12px currentColor;
}

@media (max-width: 1100px) {
  .neuro-grid {
    grid-template-columns: 1fr;
  }

  .video-list,
  .side-panel {
    min-height: auto;
  }

  .brain-card {
    min-height: 62vh;
  }

  .neuro-header,
  .stage-toolbar {
    flex-direction: column;
    align-items: stretch;
  }
}
</style>
