import service from './index'

export const getNeuroViewerProgress = () => {
  return service.get('/api/neuro-viewer/progress')
}

export const listNeuroViewerVideos = () => {
  return service.get('/api/neuro-viewer/videos')
}

export const getNeuroViewerTimeline = (videoId, maxRegions = 36) => {
  return service.get(`/api/neuro-viewer/videos/${videoId}/timeline`, {
    params: { max_regions: maxRegions }
  })
}

export const getNeuroViewerSurface = (videoId, timeStride = 1) => {
  return service.get(`/api/neuro-viewer/videos/${videoId}/surface`, {
    params: { time_stride: timeStride }
  })
}

export const neuroViewerMediaUrl = (videoId) => {
  const baseUrl = import.meta.env.VITE_API_BASE_URL || 'http://localhost:5001'
  return `${baseUrl}/api/neuro-viewer/videos/${videoId}/media`
}
