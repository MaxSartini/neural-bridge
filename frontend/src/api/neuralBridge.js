import service from './index'

const API_PREFIX = '/api/neural-bridge/v1'

const apiOrigin = () => {
  return (import.meta.env.VITE_API_BASE_URL || 'http://localhost:5001').replace(/\/$/, '')
}

const analysisPath = analysisId => {
  return `${API_PREFIX}/analyses/${encodeURIComponent(String(analysisId))}`
}

export const listNeuralBridgeAnalyses = (config = {}) => {
  return service.get(`${API_PREFIX}/analyses`, {
    ...config,
    params: {
      limit: 200,
      ...(config.params || {})
    }
  })
}

export const getNeuralBridgeTimeline = (analysisId, config = {}) => {
  return service.get(`${analysisPath(analysisId)}/timeline`, {
    ...config,
    params: {
      include_members: true,
      ...(config.params || {})
    }
  })
}

export const neuralBridgeMediaUrl = analysisId => {
  return `${apiOrigin()}${analysisPath(analysisId)}/media`
}

export const neuralBridgeJsonReportUrl = analysisId => {
  return `${apiOrigin()}${analysisPath(analysisId)}/report?format=json`
}

export const neuralBridgePredictionsCsvUrl = analysisId => {
  return `${apiOrigin()}${analysisPath(analysisId)}/predictions.csv`
}
