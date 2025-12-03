// Atlas API service

import type {
  SystemStatus,
  Metrics,
  ArchitectureInfo,
  MemoryContents,
  ProcessingResult,
  Checkpoint,
} from '../types/atlas'

const API_BASE = '/api'

async function fetchJson<T>(url: string, options?: RequestInit): Promise<T> {
  const response = await fetch(`${API_BASE}${url}`, {
    ...options,
    headers: {
      'Content-Type': 'application/json',
      ...options?.headers,
    },
  })

  if (!response.ok) {
    throw new Error(`API error: ${response.status} ${response.statusText}`)
  }

  return response.json()
}

// System endpoints
export const systemApi = {
  getStatus: () => fetchJson<SystemStatus>('/system/status'),
  getMetrics: () => fetchJson<Metrics>('/system/metrics'),
  getArchitecture: () => fetchJson<ArchitectureInfo>('/system/architecture'),
  getCheckpoints: () => fetchJson<{ checkpoints: Checkpoint[] }>('/system/checkpoints'),
}

// Data input endpoints
export const dataApi = {
  processFrame: async (imageBase64: string, learn = true): Promise<ProcessingResult> => {
    return fetchJson('/data/frame', {
      method: 'POST',
      body: JSON.stringify({ image_base64: imageBase64, learn }),
    })
  },

  processAudio: async (audioBase64: string, sampleRate = 22050, learn = true): Promise<ProcessingResult> => {
    return fetchJson('/data/audio', {
      method: 'POST',
      body: JSON.stringify({ audio_base64: audioBase64, sample_rate: sampleRate, learn }),
    })
  },

  processAVPair: async (
    imageBase64?: string,
    audioBase64?: string,
    sampleRate = 22050,
    learn = true
  ): Promise<ProcessingResult> => {
    return fetchJson('/data/av-pair', {
      method: 'POST',
      body: JSON.stringify({
        image_base64: imageBase64,
        audio_base64: audioBase64,
        sample_rate: sampleRate,
        learn,
      }),
    })
  },

  uploadFrame: async (file: File, learn = true): Promise<ProcessingResult> => {
    const formData = new FormData()
    formData.append('file', file)
    formData.append('learn', String(learn))

    const response = await fetch(`${API_BASE}/data/frame/upload`, {
      method: 'POST',
      body: formData,
    })

    if (!response.ok) {
      throw new Error(`API error: ${response.status} ${response.statusText}`)
    }

    return response.json()
  },

  getPredictions: (modality = 'visual', numSteps = 5) =>
    fetchJson<{ temporal: number[][]; cross_modal?: number[] }>(
      `/data/predictions?modality=${modality}&num_steps=${numSteps}`
    ),
}

// Memory endpoints
export const memoryApi = {
  getEpisodic: (limit = 100) => fetchJson<MemoryContents>(`/memory/episodic?limit=${limit}`),
  getSemantic: (limit = 100) => fetchJson<MemoryContents>(`/memory/semantic?limit=${limit}`),
  getWorking: () => fetchJson<MemoryContents>('/memory/working'),
  getAssociations: () => fetchJson<{
    total_associations: number
    top_associations: Array<{ visual_pattern: string; audio_pattern: string; strength: number }>
  }>('/memory/associations'),
  getStatistics: () => fetchJson('/memory/statistics'),
  search: (query: string, memoryType = 'all', limit = 10) =>
    fetchJson('/memory/search', {
      method: 'POST',
      body: JSON.stringify({ query, memory_type: memoryType, limit }),
    }),
}

// Control endpoints
export const controlApi = {
  setLearning: (enabled: boolean) =>
    fetchJson('/control/learning', {
      method: 'POST',
      body: JSON.stringify({ enabled }),
    }),

  getLearningState: () => fetchJson<{ learning_enabled: boolean }>('/control/learning'),

  setLearningRate: (rate: number) =>
    fetchJson('/control/learning-rate', {
      method: 'POST',
      body: JSON.stringify({ rate }),
    }),

  saveCheckpoint: (name?: string) =>
    fetchJson('/control/checkpoint/save', {
      method: 'POST',
      body: JSON.stringify({ name }),
    }),

  loadCheckpoint: (name: string) =>
    fetchJson('/control/checkpoint/load', {
      method: 'POST',
      body: JSON.stringify({ name }),
    }),

  deleteCheckpoint: (name: string) =>
    fetchJson(`/control/checkpoint/${name}`, {
      method: 'DELETE',
    }),

  setMode: (mode: string) =>
    fetchJson('/control/mode', {
      method: 'POST',
      body: JSON.stringify({ mode }),
    }),

  getMode: () => fetchJson<{ mode: string }>('/control/mode'),

  updateConfig: (config: {
    learning_rate?: number
    multimodal_size?: number
    prune_interval?: number
    enable_structural_plasticity?: boolean
    enable_temporal_prediction?: boolean
  }) =>
    fetchJson('/control/config', {
      method: 'POST',
      body: JSON.stringify(config),
    }),

  think: (task = 'reflect', cycles = 1) =>
    fetchJson(`/control/think?task=${task}&cycles=${cycles}`, {
      method: 'POST',
    }),

  imagine: (steps = 10) =>
    fetchJson(`/control/imagine?steps=${steps}`, {
      method: 'POST',
    }),
}
