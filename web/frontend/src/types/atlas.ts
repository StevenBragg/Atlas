// Atlas system types

export interface SystemStatus {
  initialized: boolean
  atlas_available: boolean
  learning_enabled: boolean
  stats: SystemStats
  timestamp: string
  system_state?: SystemState
  architecture?: ArchitectureOverview
}

export interface SystemStats {
  frames_processed: number
  audio_chunks_processed: number
  start_time: string | null
  last_activity: string | null
}

export interface SystemState {
  total_frames: number
  learning_step: number
}

export interface ArchitectureOverview {
  visual_layers: Record<string, unknown>
  audio_layers: Record<string, unknown>
  multimodal_size: number
}

export interface Metrics {
  frames_processed: number
  audio_chunks_processed: number
  uptime_seconds: number
  timestamp: string
  prediction_error?: number
  reconstruction_error?: number
  cross_modal_correlation?: number
  total_neurons?: number
  active_associations?: number
}

export interface ArchitectureInfo {
  visual?: {
    layers: LayerInfo[]
  }
  audio?: {
    layers: LayerInfo[]
  }
  multimodal?: {
    size: number
    associations: number
  }
  timestamp: string
}

export interface LayerInfo {
  name: string
  neurons: number
  type: string
}

export interface MemoryItem {
  id: number
  content?: string
  timestamp?: string
  summary?: string
  importance?: number
  name?: string
  category?: string
  connections?: number
}

export interface MemoryContents {
  memory_type: string
  timestamp: string
  items: MemoryItem[]
  error?: string
}

export interface ProcessingResult {
  processed: boolean
  timestamp: string
  frame_number?: number
  chunk_number?: number
  predictions?: {
    temporal?: number[][]
    cross_modal?: number[]
  }
  error?: string
}

export interface Checkpoint {
  name: string
  path: string
  size_bytes: number
  modified: string
}

export interface WebSocketMessage {
  type: string
  data?: unknown
  metrics?: Metrics
  error?: string
  timestamp?: string
}

export interface Association {
  visual_pattern: string
  audio_pattern: string
  strength: number
}
