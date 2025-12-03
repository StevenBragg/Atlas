import React, { createContext, useContext, useEffect, useState, useCallback, useRef } from 'react'
import type { SystemStatus, Metrics, WebSocketMessage } from '../types/atlas'
import { systemApi } from '../services/api'

interface AtlasContextType {
  status: SystemStatus | null
  metrics: Metrics | null
  connected: boolean
  error: string | null
  sendCommand: (command: string, payload?: Record<string, unknown>) => void
  refreshStatus: () => Promise<void>
  refreshMetrics: () => Promise<void>
}

const AtlasContext = createContext<AtlasContextType | null>(null)

export function AtlasProvider({ children }: { children: React.ReactNode }) {
  const [status, setStatus] = useState<SystemStatus | null>(null)
  const [metrics, setMetrics] = useState<Metrics | null>(null)
  const [connected, setConnected] = useState(false)
  const [error, setError] = useState<string | null>(null)

  const wsRef = useRef<WebSocket | null>(null)
  const reconnectTimeoutRef = useRef<number | null>(null)

  const connectWebSocket = useCallback(() => {
    // Determine WebSocket URL
    const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:'
    const host = window.location.host
    const wsUrl = `${protocol}//${host}/ws/stream`

    try {
      const ws = new WebSocket(wsUrl)

      ws.onopen = () => {
        console.log('WebSocket connected')
        setConnected(true)
        setError(null)
      }

      ws.onmessage = (event) => {
        try {
          const message: WebSocketMessage = JSON.parse(event.data)

          if (message.type === 'connected') {
            // Initial status from server
            if (message.data && typeof message.data === 'object' && 'initial_status' in message.data) {
              setStatus((message.data as { initial_status: SystemStatus }).initial_status)
            }
          } else if (message.type === 'heartbeat') {
            // Update metrics from heartbeat
            if (message.metrics) {
              setMetrics(message.metrics)
            }
          } else if (message.type === 'frame_processed' || message.type === 'audio_processed') {
            // Processing events - could update UI here
            console.log('Processing event:', message)
          } else if (message.type === 'learning_state_changed') {
            // Refresh status when learning state changes
            refreshStatus()
          }
        } catch (e) {
          console.error('Error parsing WebSocket message:', e)
        }
      }

      ws.onerror = (event) => {
        console.error('WebSocket error:', event)
        setError('WebSocket connection error')
      }

      ws.onclose = () => {
        console.log('WebSocket disconnected')
        setConnected(false)
        wsRef.current = null

        // Try to reconnect after 5 seconds
        reconnectTimeoutRef.current = window.setTimeout(() => {
          console.log('Attempting to reconnect...')
          connectWebSocket()
        }, 5000)
      }

      wsRef.current = ws
    } catch (e) {
      console.error('Error creating WebSocket:', e)
      setError('Failed to connect to Atlas')
    }
  }, [])

  const refreshStatus = useCallback(async () => {
    try {
      const newStatus = await systemApi.getStatus()
      setStatus(newStatus)
      setError(null)
    } catch (e) {
      console.error('Error fetching status:', e)
      setError('Failed to fetch status')
    }
  }, [])

  const refreshMetrics = useCallback(async () => {
    try {
      const newMetrics = await systemApi.getMetrics()
      setMetrics(newMetrics)
      setError(null)
    } catch (e) {
      console.error('Error fetching metrics:', e)
      setError('Failed to fetch metrics')
    }
  }, [])

  const sendCommand = useCallback((command: string, payload?: Record<string, unknown>) => {
    if (wsRef.current && wsRef.current.readyState === WebSocket.OPEN) {
      wsRef.current.send(JSON.stringify({ command, payload }))
    } else {
      console.error('WebSocket not connected')
    }
  }, [])

  // Initial connection and data fetch
  useEffect(() => {
    // Fetch initial data via REST
    refreshStatus()
    refreshMetrics()

    // Connect WebSocket for real-time updates
    connectWebSocket()

    return () => {
      // Cleanup
      if (reconnectTimeoutRef.current) {
        clearTimeout(reconnectTimeoutRef.current)
      }
      if (wsRef.current) {
        wsRef.current.close()
      }
    }
  }, [connectWebSocket, refreshStatus, refreshMetrics])

  // Periodic refresh as fallback
  useEffect(() => {
    const interval = setInterval(() => {
      if (!connected) {
        refreshStatus()
        refreshMetrics()
      }
    }, 10000) // Every 10 seconds if not connected

    return () => clearInterval(interval)
  }, [connected, refreshStatus, refreshMetrics])

  return (
    <AtlasContext.Provider
      value={{
        status,
        metrics,
        connected,
        error,
        sendCommand,
        refreshStatus,
        refreshMetrics,
      }}
    >
      {children}
    </AtlasContext.Provider>
  )
}

export function useAtlas() {
  const context = useContext(AtlasContext)
  if (!context) {
    throw new Error('useAtlas must be used within an AtlasProvider')
  }
  return context
}
