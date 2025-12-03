import { useState, useEffect } from 'react'
import { useAtlas } from '../hooks/useAtlas'
import { controlApi, systemApi } from '../services/api'
import type { Checkpoint } from '../types/atlas'
import {
  Settings,
  Play,
  Pause,
  Save,
  Upload,
  Trash2,
  Sliders,
  Brain,
  Sparkles,
  RefreshCw,
  CheckCircle,
  AlertCircle,
  Loader2,
} from 'lucide-react'

function Control() {
  const { status, refreshStatus } = useAtlas()
  const [learningEnabled, setLearningEnabled] = useState(status?.learning_enabled || false)
  const [learningRate, setLearningRate] = useState(0.01)
  const [cognitiveMode, setCognitiveMode] = useState('autonomous')
  const [checkpoints, setCheckpoints] = useState<Checkpoint[]>([])
  const [newCheckpointName, setNewCheckpointName] = useState('')
  const [loading, setLoading] = useState<string | null>(null)
  const [message, setMessage] = useState<{ type: 'success' | 'error'; text: string } | null>(null)

  // Thinking/Imagination
  const [thinkingTask, setThinkingTask] = useState('reflect')
  const [thinkingCycles, setThinkingCycles] = useState(1)
  const [imaginationSteps, setImaginationSteps] = useState(10)
  const [thinkingResult, setThinkingResult] = useState<string | null>(null)
  const [imaginationResult, setImaginationResult] = useState<string | null>(null)

  useEffect(() => {
    loadCheckpoints()
    loadCurrentMode()
  }, [])

  useEffect(() => {
    setLearningEnabled(status?.learning_enabled || false)
  }, [status])

  const loadCheckpoints = async () => {
    try {
      const result = await systemApi.getCheckpoints()
      setCheckpoints(result.checkpoints || [])
    } catch (e) {
      console.error('Error loading checkpoints:', e)
    }
  }

  const loadCurrentMode = async () => {
    try {
      const result = await controlApi.getMode()
      setCognitiveMode(result.mode)
    } catch (e) {
      console.error('Error loading mode:', e)
    }
  }

  const showMessage = (type: 'success' | 'error', text: string) => {
    setMessage({ type, text })
    setTimeout(() => setMessage(null), 3000)
  }

  const toggleLearning = async () => {
    setLoading('learning')
    try {
      await controlApi.setLearning(!learningEnabled)
      setLearningEnabled(!learningEnabled)
      refreshStatus()
      showMessage('success', `Learning ${!learningEnabled ? 'enabled' : 'disabled'}`)
    } catch (e) {
      showMessage('error', 'Failed to toggle learning')
    } finally {
      setLoading(null)
    }
  }

  const updateLearningRate = async () => {
    setLoading('rate')
    try {
      await controlApi.setLearningRate(learningRate)
      showMessage('success', `Learning rate set to ${learningRate}`)
    } catch (e) {
      showMessage('error', 'Failed to update learning rate')
    } finally {
      setLoading(null)
    }
  }

  const changeMode = async (mode: string) => {
    setLoading('mode')
    try {
      await controlApi.setMode(mode)
      setCognitiveMode(mode)
      showMessage('success', `Mode changed to ${mode}`)
    } catch (e) {
      showMessage('error', 'Failed to change mode')
    } finally {
      setLoading(null)
    }
  }

  const saveCheckpoint = async () => {
    setLoading('save')
    try {
      await controlApi.saveCheckpoint(newCheckpointName || undefined)
      setNewCheckpointName('')
      loadCheckpoints()
      showMessage('success', 'Checkpoint saved')
    } catch (e) {
      showMessage('error', 'Failed to save checkpoint')
    } finally {
      setLoading(null)
    }
  }

  const loadCheckpoint = async (name: string) => {
    setLoading(`load-${name}`)
    try {
      await controlApi.loadCheckpoint(name)
      refreshStatus()
      showMessage('success', `Loaded checkpoint: ${name}`)
    } catch (e) {
      showMessage('error', 'Failed to load checkpoint')
    } finally {
      setLoading(null)
    }
  }

  const deleteCheckpoint = async (name: string) => {
    if (!confirm(`Delete checkpoint "${name}"?`)) return

    setLoading(`delete-${name}`)
    try {
      await controlApi.deleteCheckpoint(name)
      loadCheckpoints()
      showMessage('success', 'Checkpoint deleted')
    } catch (e) {
      showMessage('error', 'Failed to delete checkpoint')
    } finally {
      setLoading(null)
    }
  }

  const triggerThinking = async () => {
    setLoading('thinking')
    try {
      const result = await controlApi.think(thinkingTask, thinkingCycles)
      setThinkingResult(JSON.stringify(result.output, null, 2))
      showMessage('success', 'Thinking complete')
    } catch (e) {
      showMessage('error', 'Failed to trigger thinking')
    } finally {
      setLoading(null)
    }
  }

  const triggerImagination = async () => {
    setLoading('imagination')
    try {
      const result = await controlApi.imagine(imaginationSteps)
      setImaginationResult(JSON.stringify(result.output, null, 2))
      showMessage('success', 'Imagination complete')
    } catch (e) {
      showMessage('error', 'Failed to trigger imagination')
    } finally {
      setLoading(null)
    }
  }

  return (
    <div className="p-8">
      {/* Header */}
      <div className="mb-8">
        <h1 className="text-3xl font-bold text-white mb-2">Control Panel</h1>
        <p className="text-slate-400">
          Configure and control Atlas behavior - guide how Atlas learns and operates
        </p>
      </div>

      {/* Message Toast */}
      {message && (
        <div
          className={`fixed top-4 right-4 px-4 py-3 rounded-lg flex items-center gap-2 z-50 ${
            message.type === 'success'
              ? 'bg-green-500/20 text-green-400 border border-green-500/30'
              : 'bg-red-500/20 text-red-400 border border-red-500/30'
          }`}
        >
          {message.type === 'success' ? (
            <CheckCircle className="w-5 h-5" />
          ) : (
            <AlertCircle className="w-5 h-5" />
          )}
          {message.text}
        </div>
      )}

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Learning Control */}
        <div className="glass rounded-xl p-6">
          <h3 className="text-lg font-semibold text-white mb-4 flex items-center gap-2">
            <Settings className="w-5 h-5 text-atlas-primary" />
            Learning Control
          </h3>

          <div className="space-y-6">
            {/* Learning Toggle */}
            <div className="flex items-center justify-between p-4 bg-slate-800/50 rounded-lg">
              <div>
                <h4 className="text-white font-medium">Learning</h4>
                <p className="text-sm text-slate-400">
                  Enable or disable weight updates
                </p>
              </div>
              <button
                onClick={toggleLearning}
                disabled={loading === 'learning'}
                className={`px-4 py-2 rounded-lg font-medium transition-all flex items-center gap-2 ${
                  learningEnabled
                    ? 'bg-green-500/20 text-green-400 border border-green-500/30'
                    : 'bg-slate-700 text-slate-400 border border-slate-600'
                }`}
              >
                {loading === 'learning' ? (
                  <Loader2 className="w-5 h-5 animate-spin" />
                ) : learningEnabled ? (
                  <Pause className="w-5 h-5" />
                ) : (
                  <Play className="w-5 h-5" />
                )}
                {learningEnabled ? 'Pause' : 'Resume'}
              </button>
            </div>

            {/* Learning Rate */}
            <div className="p-4 bg-slate-800/50 rounded-lg">
              <div className="flex items-center justify-between mb-2">
                <h4 className="text-white font-medium">Learning Rate</h4>
                <span className="text-atlas-primary font-mono">{learningRate}</span>
              </div>
              <input
                type="range"
                min="0.001"
                max="0.1"
                step="0.001"
                value={learningRate}
                onChange={(e) => setLearningRate(parseFloat(e.target.value))}
                className="w-full"
              />
              <button
                onClick={updateLearningRate}
                disabled={loading === 'rate'}
                className="w-full mt-3 py-2 bg-atlas-primary/20 text-atlas-primary hover:bg-atlas-primary/30 rounded-lg transition-all"
              >
                {loading === 'rate' ? 'Updating...' : 'Apply'}
              </button>
            </div>
          </div>
        </div>

        {/* Cognitive Mode */}
        <div className="glass rounded-xl p-6">
          <h3 className="text-lg font-semibold text-white mb-4 flex items-center gap-2">
            <Brain className="w-5 h-5 text-purple-400" />
            Cognitive Mode
          </h3>

          <div className="grid grid-cols-2 gap-3">
            {[
              { id: 'visual', label: 'Visual Focus', icon: '👁️' },
              { id: 'audio', label: 'Audio Focus', icon: '👂' },
              { id: 'reasoning', label: 'Reasoning', icon: '🧠' },
              { id: 'creative', label: 'Creative', icon: '✨' },
              { id: 'autonomous', label: 'Autonomous', icon: '🤖' },
            ].map((mode) => (
              <button
                key={mode.id}
                onClick={() => changeMode(mode.id)}
                disabled={loading === 'mode'}
                className={`p-4 rounded-lg text-left transition-all ${
                  cognitiveMode === mode.id
                    ? 'bg-purple-500/20 border border-purple-500/30'
                    : 'bg-slate-800/50 hover:bg-slate-800'
                }`}
              >
                <span className="text-2xl mb-2 block">{mode.icon}</span>
                <span
                  className={cognitiveMode === mode.id ? 'text-purple-400' : 'text-white'}
                >
                  {mode.label}
                </span>
              </button>
            ))}
          </div>
        </div>

        {/* Checkpoints */}
        <div className="glass rounded-xl p-6">
          <h3 className="text-lg font-semibold text-white mb-4 flex items-center gap-2">
            <Save className="w-5 h-5 text-cyan-400" />
            Checkpoints
          </h3>

          {/* Save New Checkpoint */}
          <div className="flex gap-3 mb-4">
            <input
              type="text"
              value={newCheckpointName}
              onChange={(e) => setNewCheckpointName(e.target.value)}
              placeholder="Checkpoint name (optional)"
              className="flex-1 bg-slate-800 border border-slate-700 rounded-lg px-4 py-2 text-white placeholder:text-slate-500 focus:outline-none focus:border-atlas-primary"
            />
            <button
              onClick={saveCheckpoint}
              disabled={loading === 'save'}
              className="px-4 py-2 bg-cyan-500/20 text-cyan-400 hover:bg-cyan-500/30 rounded-lg transition-all flex items-center gap-2"
            >
              {loading === 'save' ? (
                <Loader2 className="w-5 h-5 animate-spin" />
              ) : (
                <Save className="w-5 h-5" />
              )}
              Save
            </button>
          </div>

          {/* Checkpoint List */}
          <div className="space-y-2 max-h-64 overflow-y-auto">
            {checkpoints.length === 0 ? (
              <div className="text-center text-slate-400 py-8">
                <Save className="w-8 h-8 mx-auto mb-2 opacity-50" />
                <p>No checkpoints saved</p>
              </div>
            ) : (
              checkpoints.map((cp) => (
                <div
                  key={cp.name}
                  className="flex items-center justify-between p-3 bg-slate-800/50 rounded-lg group"
                >
                  <div>
                    <p className="text-white font-medium">{cp.name}</p>
                    <p className="text-xs text-slate-500">
                      {new Date(cp.modified).toLocaleString()} •{' '}
                      {(cp.size_bytes / 1024 / 1024).toFixed(2)} MB
                    </p>
                  </div>
                  <div className="flex gap-2">
                    <button
                      onClick={() => loadCheckpoint(cp.name)}
                      disabled={loading?.startsWith('load')}
                      className="p-2 text-atlas-primary hover:bg-atlas-primary/20 rounded-lg transition-all"
                    >
                      {loading === `load-${cp.name}` ? (
                        <Loader2 className="w-4 h-4 animate-spin" />
                      ) : (
                        <Upload className="w-4 h-4" />
                      )}
                    </button>
                    <button
                      onClick={() => deleteCheckpoint(cp.name)}
                      disabled={loading?.startsWith('delete')}
                      className="p-2 text-red-400 hover:bg-red-500/20 rounded-lg transition-all opacity-0 group-hover:opacity-100"
                    >
                      {loading === `delete-${cp.name}` ? (
                        <Loader2 className="w-4 h-4 animate-spin" />
                      ) : (
                        <Trash2 className="w-4 h-4" />
                      )}
                    </button>
                  </div>
                </div>
              ))
            )}
          </div>
        </div>

        {/* Thinking & Imagination */}
        <div className="glass rounded-xl p-6">
          <h3 className="text-lg font-semibold text-white mb-4 flex items-center gap-2">
            <Sparkles className="w-5 h-5 text-yellow-400" />
            Cognitive Actions
          </h3>

          <div className="space-y-4">
            {/* Thinking */}
            <div className="p-4 bg-slate-800/50 rounded-lg">
              <h4 className="text-white font-medium mb-3">Thinking</h4>
              <div className="flex gap-3 mb-3">
                <select
                  value={thinkingTask}
                  onChange={(e) => setThinkingTask(e.target.value)}
                  className="flex-1 bg-slate-700 border border-slate-600 rounded-lg px-3 py-2 text-white"
                >
                  <option value="reflect">Reflect</option>
                  <option value="plan">Plan</option>
                  <option value="reason">Reason</option>
                  <option value="analyze">Analyze</option>
                </select>
                <input
                  type="number"
                  min="1"
                  max="10"
                  value={thinkingCycles}
                  onChange={(e) => setThinkingCycles(parseInt(e.target.value))}
                  className="w-20 bg-slate-700 border border-slate-600 rounded-lg px-3 py-2 text-white"
                />
                <button
                  onClick={triggerThinking}
                  disabled={loading === 'thinking'}
                  className="px-4 py-2 bg-yellow-500/20 text-yellow-400 hover:bg-yellow-500/30 rounded-lg transition-all"
                >
                  {loading === 'thinking' ? (
                    <Loader2 className="w-5 h-5 animate-spin" />
                  ) : (
                    <Brain className="w-5 h-5" />
                  )}
                </button>
              </div>
              {thinkingResult && (
                <pre className="text-xs text-slate-400 bg-slate-900 p-3 rounded-lg overflow-auto max-h-32">
                  {thinkingResult}
                </pre>
              )}
            </div>

            {/* Imagination */}
            <div className="p-4 bg-slate-800/50 rounded-lg">
              <h4 className="text-white font-medium mb-3">Imagination</h4>
              <div className="flex gap-3 mb-3">
                <div className="flex-1">
                  <label className="text-xs text-slate-400">Steps: {imaginationSteps}</label>
                  <input
                    type="range"
                    min="1"
                    max="50"
                    value={imaginationSteps}
                    onChange={(e) => setImaginationSteps(parseInt(e.target.value))}
                    className="w-full"
                  />
                </div>
                <button
                  onClick={triggerImagination}
                  disabled={loading === 'imagination'}
                  className="px-4 py-2 bg-purple-500/20 text-purple-400 hover:bg-purple-500/30 rounded-lg transition-all"
                >
                  {loading === 'imagination' ? (
                    <Loader2 className="w-5 h-5 animate-spin" />
                  ) : (
                    <Sparkles className="w-5 h-5" />
                  )}
                </button>
              </div>
              {imaginationResult && (
                <pre className="text-xs text-slate-400 bg-slate-900 p-3 rounded-lg overflow-auto max-h-32">
                  {imaginationResult}
                </pre>
              )}
            </div>
          </div>
        </div>
      </div>
    </div>
  )
}

export default Control
