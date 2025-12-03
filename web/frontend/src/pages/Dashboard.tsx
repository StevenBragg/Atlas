import { useEffect, useState } from 'react'
import { useAtlas } from '../hooks/useAtlas'
import { systemApi } from '../services/api'
import type { ArchitectureInfo } from '../types/atlas'
import MetricCard from '../components/MetricCard'
import NeuralViz from '../components/NeuralViz'
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
} from 'recharts'
import {
  Eye,
  Ear,
  Brain,
  Zap,
  Clock,
  Activity,
  Layers,
  Link2,
} from 'lucide-react'

function Dashboard() {
  const { status, metrics } = useAtlas()
  const [architecture, setArchitecture] = useState<ArchitectureInfo | null>(null)
  const [metricsHistory, setMetricsHistory] = useState<Array<{
    time: string
    frames: number
    error: number
  }>>([])

  useEffect(() => {
    systemApi.getArchitecture().then(setArchitecture).catch(console.error)
  }, [])

  // Track metrics history
  useEffect(() => {
    if (metrics) {
      setMetricsHistory((prev) => {
        const newEntry = {
          time: new Date().toLocaleTimeString(),
          frames: metrics.frames_processed,
          error: metrics.prediction_error || 0,
        }
        const updated = [...prev, newEntry].slice(-20) // Keep last 20 entries
        return updated
      })
    }
  }, [metrics])

  const formatUptime = (seconds: number) => {
    const hours = Math.floor(seconds / 3600)
    const minutes = Math.floor((seconds % 3600) / 60)
    const secs = Math.floor(seconds % 60)
    return `${hours}h ${minutes}m ${secs}s`
  }

  return (
    <div className="p-8">
      {/* Header */}
      <div className="mb-8">
        <h1 className="text-3xl font-bold text-white mb-2">Atlas Dashboard</h1>
        <p className="text-slate-400">
          Real-time monitoring of the self-organizing audio-visual learning system
        </p>
      </div>

      {/* Main Metrics Grid */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6 mb-8">
        <MetricCard
          title="Frames Processed"
          value={metrics?.frames_processed?.toLocaleString() || '0'}
          subtitle="Visual inputs"
          icon={<Eye className="w-6 h-6" />}
          color="primary"
        />
        <MetricCard
          title="Audio Chunks"
          value={metrics?.audio_chunks_processed?.toLocaleString() || '0'}
          subtitle="Audio inputs"
          icon={<Ear className="w-6 h-6" />}
          color="secondary"
        />
        <MetricCard
          title="Total Neurons"
          value={metrics?.total_neurons?.toLocaleString() || '350'}
          subtitle="Across all layers"
          icon={<Brain className="w-6 h-6" />}
          color="accent"
        />
        <MetricCard
          title="Uptime"
          value={formatUptime(metrics?.uptime_seconds || 0)}
          subtitle="Since last restart"
          icon={<Clock className="w-6 h-6" />}
          color="success"
        />
      </div>

      {/* Learning Metrics */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6 mb-8">
        <MetricCard
          title="Prediction Error"
          value={(metrics?.prediction_error || 0).toFixed(4)}
          icon={<Zap className="w-6 h-6" />}
          trend={(metrics?.prediction_error || 0) < 0.5 ? 'down' : 'up'}
          trendValue="Lower is better"
          color={(metrics?.prediction_error || 0) < 0.5 ? 'success' : 'warning'}
        />
        <MetricCard
          title="Cross-Modal Correlation"
          value={((metrics?.cross_modal_correlation || 0) * 100).toFixed(1) + '%'}
          icon={<Link2 className="w-6 h-6" />}
          trend={(metrics?.cross_modal_correlation || 0) > 0.5 ? 'up' : 'neutral'}
          trendValue="Audio-visual binding"
          color={(metrics?.cross_modal_correlation || 0) > 0.5 ? 'success' : 'primary'}
        />
        <MetricCard
          title="Active Associations"
          value={metrics?.active_associations?.toLocaleString() || '150'}
          icon={<Layers className="w-6 h-6" />}
          color="accent"
        />
      </div>

      {/* Charts and Visualizations */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6 mb-8">
        {/* Processing Activity Chart */}
        <div className="glass rounded-xl p-6">
          <h3 className="text-lg font-semibold text-white mb-4 flex items-center gap-2">
            <Activity className="w-5 h-5 text-atlas-primary" />
            Processing Activity
          </h3>
          <div className="h-64">
            <ResponsiveContainer width="100%" height="100%">
              <LineChart data={metricsHistory}>
                <CartesianGrid strokeDasharray="3 3" stroke="#334155" />
                <XAxis dataKey="time" stroke="#64748b" fontSize={12} />
                <YAxis stroke="#64748b" fontSize={12} />
                <Tooltip
                  contentStyle={{
                    backgroundColor: '#1e293b',
                    border: '1px solid #334155',
                    borderRadius: '8px',
                  }}
                />
                <Line
                  type="monotone"
                  dataKey="frames"
                  stroke="#6366f1"
                  strokeWidth={2}
                  dot={false}
                />
              </LineChart>
            </ResponsiveContainer>
          </div>
        </div>

        {/* System Status */}
        <div className="glass rounded-xl p-6">
          <h3 className="text-lg font-semibold text-white mb-4">System Status</h3>
          <div className="space-y-4">
            <div className="flex items-center justify-between p-3 bg-slate-800/50 rounded-lg">
              <span className="text-slate-300">Atlas Core</span>
              <span
                className={`px-3 py-1 rounded-full text-sm ${
                  status?.atlas_available
                    ? 'bg-green-500/20 text-green-400'
                    : 'bg-yellow-500/20 text-yellow-400'
                }`}
              >
                {status?.atlas_available ? 'Available' : 'Demo Mode'}
              </span>
            </div>
            <div className="flex items-center justify-between p-3 bg-slate-800/50 rounded-lg">
              <span className="text-slate-300">Learning</span>
              <span
                className={`px-3 py-1 rounded-full text-sm ${
                  status?.learning_enabled
                    ? 'bg-green-500/20 text-green-400'
                    : 'bg-slate-500/20 text-slate-400'
                }`}
              >
                {status?.learning_enabled ? 'Active' : 'Paused'}
              </span>
            </div>
            <div className="flex items-center justify-between p-3 bg-slate-800/50 rounded-lg">
              <span className="text-slate-300">Initialization</span>
              <span
                className={`px-3 py-1 rounded-full text-sm ${
                  status?.initialized
                    ? 'bg-green-500/20 text-green-400'
                    : 'bg-red-500/20 text-red-400'
                }`}
              >
                {status?.initialized ? 'Complete' : 'Pending'}
              </span>
            </div>
            <div className="flex items-center justify-between p-3 bg-slate-800/50 rounded-lg">
              <span className="text-slate-300">Last Activity</span>
              <span className="text-slate-400 text-sm">
                {status?.stats?.last_activity
                  ? new Date(status.stats.last_activity).toLocaleString()
                  : 'N/A'}
              </span>
            </div>
          </div>
        </div>
      </div>

      {/* Neural Network Visualizations */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        <div className="glass rounded-xl p-6">
          <h3 className="text-lg font-semibold text-white mb-4 flex items-center gap-2">
            <Eye className="w-5 h-5 text-atlas-primary" />
            Visual Pathway
          </h3>
          <NeuralViz
            layers={
              architecture?.visual?.layers || [
                { name: 'V1', neurons: 200 },
                { name: 'V2', neurons: 100 },
                { name: 'V3', neurons: 50 },
              ]
            }
            type="visual"
          />
        </div>

        <div className="glass rounded-xl p-6">
          <h3 className="text-lg font-semibold text-white mb-4 flex items-center gap-2">
            <Ear className="w-5 h-5 text-purple-400" />
            Audio Pathway
          </h3>
          <NeuralViz
            layers={
              architecture?.audio?.layers || [
                { name: 'A1', neurons: 150 },
                { name: 'A2', neurons: 75 },
                { name: 'A3', neurons: 40 },
              ]
            }
            type="audio"
          />
        </div>

        <div className="glass rounded-xl p-6">
          <h3 className="text-lg font-semibold text-white mb-4 flex items-center gap-2">
            <Brain className="w-5 h-5 text-cyan-400" />
            Multimodal Integration
          </h3>
          <NeuralViz
            layers={[
              { name: 'Visual', neurons: 50 },
              { name: 'Fusion', neurons: architecture?.multimodal?.size || 100 },
              { name: 'Audio', neurons: 40 },
            ]}
            type="multimodal"
          />
        </div>
      </div>
    </div>
  )
}

export default Dashboard
