import { useState, useEffect } from 'react'
import { systemApi } from '../services/api'
import type { ArchitectureInfo } from '../types/atlas'
import NeuralViz from '../components/NeuralViz'
import {
  Network,
  Eye,
  Ear,
  Brain,
  Layers,
  Activity,
  Loader2,
  RefreshCw,
} from 'lucide-react'

function Architecture() {
  const [architecture, setArchitecture] = useState<ArchitectureInfo | null>(null)
  const [loading, setLoading] = useState(true)

  const loadArchitecture = async () => {
    setLoading(true)
    try {
      const data = await systemApi.getArchitecture()
      setArchitecture(data)
    } catch (e) {
      console.error('Error loading architecture:', e)
    } finally {
      setLoading(false)
    }
  }

  useEffect(() => {
    loadArchitecture()
  }, [])

  const visualLayers = architecture?.visual?.layers || [
    { name: 'V1', neurons: 200, type: 'edge_detectors' },
    { name: 'V2', neurons: 100, type: 'texture_patterns' },
    { name: 'V3', neurons: 50, type: 'object_features' },
  ]

  const audioLayers = architecture?.audio?.layers || [
    { name: 'A1', neurons: 150, type: 'frequency_bands' },
    { name: 'A2', neurons: 75, type: 'spectral_patterns' },
    { name: 'A3', neurons: 40, type: 'sound_objects' },
  ]

  const totalVisualNeurons = visualLayers.reduce((sum, l) => sum + l.neurons, 0)
  const totalAudioNeurons = audioLayers.reduce((sum, l) => sum + l.neurons, 0)
  const multimodalSize = architecture?.multimodal?.size || 100

  return (
    <div className="p-8">
      {/* Header */}
      <div className="mb-8 flex items-center justify-between">
        <div>
          <h1 className="text-3xl font-bold text-white mb-2">Neural Architecture</h1>
          <p className="text-slate-400">
            Explore the structure of Atlas's neural pathways and processing systems
          </p>
        </div>
        <button
          onClick={loadArchitecture}
          disabled={loading}
          className="px-4 py-2 bg-atlas-primary/20 text-atlas-primary hover:bg-atlas-primary/30 rounded-lg transition-all flex items-center gap-2"
        >
          {loading ? (
            <Loader2 className="w-5 h-5 animate-spin" />
          ) : (
            <RefreshCw className="w-5 h-5" />
          )}
          Refresh
        </button>
      </div>

      {/* Overview Stats */}
      <div className="grid grid-cols-1 md:grid-cols-4 gap-4 mb-8">
        <div className="glass rounded-xl p-4">
          <div className="flex items-center gap-3">
            <div className="p-3 bg-atlas-primary/20 rounded-lg">
              <Network className="w-6 h-6 text-atlas-primary" />
            </div>
            <div>
              <p className="text-2xl font-bold text-white">
                {totalVisualNeurons + totalAudioNeurons + multimodalSize}
              </p>
              <p className="text-sm text-slate-400">Total Neurons</p>
            </div>
          </div>
        </div>
        <div className="glass rounded-xl p-4">
          <div className="flex items-center gap-3">
            <div className="p-3 bg-blue-500/20 rounded-lg">
              <Eye className="w-6 h-6 text-blue-400" />
            </div>
            <div>
              <p className="text-2xl font-bold text-white">{totalVisualNeurons}</p>
              <p className="text-sm text-slate-400">Visual Neurons</p>
            </div>
          </div>
        </div>
        <div className="glass rounded-xl p-4">
          <div className="flex items-center gap-3">
            <div className="p-3 bg-purple-500/20 rounded-lg">
              <Ear className="w-6 h-6 text-purple-400" />
            </div>
            <div>
              <p className="text-2xl font-bold text-white">{totalAudioNeurons}</p>
              <p className="text-sm text-slate-400">Audio Neurons</p>
            </div>
          </div>
        </div>
        <div className="glass rounded-xl p-4">
          <div className="flex items-center gap-3">
            <div className="p-3 bg-cyan-500/20 rounded-lg">
              <Brain className="w-6 h-6 text-cyan-400" />
            </div>
            <div>
              <p className="text-2xl font-bold text-white">{multimodalSize}</p>
              <p className="text-sm text-slate-400">Multimodal Neurons</p>
            </div>
          </div>
        </div>
      </div>

      {/* Architecture Diagram */}
      <div className="glass rounded-xl p-6 mb-8">
        <h3 className="text-lg font-semibold text-white mb-6">Processing Pipeline</h3>
        <div className="relative">
          {/* Visual Pathway */}
          <div className="mb-8">
            <div className="flex items-center gap-2 mb-4">
              <Eye className="w-5 h-5 text-atlas-primary" />
              <h4 className="text-white font-medium">Visual Pathway</h4>
            </div>
            <div className="flex items-center gap-4">
              <div className="glass p-4 rounded-lg text-center">
                <p className="text-sm text-slate-400">Input</p>
                <p className="text-white font-mono">64x64</p>
              </div>
              <div className="flex-1 flex items-center gap-2">
                {visualLayers.map((layer, i) => (
                  <div key={layer.name} className="flex-1 flex items-center">
                    <div className="flex-1 h-px bg-gradient-to-r from-atlas-primary/50 to-atlas-primary" />
                    <div className="glass p-4 rounded-lg text-center border border-atlas-primary/30">
                      <p className="text-atlas-primary font-bold">{layer.name}</p>
                      <p className="text-white text-lg">{layer.neurons}</p>
                      <p className="text-xs text-slate-400">{layer.type.replace('_', ' ')}</p>
                    </div>
                    {i < visualLayers.length - 1 && (
                      <div className="flex-1 h-px bg-gradient-to-r from-atlas-primary to-atlas-primary/50" />
                    )}
                  </div>
                ))}
              </div>
            </div>
          </div>

          {/* Audio Pathway */}
          <div className="mb-8">
            <div className="flex items-center gap-2 mb-4">
              <Ear className="w-5 h-5 text-purple-400" />
              <h4 className="text-white font-medium">Audio Pathway</h4>
            </div>
            <div className="flex items-center gap-4">
              <div className="glass p-4 rounded-lg text-center">
                <p className="text-sm text-slate-400">Input</p>
                <p className="text-white font-mono">64 Mel</p>
              </div>
              <div className="flex-1 flex items-center gap-2">
                {audioLayers.map((layer, i) => (
                  <div key={layer.name} className="flex-1 flex items-center">
                    <div className="flex-1 h-px bg-gradient-to-r from-purple-500/50 to-purple-500" />
                    <div className="glass p-4 rounded-lg text-center border border-purple-500/30">
                      <p className="text-purple-400 font-bold">{layer.name}</p>
                      <p className="text-white text-lg">{layer.neurons}</p>
                      <p className="text-xs text-slate-400">{layer.type.replace('_', ' ')}</p>
                    </div>
                    {i < audioLayers.length - 1 && (
                      <div className="flex-1 h-px bg-gradient-to-r from-purple-500 to-purple-500/50" />
                    )}
                  </div>
                ))}
              </div>
            </div>
          </div>

          {/* Multimodal Integration */}
          <div className="text-center">
            <div className="inline-block glass p-6 rounded-xl border border-cyan-500/30">
              <Brain className="w-8 h-8 text-cyan-400 mx-auto mb-2" />
              <p className="text-cyan-400 font-bold">Multimodal Integration</p>
              <p className="text-white text-2xl font-bold">{multimodalSize}</p>
              <p className="text-sm text-slate-400">neurons</p>
              {architecture?.multimodal?.associations && (
                <p className="text-xs text-slate-500 mt-2">
                  {architecture.multimodal.associations} associations
                </p>
              )}
            </div>
          </div>
        </div>
      </div>

      {/* Detailed Views */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6 mb-8">
        <div className="glass rounded-xl p-6">
          <h3 className="text-lg font-semibold text-white mb-4 flex items-center gap-2">
            <Eye className="w-5 h-5 text-atlas-primary" />
            Visual Pathway Details
          </h3>
          <NeuralViz layers={visualLayers} type="visual" />
          <div className="mt-4 space-y-2">
            {visualLayers.map((layer) => (
              <div key={layer.name} className="flex justify-between text-sm">
                <span className="text-slate-400">{layer.name}</span>
                <span className="text-white">{layer.neurons} neurons</span>
              </div>
            ))}
          </div>
        </div>

        <div className="glass rounded-xl p-6">
          <h3 className="text-lg font-semibold text-white mb-4 flex items-center gap-2">
            <Ear className="w-5 h-5 text-purple-400" />
            Audio Pathway Details
          </h3>
          <NeuralViz layers={audioLayers} type="audio" />
          <div className="mt-4 space-y-2">
            {audioLayers.map((layer) => (
              <div key={layer.name} className="flex justify-between text-sm">
                <span className="text-slate-400">{layer.name}</span>
                <span className="text-white">{layer.neurons} neurons</span>
              </div>
            ))}
          </div>
        </div>

        <div className="glass rounded-xl p-6">
          <h3 className="text-lg font-semibold text-white mb-4 flex items-center gap-2">
            <Brain className="w-5 h-5 text-cyan-400" />
            Multimodal Integration
          </h3>
          <NeuralViz
            layers={[
              { name: 'V3', neurons: visualLayers[visualLayers.length - 1]?.neurons || 50 },
              { name: 'MM', neurons: multimodalSize },
              { name: 'A3', neurons: audioLayers[audioLayers.length - 1]?.neurons || 40 },
            ]}
            type="multimodal"
          />
          <div className="mt-4 space-y-2 text-sm">
            <div className="flex justify-between">
              <span className="text-slate-400">Integration Layer</span>
              <span className="text-white">{multimodalSize} neurons</span>
            </div>
            <div className="flex justify-between">
              <span className="text-slate-400">Associations</span>
              <span className="text-white">{architecture?.multimodal?.associations || 150}</span>
            </div>
          </div>
        </div>
      </div>

      {/* Architecture Description */}
      <div className="glass rounded-xl p-6">
        <h3 className="text-lg font-semibold text-white mb-4 flex items-center gap-2">
          <Layers className="w-5 h-5 text-yellow-400" />
          Architecture Overview
        </h3>
        <div className="grid grid-cols-1 md:grid-cols-2 gap-6 text-sm text-slate-400">
          <div>
            <h4 className="text-white font-medium mb-2">Visual Processing (V1 → V2 → V3)</h4>
            <p>
              The visual pathway processes input images through three hierarchical layers:
            </p>
            <ul className="list-disc list-inside mt-2 space-y-1">
              <li><strong className="text-atlas-primary">V1:</strong> Edge detectors and color patches (low-level features)</li>
              <li><strong className="text-atlas-primary">V2:</strong> Texture and shape patterns (mid-level features)</li>
              <li><strong className="text-atlas-primary">V3:</strong> Object-like representations (high-level features)</li>
            </ul>
          </div>
          <div>
            <h4 className="text-white font-medium mb-2">Audio Processing (A1 → A2 → A3)</h4>
            <p>
              The auditory pathway processes mel-spectrograms through three layers:
            </p>
            <ul className="list-disc list-inside mt-2 space-y-1">
              <li><strong className="text-purple-400">A1:</strong> Frequency band detectors (cochlear-like)</li>
              <li><strong className="text-purple-400">A2:</strong> Spectral patterns and harmonics</li>
              <li><strong className="text-purple-400">A3:</strong> Sound objects and events</li>
            </ul>
          </div>
          <div className="md:col-span-2">
            <h4 className="text-white font-medium mb-2">Multimodal Integration</h4>
            <p>
              The multimodal layer integrates visual and auditory representations using Hebbian
              learning. When visual and audio patterns co-occur, they form associations. This enables:
            </p>
            <ul className="list-disc list-inside mt-2 space-y-1">
              <li><strong className="text-cyan-400">Cross-modal prediction:</strong> Predict audio from visual or visual from audio</li>
              <li><strong className="text-cyan-400">Temporal binding:</strong> Associate sequences of events across modalities</li>
              <li><strong className="text-cyan-400">Emergent concepts:</strong> Higher-order representations that span both modalities</li>
            </ul>
          </div>
        </div>
      </div>
    </div>
  )
}

export default Architecture
