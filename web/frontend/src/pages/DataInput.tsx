import { useState, useRef, useCallback } from 'react'
import { useAtlas } from '../hooks/useAtlas'
import { dataApi } from '../services/api'
import type { ProcessingResult } from '../types/atlas'
import {
  Upload,
  Camera,
  Mic,
  Play,
  Pause,
  Image as ImageIcon,
  FileAudio,
  CheckCircle,
  XCircle,
  Loader2,
} from 'lucide-react'

function DataInput() {
  const { status } = useAtlas()
  const [activeTab, setActiveTab] = useState<'image' | 'audio' | 'webcam'>('image')
  const [isProcessing, setIsProcessing] = useState(false)
  const [result, setResult] = useState<ProcessingResult | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [learnEnabled, setLearnEnabled] = useState(true)

  // Image upload state
  const [selectedImage, setSelectedImage] = useState<string | null>(null)
  const imageInputRef = useRef<HTMLInputElement>(null)

  // Webcam state
  const [webcamActive, setWebcamActive] = useState(false)
  const videoRef = useRef<HTMLVideoElement>(null)
  const streamRef = useRef<MediaStream | null>(null)

  // Audio state
  const [selectedAudio, setSelectedAudio] = useState<File | null>(null)
  const audioInputRef = useRef<HTMLInputElement>(null)

  const handleImageSelect = useCallback((e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0]
    if (file) {
      const reader = new FileReader()
      reader.onload = () => {
        setSelectedImage(reader.result as string)
        setResult(null)
        setError(null)
      }
      reader.readAsDataURL(file)
    }
  }, [])

  const handleImageUpload = async () => {
    if (!selectedImage) return

    setIsProcessing(true)
    setError(null)

    try {
      // Extract base64 data (remove data URL prefix)
      const base64Data = selectedImage.split(',')[1]
      const result = await dataApi.processFrame(base64Data, learnEnabled)
      setResult(result)
    } catch (e) {
      setError(e instanceof Error ? e.message : 'Failed to process image')
    } finally {
      setIsProcessing(false)
    }
  }

  const startWebcam = async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ video: true })
      if (videoRef.current) {
        videoRef.current.srcObject = stream
        streamRef.current = stream
        setWebcamActive(true)
      }
    } catch (e) {
      setError('Failed to access webcam')
    }
  }

  const stopWebcam = () => {
    if (streamRef.current) {
      streamRef.current.getTracks().forEach((track) => track.stop())
      streamRef.current = null
      setWebcamActive(false)
    }
  }

  const captureFrame = async () => {
    if (!videoRef.current) return

    const canvas = document.createElement('canvas')
    canvas.width = videoRef.current.videoWidth
    canvas.height = videoRef.current.videoHeight
    const ctx = canvas.getContext('2d')
    if (!ctx) return

    ctx.drawImage(videoRef.current, 0, 0)
    const base64Data = canvas.toDataURL('image/jpeg').split(',')[1]

    setIsProcessing(true)
    setError(null)

    try {
      const result = await dataApi.processFrame(base64Data, learnEnabled)
      setResult(result)
    } catch (e) {
      setError(e instanceof Error ? e.message : 'Failed to process frame')
    } finally {
      setIsProcessing(false)
    }
  }

  const handleAudioSelect = useCallback((e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0]
    if (file) {
      setSelectedAudio(file)
      setResult(null)
      setError(null)
    }
  }, [])

  const handleAudioUpload = async () => {
    if (!selectedAudio) return

    setIsProcessing(true)
    setError(null)

    try {
      // Read file as array buffer
      const arrayBuffer = await selectedAudio.arrayBuffer()
      const base64Data = btoa(
        new Uint8Array(arrayBuffer).reduce(
          (data, byte) => data + String.fromCharCode(byte),
          ''
        )
      )
      const result = await dataApi.processAudio(base64Data, 22050, learnEnabled)
      setResult(result)
    } catch (e) {
      setError(e instanceof Error ? e.message : 'Failed to process audio')
    } finally {
      setIsProcessing(false)
    }
  }

  return (
    <div className="p-8">
      {/* Header */}
      <div className="mb-8">
        <h1 className="text-3xl font-bold text-white mb-2">Data Input</h1>
        <p className="text-slate-400">
          Feed sensory data to Atlas - this is how the world interfaces with Atlas
        </p>
      </div>

      {/* Learning Toggle */}
      <div className="glass rounded-xl p-4 mb-6 flex items-center justify-between">
        <div>
          <h3 className="text-white font-medium">Learning Mode</h3>
          <p className="text-sm text-slate-400">
            When enabled, Atlas will learn from the input data
          </p>
        </div>
        <button
          onClick={() => setLearnEnabled(!learnEnabled)}
          className={`px-4 py-2 rounded-lg font-medium transition-all ${
            learnEnabled
              ? 'bg-green-500/20 text-green-400 border border-green-500/30'
              : 'bg-slate-700 text-slate-400 border border-slate-600'
          }`}
        >
          {learnEnabled ? 'Learning ON' : 'Learning OFF'}
        </button>
      </div>

      {/* Tab Navigation */}
      <div className="flex gap-4 mb-6">
        {[
          { id: 'image' as const, icon: ImageIcon, label: 'Image Upload' },
          { id: 'webcam' as const, icon: Camera, label: 'Webcam' },
          { id: 'audio' as const, icon: FileAudio, label: 'Audio Upload' },
        ].map((tab) => (
          <button
            key={tab.id}
            onClick={() => setActiveTab(tab.id)}
            className={`flex items-center gap-2 px-4 py-2 rounded-lg transition-all ${
              activeTab === tab.id
                ? 'bg-atlas-primary/20 text-atlas-primary border border-atlas-primary/30'
                : 'bg-slate-800 text-slate-400 hover:text-white'
            }`}
          >
            <tab.icon className="w-5 h-5" />
            {tab.label}
          </button>
        ))}
      </div>

      {/* Input Panels */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Input Area */}
        <div className="glass rounded-xl p-6">
          {activeTab === 'image' && (
            <div>
              <h3 className="text-lg font-semibold text-white mb-4">Upload Image</h3>
              <div
                onClick={() => imageInputRef.current?.click()}
                className={`border-2 border-dashed rounded-xl p-8 text-center cursor-pointer transition-all ${
                  selectedImage
                    ? 'border-atlas-primary/50 bg-atlas-primary/5'
                    : 'border-slate-600 hover:border-atlas-primary/50'
                }`}
              >
                {selectedImage ? (
                  <img
                    src={selectedImage}
                    alt="Selected"
                    className="max-h-64 mx-auto rounded-lg"
                  />
                ) : (
                  <div className="text-slate-400">
                    <Upload className="w-12 h-12 mx-auto mb-4" />
                    <p>Click or drag to upload an image</p>
                    <p className="text-sm text-slate-500 mt-2">
                      PNG, JPEG, GIF supported
                    </p>
                  </div>
                )}
              </div>
              <input
                ref={imageInputRef}
                type="file"
                accept="image/*"
                onChange={handleImageSelect}
                className="hidden"
              />
              {selectedImage && (
                <button
                  onClick={handleImageUpload}
                  disabled={isProcessing}
                  className="w-full mt-4 py-3 bg-atlas-primary hover:bg-atlas-primary/80 text-white rounded-lg font-medium transition-all disabled:opacity-50 flex items-center justify-center gap-2"
                >
                  {isProcessing ? (
                    <>
                      <Loader2 className="w-5 h-5 animate-spin" />
                      Processing...
                    </>
                  ) : (
                    <>
                      <Upload className="w-5 h-5" />
                      Send to Atlas
                    </>
                  )}
                </button>
              )}
            </div>
          )}

          {activeTab === 'webcam' && (
            <div>
              <h3 className="text-lg font-semibold text-white mb-4">Webcam Feed</h3>
              <div className="bg-slate-800 rounded-xl overflow-hidden mb-4">
                <video
                  ref={videoRef}
                  autoPlay
                  playsInline
                  muted
                  className={`w-full ${webcamActive ? '' : 'hidden'}`}
                />
                {!webcamActive && (
                  <div className="h-64 flex items-center justify-center text-slate-400">
                    <Camera className="w-12 h-12" />
                  </div>
                )}
              </div>
              <div className="flex gap-4">
                {!webcamActive ? (
                  <button
                    onClick={startWebcam}
                    className="flex-1 py-3 bg-green-500/20 text-green-400 hover:bg-green-500/30 rounded-lg font-medium transition-all flex items-center justify-center gap-2"
                  >
                    <Play className="w-5 h-5" />
                    Start Webcam
                  </button>
                ) : (
                  <>
                    <button
                      onClick={stopWebcam}
                      className="flex-1 py-3 bg-red-500/20 text-red-400 hover:bg-red-500/30 rounded-lg font-medium transition-all flex items-center justify-center gap-2"
                    >
                      <Pause className="w-5 h-5" />
                      Stop
                    </button>
                    <button
                      onClick={captureFrame}
                      disabled={isProcessing}
                      className="flex-1 py-3 bg-atlas-primary hover:bg-atlas-primary/80 text-white rounded-lg font-medium transition-all disabled:opacity-50 flex items-center justify-center gap-2"
                    >
                      {isProcessing ? (
                        <Loader2 className="w-5 h-5 animate-spin" />
                      ) : (
                        <Camera className="w-5 h-5" />
                      )}
                      Capture Frame
                    </button>
                  </>
                )}
              </div>
            </div>
          )}

          {activeTab === 'audio' && (
            <div>
              <h3 className="text-lg font-semibold text-white mb-4">Upload Audio</h3>
              <div
                onClick={() => audioInputRef.current?.click()}
                className={`border-2 border-dashed rounded-xl p-8 text-center cursor-pointer transition-all ${
                  selectedAudio
                    ? 'border-purple-500/50 bg-purple-500/5'
                    : 'border-slate-600 hover:border-purple-500/50'
                }`}
              >
                {selectedAudio ? (
                  <div className="text-purple-400">
                    <FileAudio className="w-12 h-12 mx-auto mb-4" />
                    <p>{selectedAudio.name}</p>
                    <p className="text-sm text-slate-500 mt-2">
                      {(selectedAudio.size / 1024).toFixed(2)} KB
                    </p>
                  </div>
                ) : (
                  <div className="text-slate-400">
                    <Mic className="w-12 h-12 mx-auto mb-4" />
                    <p>Click or drag to upload audio</p>
                    <p className="text-sm text-slate-500 mt-2">WAV, PCM supported</p>
                  </div>
                )}
              </div>
              <input
                ref={audioInputRef}
                type="file"
                accept="audio/*"
                onChange={handleAudioSelect}
                className="hidden"
              />
              {selectedAudio && (
                <button
                  onClick={handleAudioUpload}
                  disabled={isProcessing}
                  className="w-full mt-4 py-3 bg-purple-500 hover:bg-purple-500/80 text-white rounded-lg font-medium transition-all disabled:opacity-50 flex items-center justify-center gap-2"
                >
                  {isProcessing ? (
                    <>
                      <Loader2 className="w-5 h-5 animate-spin" />
                      Processing...
                    </>
                  ) : (
                    <>
                      <Upload className="w-5 h-5" />
                      Send to Atlas
                    </>
                  )}
                </button>
              )}
            </div>
          )}
        </div>

        {/* Results Panel */}
        <div className="glass rounded-xl p-6">
          <h3 className="text-lg font-semibold text-white mb-4">Processing Result</h3>

          {error && (
            <div className="bg-red-500/10 border border-red-500/30 rounded-lg p-4 mb-4">
              <div className="flex items-center gap-2 text-red-400">
                <XCircle className="w-5 h-5" />
                <span>{error}</span>
              </div>
            </div>
          )}

          {result && (
            <div className="space-y-4">
              <div className="flex items-center gap-2 text-green-400 bg-green-500/10 rounded-lg p-4">
                <CheckCircle className="w-5 h-5" />
                <span>Successfully processed!</span>
              </div>

              <div className="bg-slate-800/50 rounded-lg p-4">
                <h4 className="text-sm text-slate-400 mb-2">Details</h4>
                <div className="space-y-2 text-sm">
                  {result.frame_number && (
                    <div className="flex justify-between">
                      <span className="text-slate-400">Frame Number:</span>
                      <span className="text-white">{result.frame_number}</span>
                    </div>
                  )}
                  {result.chunk_number && (
                    <div className="flex justify-between">
                      <span className="text-slate-400">Chunk Number:</span>
                      <span className="text-white">{result.chunk_number}</span>
                    </div>
                  )}
                  <div className="flex justify-between">
                    <span className="text-slate-400">Timestamp:</span>
                    <span className="text-white">
                      {new Date(result.timestamp).toLocaleString()}
                    </span>
                  </div>
                </div>
              </div>

              {result.predictions && (
                <div className="bg-slate-800/50 rounded-lg p-4">
                  <h4 className="text-sm text-slate-400 mb-2">Predictions</h4>
                  {result.predictions.temporal && (
                    <div className="mb-2">
                      <span className="text-xs text-slate-500">
                        Temporal ({result.predictions.temporal.length} steps)
                      </span>
                      <div className="flex gap-1 mt-1">
                        {result.predictions.temporal.slice(0, 5).map((step, i) => (
                          <div
                            key={i}
                            className="flex-1 h-8 bg-atlas-primary/20 rounded flex items-center justify-center text-xs text-atlas-primary"
                          >
                            t+{i + 1}
                          </div>
                        ))}
                      </div>
                    </div>
                  )}
                  {result.predictions.cross_modal && (
                    <div>
                      <span className="text-xs text-slate-500">Cross-Modal</span>
                      <div className="flex gap-1 mt-1">
                        {result.predictions.cross_modal.slice(0, 8).map((val, i) => (
                          <div
                            key={i}
                            className="flex-1 h-8 bg-purple-500/20 rounded"
                            style={{
                              opacity: 0.3 + Math.abs(val) * 0.7,
                            }}
                          />
                        ))}
                      </div>
                    </div>
                  )}
                </div>
              )}
            </div>
          )}

          {!result && !error && (
            <div className="text-center text-slate-400 py-12">
              <Upload className="w-12 h-12 mx-auto mb-4 opacity-50" />
              <p>Upload or capture data to see results</p>
            </div>
          )}
        </div>
      </div>

      {/* Info Box */}
      <div className="glass rounded-xl p-6 mt-6">
        <h3 className="text-lg font-semibold text-white mb-4">About Data Input</h3>
        <div className="grid grid-cols-1 md:grid-cols-3 gap-6 text-sm text-slate-400">
          <div>
            <h4 className="text-white font-medium mb-2">Visual Processing</h4>
            <p>
              Images are processed through the visual pathway (V1 → V2 → V3),
              extracting features from edges to complex patterns.
            </p>
          </div>
          <div>
            <h4 className="text-white font-medium mb-2">Audio Processing</h4>
            <p>
              Audio is converted to mel-spectrograms and processed through the
              auditory pathway (A1 → A2 → A3).
            </p>
          </div>
          <div>
            <h4 className="text-white font-medium mb-2">Cross-Modal Learning</h4>
            <p>
              When learning is enabled, Atlas forms associations between visual
              and audio patterns that occur together.
            </p>
          </div>
        </div>
      </div>
    </div>
  )
}

export default DataInput
