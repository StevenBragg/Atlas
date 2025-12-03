import { useState, useEffect } from 'react'
import { memoryApi } from '../services/api'
import type { MemoryContents, MemoryItem } from '../types/atlas'
import {
  Database,
  Brain,
  Clock,
  Search,
  Link2,
  Layers,
  ChevronRight,
  Loader2,
} from 'lucide-react'

function Memory() {
  const [activeTab, setActiveTab] = useState<'episodic' | 'semantic' | 'working' | 'associations'>('episodic')
  const [episodic, setEpisodic] = useState<MemoryContents | null>(null)
  const [semantic, setSemantic] = useState<MemoryContents | null>(null)
  const [working, setWorking] = useState<MemoryContents | null>(null)
  const [associations, setAssociations] = useState<{
    total_associations: number
    top_associations: Array<{ visual_pattern: string; audio_pattern: string; strength: number }>
  } | null>(null)
  const [loading, setLoading] = useState(false)
  const [searchQuery, setSearchQuery] = useState('')
  const [searchResults, setSearchResults] = useState<MemoryItem[]>([])

  useEffect(() => {
    loadMemoryData()
  }, [activeTab])

  const loadMemoryData = async () => {
    setLoading(true)
    try {
      if (activeTab === 'episodic' && !episodic) {
        setEpisodic(await memoryApi.getEpisodic())
      } else if (activeTab === 'semantic' && !semantic) {
        setSemantic(await memoryApi.getSemantic())
      } else if (activeTab === 'working') {
        setWorking(await memoryApi.getWorking())
      } else if (activeTab === 'associations' && !associations) {
        setAssociations(await memoryApi.getAssociations())
      }
    } catch (e) {
      console.error('Error loading memory data:', e)
    } finally {
      setLoading(false)
    }
  }

  const handleSearch = async () => {
    if (!searchQuery.trim()) return

    setLoading(true)
    try {
      const result = await memoryApi.search(searchQuery)
      setSearchResults(result.results || [])
    } catch (e) {
      console.error('Error searching memory:', e)
    } finally {
      setLoading(false)
    }
  }

  const renderMemoryList = (items: MemoryItem[], type: string) => (
    <div className="space-y-3">
      {items.length === 0 ? (
        <div className="text-center text-slate-400 py-8">
          <Database className="w-12 h-12 mx-auto mb-4 opacity-50" />
          <p>No {type} memories stored yet</p>
        </div>
      ) : (
        items.map((item) => (
          <div
            key={item.id}
            className="bg-slate-800/50 rounded-lg p-4 hover:bg-slate-800 transition-all cursor-pointer group"
          >
            <div className="flex items-start justify-between">
              <div className="flex-1">
                {item.name && (
                  <h4 className="text-white font-medium">{item.name}</h4>
                )}
                {item.summary && (
                  <p className="text-slate-400 text-sm mt-1">{item.summary}</p>
                )}
                {item.content && (
                  <p className="text-slate-400 text-sm mt-1">{item.content}</p>
                )}
                <div className="flex gap-4 mt-2 text-xs text-slate-500">
                  {item.timestamp && (
                    <span className="flex items-center gap-1">
                      <Clock className="w-3 h-3" />
                      {new Date(item.timestamp).toLocaleString()}
                    </span>
                  )}
                  {item.category && (
                    <span className="px-2 py-0.5 bg-atlas-primary/20 text-atlas-primary rounded">
                      {item.category}
                    </span>
                  )}
                  {item.importance !== undefined && (
                    <span className="flex items-center gap-1">
                      Importance: {(item.importance * 100).toFixed(0)}%
                    </span>
                  )}
                  {item.connections !== undefined && (
                    <span className="flex items-center gap-1">
                      <Link2 className="w-3 h-3" />
                      {item.connections} connections
                    </span>
                  )}
                </div>
              </div>
              <ChevronRight className="w-5 h-5 text-slate-600 group-hover:text-atlas-primary transition-all" />
            </div>
          </div>
        ))
      )}
    </div>
  )

  return (
    <div className="p-8">
      {/* Header */}
      <div className="mb-8">
        <h1 className="text-3xl font-bold text-white mb-2">Memory Systems</h1>
        <p className="text-slate-400">
          Explore what Atlas has learned and remembered - this is how Atlas interfaces with the world
        </p>
      </div>

      {/* Search Bar */}
      <div className="glass rounded-xl p-4 mb-6">
        <div className="flex gap-4">
          <div className="flex-1 relative">
            <Search className="w-5 h-5 absolute left-3 top-1/2 -translate-y-1/2 text-slate-400" />
            <input
              type="text"
              value={searchQuery}
              onChange={(e) => setSearchQuery(e.target.value)}
              onKeyDown={(e) => e.key === 'Enter' && handleSearch()}
              placeholder="Search across all memories..."
              className="w-full bg-slate-800 border border-slate-700 rounded-lg pl-10 pr-4 py-2 text-white placeholder:text-slate-500 focus:outline-none focus:border-atlas-primary"
            />
          </div>
          <button
            onClick={handleSearch}
            disabled={loading}
            className="px-6 py-2 bg-atlas-primary hover:bg-atlas-primary/80 text-white rounded-lg font-medium transition-all disabled:opacity-50"
          >
            Search
          </button>
        </div>
        {searchResults.length > 0 && (
          <div className="mt-4 border-t border-slate-700 pt-4">
            <h4 className="text-sm text-slate-400 mb-2">
              Search Results ({searchResults.length})
            </h4>
            {renderMemoryList(searchResults, 'search')}
          </div>
        )}
      </div>

      {/* Tab Navigation */}
      <div className="flex gap-4 mb-6">
        {[
          { id: 'episodic' as const, icon: Clock, label: 'Episodic' },
          { id: 'semantic' as const, icon: Brain, label: 'Semantic' },
          { id: 'working' as const, icon: Layers, label: 'Working' },
          { id: 'associations' as const, icon: Link2, label: 'Associations' },
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

      {/* Content Area */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Memory List */}
        <div className="lg:col-span-2 glass rounded-xl p-6">
          <div className="flex items-center justify-between mb-4">
            <h3 className="text-lg font-semibold text-white">
              {activeTab === 'episodic' && 'Episodic Memories'}
              {activeTab === 'semantic' && 'Semantic Knowledge'}
              {activeTab === 'working' && 'Working Memory'}
              {activeTab === 'associations' && 'Cross-Modal Associations'}
            </h3>
            <button
              onClick={loadMemoryData}
              disabled={loading}
              className="text-atlas-primary hover:text-atlas-primary/80"
            >
              {loading ? (
                <Loader2 className="w-5 h-5 animate-spin" />
              ) : (
                'Refresh'
              )}
            </button>
          </div>

          {loading && !episodic && !semantic && !working && !associations ? (
            <div className="text-center py-12">
              <Loader2 className="w-8 h-8 animate-spin text-atlas-primary mx-auto" />
              <p className="text-slate-400 mt-4">Loading memories...</p>
            </div>
          ) : (
            <>
              {activeTab === 'episodic' && episodic && renderMemoryList(episodic.items, 'episodic')}
              {activeTab === 'semantic' && semantic && renderMemoryList(semantic.items, 'semantic')}
              {activeTab === 'working' && working && renderMemoryList(working.items, 'working')}
              {activeTab === 'associations' && associations && (
                <div className="space-y-3">
                  <div className="bg-slate-800/50 rounded-lg p-4 mb-4">
                    <span className="text-slate-400">Total Associations:</span>
                    <span className="text-white ml-2 text-2xl font-bold">
                      {associations.total_associations}
                    </span>
                  </div>
                  {associations.top_associations.map((assoc, i) => (
                    <div
                      key={i}
                      className="bg-slate-800/50 rounded-lg p-4 flex items-center justify-between"
                    >
                      <div className="flex items-center gap-4">
                        <span className="text-atlas-primary">{assoc.visual_pattern}</span>
                        <Link2 className="w-4 h-4 text-slate-500" />
                        <span className="text-purple-400">{assoc.audio_pattern}</span>
                      </div>
                      <div className="flex items-center gap-2">
                        <div className="w-24 h-2 bg-slate-700 rounded-full overflow-hidden">
                          <div
                            className="h-full bg-gradient-to-r from-atlas-primary to-purple-500"
                            style={{ width: `${assoc.strength * 100}%` }}
                          />
                        </div>
                        <span className="text-sm text-slate-400 w-12 text-right">
                          {(assoc.strength * 100).toFixed(0)}%
                        </span>
                      </div>
                    </div>
                  ))}
                </div>
              )}
            </>
          )}
        </div>

        {/* Info Sidebar */}
        <div className="space-y-6">
          {/* Memory Type Info */}
          <div className="glass rounded-xl p-6">
            <h3 className="text-lg font-semibold text-white mb-4">
              About {activeTab.charAt(0).toUpperCase() + activeTab.slice(1)} Memory
            </h3>
            <p className="text-sm text-slate-400">
              {activeTab === 'episodic' &&
                'Episodic memory stores specific experiences with timestamps. Like human episodic memory, it captures "what happened when" - the raw experiences that Atlas has processed.'}
              {activeTab === 'semantic' &&
                'Semantic memory holds abstract knowledge and concepts. It represents what Atlas "knows" - generalized patterns extracted from many experiences, organized into a concept graph.'}
              {activeTab === 'working' &&
                "Working memory is Atlas's active attention - what it's currently \"thinking about\". Limited in capacity (like human working memory), it holds the most relevant items for current processing."}
              {activeTab === 'associations' &&
                "Cross-modal associations link visual and auditory patterns. These are the learned connections between what Atlas sees and hears - the foundation of its understanding of how sights and sounds relate."}
            </p>
          </div>

          {/* Statistics */}
          <div className="glass rounded-xl p-6">
            <h3 className="text-lg font-semibold text-white mb-4">Statistics</h3>
            <div className="space-y-3">
              <div className="flex justify-between">
                <span className="text-slate-400">Episodic Items</span>
                <span className="text-white">{episodic?.items.length || 0}</span>
              </div>
              <div className="flex justify-between">
                <span className="text-slate-400">Semantic Concepts</span>
                <span className="text-white">{semantic?.items.length || 0}</span>
              </div>
              <div className="flex justify-between">
                <span className="text-slate-400">Working Memory</span>
                <span className="text-white">{working?.items.length || 0} / 7</span>
              </div>
              <div className="flex justify-between">
                <span className="text-slate-400">Associations</span>
                <span className="text-white">{associations?.total_associations || 0}</span>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  )
}

export default Memory
