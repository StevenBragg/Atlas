import { Outlet, NavLink } from 'react-router-dom'
import { useAtlas } from '../hooks/useAtlas'
import {
  Brain,
  LayoutDashboard,
  Upload,
  Database,
  Settings,
  Network,
  Wifi,
  WifiOff,
} from 'lucide-react'

function Layout() {
  const { connected, status } = useAtlas()

  const navItems = [
    { path: '/', icon: LayoutDashboard, label: 'Dashboard' },
    { path: '/input', icon: Upload, label: 'Data Input' },
    { path: '/memory', icon: Database, label: 'Memory' },
    { path: '/control', icon: Settings, label: 'Control' },
    { path: '/architecture', icon: Network, label: 'Architecture' },
  ]

  return (
    <div className="min-h-screen bg-atlas-darker flex">
      {/* Sidebar */}
      <aside className="w-64 bg-atlas-dark border-r border-slate-700/50 flex flex-col">
        {/* Logo */}
        <div className="p-6 border-b border-slate-700/50">
          <div className="flex items-center gap-3">
            <div className="relative">
              <Brain className="w-10 h-10 text-atlas-primary" />
              <div className="absolute inset-0 bg-atlas-primary/20 rounded-full blur-xl" />
            </div>
            <div>
              <h1 className="text-xl font-bold gradient-text">ATLAS</h1>
              <p className="text-xs text-slate-400">Self-Organizing Intelligence</p>
            </div>
          </div>
        </div>

        {/* Navigation */}
        <nav className="flex-1 p-4">
          <ul className="space-y-2">
            {navItems.map((item) => (
              <li key={item.path}>
                <NavLink
                  to={item.path}
                  className={({ isActive }) =>
                    `flex items-center gap-3 px-4 py-3 rounded-lg transition-all ${
                      isActive
                        ? 'bg-atlas-primary/20 text-atlas-primary border border-atlas-primary/30'
                        : 'text-slate-400 hover:bg-slate-800/50 hover:text-white'
                    }`
                  }
                >
                  <item.icon className="w-5 h-5" />
                  <span>{item.label}</span>
                </NavLink>
              </li>
            ))}
          </ul>
        </nav>

        {/* Status Footer */}
        <div className="p-4 border-t border-slate-700/50">
          <div className="glass rounded-lg p-4">
            <div className="flex items-center justify-between mb-2">
              <span className="text-sm text-slate-400">Connection</span>
              {connected ? (
                <div className="flex items-center gap-2 text-green-400">
                  <Wifi className="w-4 h-4" />
                  <span className="text-xs">Live</span>
                </div>
              ) : (
                <div className="flex items-center gap-2 text-yellow-400">
                  <WifiOff className="w-4 h-4" />
                  <span className="text-xs">Polling</span>
                </div>
              )}
            </div>
            <div className="flex items-center justify-between">
              <span className="text-sm text-slate-400">Learning</span>
              <span
                className={`text-xs px-2 py-0.5 rounded-full ${
                  status?.learning_enabled
                    ? 'bg-green-500/20 text-green-400'
                    : 'bg-slate-500/20 text-slate-400'
                }`}
              >
                {status?.learning_enabled ? 'Active' : 'Paused'}
              </span>
            </div>
          </div>
        </div>
      </aside>

      {/* Main Content */}
      <main className="flex-1 overflow-auto">
        <Outlet />
      </main>
    </div>
  )
}

export default Layout
