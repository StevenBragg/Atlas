import { BrowserRouter, Routes, Route } from 'react-router-dom'
import { AtlasProvider } from './hooks/useAtlas'
import Layout from './components/Layout'
import Dashboard from './pages/Dashboard'
import DataInput from './pages/DataInput'
import Memory from './pages/Memory'
import Control from './pages/Control'
import Architecture from './pages/Architecture'

function App() {
  return (
    <AtlasProvider>
      <BrowserRouter>
        <Routes>
          <Route path="/" element={<Layout />}>
            <Route index element={<Dashboard />} />
            <Route path="input" element={<DataInput />} />
            <Route path="memory" element={<Memory />} />
            <Route path="control" element={<Control />} />
            <Route path="architecture" element={<Architecture />} />
          </Route>
        </Routes>
      </BrowserRouter>
    </AtlasProvider>
  )
}

export default App
