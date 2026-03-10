import { Navigate, Route, Routes } from 'react-router-dom'

import './App.css'
import RenderPage from './pages/RenderPage'

function App() {
  return (
    <div className="route-shell">
      <Routes>
        <Route path="/render" element={<RenderPage />} />
        <Route path="*" element={<Navigate to="/render" replace />} />
      </Routes>
    </div>
  )
}

export default App
