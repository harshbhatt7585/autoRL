import { useState } from 'react'

import Replay from '../features/render/Replay'
import type { RenderGame } from '../features/render/types'

function RenderPage() {
  const [game, setGame] = useState<RenderGame>('snake')

  return (
    <main className="app-shell">
      <Replay game={game} onGameChange={setGame} baseUrl={`/${game}`} />
    </main>
  )
}

export default RenderPage
