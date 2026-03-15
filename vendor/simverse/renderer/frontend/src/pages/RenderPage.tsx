import { useEffect, useMemo, useState } from 'react'
import { useSearchParams } from 'react-router-dom'

import Replay from '../features/render/Replay'
import type { RenderGame } from '../features/render/types'

function RenderPage() {
  const [searchParams, setSearchParams] = useSearchParams()
  const initialGame = useMemo<RenderGame>(() => {
    const requestedGame = searchParams.get('game')
    if (requestedGame === 'maze' || requestedGame === 'battle-grid' || requestedGame === 'snake') {
      return requestedGame
    }
    return 'snake'
  }, [searchParams])
  const [game, setGame] = useState<RenderGame>(initialGame)

  useEffect(() => {
    setGame(initialGame)
  }, [initialGame])

  const handleGameChange = (nextGame: RenderGame) => {
    const nextParams = new URLSearchParams(searchParams)
    nextParams.set('game', nextGame)
    nextParams.delete('dir')
    setSearchParams(nextParams, { replace: true })
    setGame(nextGame)
  }

  return (
    <main className="app-shell">
      <Replay game={game} onGameChange={handleGameChange} baseUrl={`/${game}`} />
    </main>
  )
}

export default RenderPage
