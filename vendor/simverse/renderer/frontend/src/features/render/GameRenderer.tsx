import { useMemo } from 'react'

import type { GenericFrame, RenderGame } from './types'
import GridRenderer from './renderer/GridRenderer'
import { battleGridFrameToGridModel } from './renderer/adapters/battleGridAdapter'
import { mazeFrameToGridModel } from './renderer/adapters/mazeAdapter'
import { snakeFrameToGridModel } from './renderer/adapters/snakeAdapter'

type GameRendererProps = {
  game: RenderGame
  frame: GenericFrame | null
}

function GameRenderer({ game, frame }: GameRendererProps) {
  const model = useMemo(() => {
    if (game === 'battle-grid') {
      return battleGridFrameToGridModel(frame)
    }
    if (game === 'maze') {
      return mazeFrameToGridModel(frame)
    }
    return snakeFrameToGridModel(frame)
  }, [frame, game])

  return <GridRenderer model={model} />
}

export default GameRenderer
