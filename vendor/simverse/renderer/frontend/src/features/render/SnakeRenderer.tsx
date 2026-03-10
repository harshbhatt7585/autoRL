import { useMemo } from 'react'

import type { GenericFrame } from './types'
import GridRenderer from './renderer/GridRenderer'
import { snakeFrameToGridModel } from './renderer/adapters/snakeAdapter'

type SnakeRendererProps = {
  frame: GenericFrame | null
}

function SnakeRenderer({ frame }: SnakeRendererProps) {
  const model = useMemo(() => snakeFrameToGridModel(frame), [frame])

  return <GridRenderer model={model} />
}

export default SnakeRenderer
