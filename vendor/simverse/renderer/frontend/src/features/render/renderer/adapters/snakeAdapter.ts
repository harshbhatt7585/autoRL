import type { GenericFrame } from '../../types'
import { as3DObservation } from '../../utils'
import type { GridRenderModel } from '../types'

const SNAKE_COLORS = {
  bg: '#0e141b',
  floor: '#eef3f8',
  wall: '#3c4e62',
  food: '#d93f47',
  head: '#23924c',
  body: '#5acb85',
  grid: 'rgba(0,0,0,0.08)',
} as const

export function snakeFrameToGridModel(frame: GenericFrame | null): GridRenderModel | null {
  if (!frame) {
    return null
  }

  const obs = as3DObservation(frame.observation)
  if (obs.length < 4) {
    return null
  }

  const walls = obs[0]
  const food = obs[1]
  const head = obs[2]
  const body = obs[3]

  const height = walls.length
  const width = height > 0 ? walls[0].length : 0
  if (!width || !height) {
    return null
  }

  return {
    width,
    height,
    backgroundColor: SNAKE_COLORS.bg,
    gridLineColor: SNAKE_COLORS.grid,
    baseLayer: {
      cells: walls,
      activeColor: SNAKE_COLORS.wall,
      inactiveColor: SNAKE_COLORS.floor,
    },
    overlays: [
      { cells: body, color: SNAKE_COLORS.body, inset: 2 },
      { cells: head, color: SNAKE_COLORS.head, inset: 2 },
      { cells: food, color: SNAKE_COLORS.food, inset: 2 },
    ],
  }
}
