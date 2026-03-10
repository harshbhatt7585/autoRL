import type { GenericFrame } from '../../types'
import { as3DObservation } from '../../utils'
import type { GridRenderModel } from '../types'

const MAZE_COLORS = {
  bg: '#101317',
  floor: '#f1f3f5',
  wall: '#4a5568',
  goal: '#ffd166',
  agent: '#3a86ff',
  grid: 'rgba(0,0,0,0.09)',
} as const

export function mazeFrameToGridModel(frame: GenericFrame | null): GridRenderModel | null {
  if (!frame) {
    return null
  }

  const obs = as3DObservation(frame.observation)
  if (obs.length < 3) {
    return null
  }

  const walls = obs[0]
  const goal = obs[1]
  const agent = obs[2]

  const height = walls.length
  const width = height > 0 ? walls[0].length : 0
  if (!width || !height) {
    return null
  }

  return {
    width,
    height,
    backgroundColor: MAZE_COLORS.bg,
    gridLineColor: MAZE_COLORS.grid,
    baseLayer: {
      cells: walls,
      activeColor: MAZE_COLORS.wall,
      inactiveColor: MAZE_COLORS.floor,
    },
    overlays: [
      { cells: goal, color: MAZE_COLORS.goal, inset: 2 },
      { cells: agent, color: MAZE_COLORS.agent, inset: 2 },
    ],
  }
}
