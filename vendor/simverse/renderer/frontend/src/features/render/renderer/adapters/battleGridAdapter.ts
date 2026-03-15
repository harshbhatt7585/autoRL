import type { GenericFrame } from '../../types'
import { as3DObservation } from '../../utils'
import type { GridRenderModel } from '../types'

const BATTLE_GRID_COLORS = {
  bg: '#12161d',
  floor: '#eef3f8',
  team0: '#3a86ff',
  team1: '#ef476f',
  grid: 'rgba(0,0,0,0.08)',
} as const

export function battleGridFrameToGridModel(frame: GenericFrame | null): GridRenderModel | null {
  if (!frame) {
    return null
  }

  const obs = as3DObservation(frame.observation)
  if (obs.length < 2) {
    return null
  }

  const team0 = obs[0]
  const team1 = obs[1]
  const height = team0.length
  const width = height > 0 ? team0[0].length : 0
  if (!width || !height) {
    return null
  }

  const floor = Array.from({ length: height }, () => Array(width).fill(0))

  return {
    width,
    height,
    backgroundColor: BATTLE_GRID_COLORS.bg,
    gridLineColor: BATTLE_GRID_COLORS.grid,
    baseLayer: {
      cells: floor,
      activeColor: BATTLE_GRID_COLORS.floor,
      inactiveColor: BATTLE_GRID_COLORS.floor,
    },
    overlays: [
      { cells: team0, color: BATTLE_GRID_COLORS.team0, inset: 2 },
      { cells: team1, color: BATTLE_GRID_COLORS.team1, inset: 2 },
    ],
  }
}
