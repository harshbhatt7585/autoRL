export type RenderGame = 'snake' | 'maze' | 'battle-grid'

export type GenericFrame = {
  step?: number
  episode?: unknown
  observation?: unknown
  features?: unknown
  rewards?: unknown
  info?: Record<string, unknown>
  done?: boolean
}

export type ReplayFile = {
  episode?: number
  steps?: number
  frames?: GenericFrame[]
  [key: string]: unknown
}

export type ReplaySummary = {
  id: string
  name: string
}

export type ReplaysResponse = {
  episodes: ReplaySummary[]
}

export type ReplayDetail = {
  id: string
  name: string
  data: ReplayFile
}
