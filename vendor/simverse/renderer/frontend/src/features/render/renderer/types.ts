export type BinaryLayer = {
  cells: number[][]
  activeColor: string
  inactiveColor: string
  threshold?: number
}

export type OverlayLayer = {
  cells: number[][]
  color: string
  threshold?: number
  inset?: number
}

export type GridRenderModel = {
  width: number
  height: number
  backgroundColor: string
  gridLineColor: string
  baseLayer: BinaryLayer
  overlays: OverlayLayer[]
}
