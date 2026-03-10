import { useEffect, useRef } from 'react'

import type { GridRenderModel } from './types'

type GridRendererProps = {
  model: GridRenderModel | null
}

const MAX_CANVAS_SIZE = 760
const MIN_CELL_SIZE = 10
const MAX_CELL_SIZE = 34

function resolveCellSize(width: number, height: number): number {
  return Math.max(
    MIN_CELL_SIZE,
    Math.floor(Math.min(MAX_CELL_SIZE, MAX_CANVAS_SIZE / Math.max(width, height))),
  )
}

function GridRenderer({ model }: GridRendererProps) {
  const canvasRef = useRef<HTMLCanvasElement | null>(null)
  const dimsRef = useRef<{ width: number; height: number; cellSize: number } | null>(null)

  useEffect(() => {
    const canvas = canvasRef.current
    if (!canvas || !model) {
      return
    }

    const ctx = canvas.getContext('2d')
    if (!ctx) {
      return
    }

    if (!model.width || !model.height) {
      return
    }

    if (
      !dimsRef.current ||
      dimsRef.current.width !== model.width ||
      dimsRef.current.height !== model.height
    ) {
      const cellSize = resolveCellSize(model.width, model.height)
      canvas.width = model.width * cellSize
      canvas.height = model.height * cellSize
      dimsRef.current = { width: model.width, height: model.height, cellSize }
    }

    const cellSize = dimsRef.current.cellSize
    const baseThreshold = model.baseLayer.threshold ?? 0.5

    ctx.fillStyle = model.backgroundColor
    ctx.fillRect(0, 0, canvas.width, canvas.height)

    for (let y = 0; y < model.height; y += 1) {
      for (let x = 0; x < model.width; x += 1) {
        const active = (model.baseLayer.cells[y]?.[x] ?? 0) > baseThreshold
        ctx.fillStyle = active ? model.baseLayer.activeColor : model.baseLayer.inactiveColor
        ctx.fillRect(x * cellSize, y * cellSize, cellSize, cellSize)

        ctx.strokeStyle = model.gridLineColor
        ctx.strokeRect(x * cellSize, y * cellSize, cellSize, cellSize)
      }
    }

    for (const overlay of model.overlays) {
      const threshold = overlay.threshold ?? 0.5
      const inset = Math.max(0, overlay.inset ?? 0)
      const drawSize = Math.max(1, cellSize - inset * 2)

      for (let y = 0; y < model.height; y += 1) {
        for (let x = 0; x < model.width; x += 1) {
          if ((overlay.cells[y]?.[x] ?? 0) <= threshold) {
            continue
          }
          ctx.fillStyle = overlay.color
          ctx.fillRect(x * cellSize + inset, y * cellSize + inset, drawSize, drawSize)
        }
      }
    }
  }, [model])

  return <canvas ref={canvasRef} className="main-canvas" />
}

export default GridRenderer
