import { useEffect, useState } from 'react'

import GameRenderer from './GameRenderer'
import type { GenericFrame, RenderGame, ReplayDetail } from './types'
import { firstScalar, parseNumber, parseReward, resolveUrl } from './utils'

type ReplayProps = {
  game: RenderGame
  onGameChange: (game: RenderGame) => void
  baseUrl: string
}

function Replay({ game, onGameChange, baseUrl }: ReplayProps) {
  const [selectedReplay, setSelectedReplay] = useState<ReplayDetail | null>(null)
  const [frameIndex, setFrameIndex] = useState(0)
  const [playing, setPlaying] = useState(false)
  const [speed, setSpeed] = useState(1)
  const [status, setStatus] = useState('Loading replay...')
  const [error, setError] = useState('')
  const [refreshToken, setRefreshToken] = useState(0)

  const frames = Array.isArray(selectedReplay?.data?.frames) ? selectedReplay.data.frames : []
  const currentFrame: GenericFrame | null =
    frames.length > 0 ? frames[Math.max(0, Math.min(frameIndex, frames.length - 1))] : null

  useEffect(() => {
    const timer = window.setInterval(() => {
      setRefreshToken((value) => value + 1)
    }, 10_000)

    return () => {
      window.clearInterval(timer)
    }
  }, [])

  useEffect(() => {
    void (async () => {
      setError('')
      setStatus('Loading replay...')
      try {
        const response = await fetch(resolveUrl(baseUrl, '/replay/'), {
          cache: 'no-store',
        })
        if (!response.ok) {
          throw new Error(`Unable to load replay (${response.status})`)
        }
        const payload = (await response.json()) as ReplayDetail
        if (!payload || typeof payload.name !== 'string' || typeof payload.id !== 'string') {
          throw new Error('Invalid replay response format')
        }

        const previousFrameCount = Array.isArray(selectedReplay?.data?.frames)
          ? selectedReplay.data.frames.length
          : 0
        const nextFrameCount = Array.isArray(payload.data?.frames) ? payload.data.frames.length : 0
        const wasAtTail = previousFrameCount > 0 && frameIndex >= previousFrameCount - 1

        setSelectedReplay(payload)
        if (wasAtTail && nextFrameCount > previousFrameCount) {
          setFrameIndex(nextFrameCount - 1)
        } else {
          setFrameIndex((current) => Math.max(0, Math.min(current, Math.max(nextFrameCount - 1, 0))))
        }
      } catch (err) {
        const message = err instanceof Error ? err.message : String(err)
        setError(message)
        setSelectedReplay(null)
      }
    })()
  }, [baseUrl, refreshToken])

  useEffect(() => {
    if (frameIndex >= frames.length) {
      setFrameIndex(Math.max(0, frames.length - 1))
    }
  }, [frameIndex, frames.length])

  useEffect(() => {
    if (!playing || frames.length === 0) {
      return
    }

    const fps = 18 * speed
    const intervalMs = Math.max(16, Math.floor(1000 / fps))
    const timer = window.setInterval(() => {
      setFrameIndex((current) => {
        const next = current + 1
        if (next >= frames.length) {
          setPlaying(false)
          return Math.max(frames.length - 1, 0)
        }
        return next
      })
    }, intervalMs)

    return () => {
      window.clearInterval(timer)
    }
  }, [frames.length, playing, speed])

  useEffect(() => {
    if (!currentFrame) {
      if (!error) {
        setStatus('No replay data yet. Waiting for replay.json updates...')
      }
      return
    }

    const info = currentFrame.info ?? {}
    const reward = parseReward(currentFrame.rewards)
    const episode = firstScalar(currentFrame.episode, 0)
    const score = firstScalar(info.score, 0)
    const steps = firstScalar(info.steps, 0)
    const term = firstScalar(info.termination_reason, 0)
    const done = currentFrame.done ? 'yes' : 'no'

    setStatus(
      [
        `game: ${game}`,
        `file: ${selectedReplay?.name ?? 'replay.json'}`,
        `episode: ${episode}`,
        `frame: ${frameIndex + 1}/${frames.length}`,
        `step: ${currentFrame.step ?? '?'}`,
        `done: ${done}`,
        `term: ${term}`,
        `reward: ${reward.toFixed(3)}`,
        `score: ${score}`,
        `steps: ${steps}`,
      ].join('\n'),
    )
  }, [currentFrame, error, frameIndex, frames.length, game, selectedReplay?.name])

  return (
    <div className="viewer-grid">
      <div className="canvas-wrap">
        <GameRenderer game={game} frame={currentFrame} />
      </div>
      <aside className="panel">
        <label className="inline-label" htmlFor="replay-game">
          Game
        </label>
        <select
          id="replay-game"
          value={game}
          onChange={(event) => {
            onGameChange(event.target.value as RenderGame)
            setSelectedReplay(null)
            setFrameIndex(0)
            setPlaying(false)
            setRefreshToken((value) => value + 1)
          }}
        >
          <option value="snake">Snake</option>
          <option value="maze">Maze Race</option>
          <option value="battle-grid">Battle Grid</option>
        </select>

        <div className="control-row compact">
          <button
            type="button"
            disabled={frames.length === 0}
            onClick={() => {
              setFrameIndex(0)
              setPlaying(true)
            }}
          >
            Play
          </button>
          <button
            type="button"
            disabled={frames.length === 0}
            onClick={() => {
              setPlaying(false)
            }}
          >
            Pause
          </button>
          <button
            type="button"
            disabled={frames.length === 0}
            onClick={() => {
              setFrameIndex(0)
              setPlaying(false)
            }}
          >
            Reset
          </button>
        </div>

        <label className="inline-label" htmlFor="replay-speed">
          Speed: {speed.toFixed(2)}x
        </label>
        <input
          id="replay-speed"
          type="range"
          min={0.25}
          max={4}
          step={0.25}
          value={speed}
          onChange={(event) => {
            setSpeed(parseNumber(event.target.value, 1))
          }}
        />

        <pre className="status">{status}</pre>
        {error ? <p className="error">{error}</p> : null}
      </aside>
    </div>
  )
}

export default Replay
