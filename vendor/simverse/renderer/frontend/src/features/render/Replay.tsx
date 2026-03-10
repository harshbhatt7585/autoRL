import { useEffect, useMemo, useState } from 'react'

import GameRenderer from './GameRenderer'
import type {
  GenericFrame,
  RenderGame,
  ReplayDetail,
  ReplaySummary,
  ReplaysResponse,
} from './types'
import { firstScalar, parseNumber, parseReward, resolveUrl } from './utils'

type ReplayProps = {
  game: RenderGame
  onGameChange: (game: RenderGame) => void
  baseUrl: string
}

function Replay({ game, onGameChange, baseUrl }: ReplayProps) {
  const [episodes, setEpisodes] = useState<ReplaySummary[]>([])
  const [selectedReplayId, setSelectedReplayId] = useState('')
  const [selectedReplay, setSelectedReplay] = useState<ReplayDetail | null>(null)
  const [frameIndex, setFrameIndex] = useState(0)
  const [playing, setPlaying] = useState(false)
  const [speed, setSpeed] = useState(1)
  const [status, setStatus] = useState('Loading replays...')
  const [error, setError] = useState('')
  const [refreshToken, setRefreshToken] = useState(0)

  const selectedReplayName = useMemo(() => {
    if (selectedReplay?.name) {
      return selectedReplay.name
    }
    return episodes.find((episode) => episode.id === selectedReplayId)?.name ?? ''
  }, [episodes, selectedReplay, selectedReplayId])
  const selectedReplayIndex = useMemo(
    () => episodes.findIndex((episode) => episode.id === selectedReplayId),
    [episodes, selectedReplayId],
  )

  const frames = Array.isArray(selectedReplay?.data?.frames) ? selectedReplay.data.frames : []
  const currentFrame: GenericFrame | null =
    frames.length > 0 ? frames[Math.max(0, Math.min(frameIndex, frames.length - 1))] : null

  useEffect(() => {
    void (async () => {
      setError('')
      setStatus('Loading replays...')
      try {
        const response = await fetch(resolveUrl(baseUrl, '/replays/'), { cache: 'no-store' })
        if (!response.ok) {
          throw new Error(`Unable to load replays (${response.status})`)
        }

        const payload = (await response.json()) as ReplaysResponse
        const nextEpisodes = Array.isArray(payload.episodes)
          ? payload.episodes
              .map((episode) => {
                if (!episode || typeof episode.name !== 'string') {
                  return null
                }
                const fallbackId = episode.name.replace(/\.json$/i, '')
                const id =
                  typeof episode.id === 'string' && episode.id.length > 0 ? episode.id : fallbackId
                return { id, name: episode.name } satisfies ReplaySummary
              })
              .filter((episode): episode is ReplaySummary => episode !== null)
          : []

        setEpisodes(nextEpisodes)
        if (nextEpisodes.length === 0) {
          setSelectedReplayId('')
          setSelectedReplay(null)
          setFrameIndex(0)
          setStatus('No replay JSON files found.')
          return
        }

        setSelectedReplayId((currentId) => {
          if (nextEpisodes.some((episode) => episode.id === currentId)) {
            return currentId
          }
          return nextEpisodes[0].id
        })
        setStatus(`Loaded ${nextEpisodes.length} replay files.`)
      } catch (err) {
        const message = err instanceof Error ? err.message : String(err)
        setError(message)
      }
    })()
  }, [baseUrl, refreshToken])

  useEffect(() => {
    if (!selectedReplayId) {
      setSelectedReplay(null)
      return
    }

    void (async () => {
      setError('')
      setStatus(`Loading replay ${selectedReplayId}...`)
      try {
        const response = await fetch(resolveUrl(baseUrl, `/replays/${encodeURIComponent(selectedReplayId)}`), {
          cache: 'no-store',
        })
        if (!response.ok) {
          throw new Error(`Unable to load replay ${selectedReplayId} (${response.status})`)
        }

        const payload = (await response.json()) as ReplayDetail
        if (!payload || typeof payload.name !== 'string' || typeof payload.id !== 'string') {
          throw new Error('Invalid replay response format')
        }

        setSelectedReplay(payload)
        setFrameIndex(0)
      } catch (err) {
        const message = err instanceof Error ? err.message : String(err)
        setError(message)
      }
    })()
  }, [baseUrl, selectedReplayId])

  const selectEpisodeByOffset = (offset: number, autoPlay = false) => {
    if (episodes.length === 0 || selectedReplayIndex < 0) {
      return
    }
    const nextIndex = Math.max(0, Math.min(episodes.length - 1, selectedReplayIndex + offset))
    const nextEpisode = episodes[nextIndex]
    if (!nextEpisode) {
      return
    }
    setSelectedReplayId(nextEpisode.id)
    setFrameIndex(0)
    setPlaying(autoPlay)
  }

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
          const nextEpisode = episodes[selectedReplayIndex + 1]
          if (nextEpisode) {
            setSelectedReplayId(nextEpisode.id)
            return 0
          }
          setPlaying(false)
          return Math.max(frames.length - 1, 0)
        }
        return next
      })
    }, intervalMs)

    return () => {
      window.clearInterval(timer)
    }
  }, [episodes, frames.length, playing, selectedReplayIndex, speed])

  useEffect(() => {
    if (!currentFrame) {
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
        `file: ${selectedReplayName}`,
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
  }, [currentFrame, frameIndex, frames.length, game, selectedReplayName])

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
            setSelectedReplayId('')
            setSelectedReplay(null)
            setFrameIndex(0)
            setPlaying(false)
            setRefreshToken((value) => value + 1)
          }}
        >
          <option value="snake">Snake</option>
          <option value="maze">Maze Runner</option>
        </select>

        <label className="inline-label" htmlFor="replay-episode">
          Replay File
        </label>
        <select
          id="replay-episode"
          value={selectedReplayId}
          onChange={(event) => {
            setSelectedReplayId(event.target.value)
            setFrameIndex(0)
            setPlaying(false)
          }}
          disabled={episodes.length === 0}
        >
          {episodes.map((episode) => (
            <option key={episode.id} value={episode.id}>
              {episode.name}
            </option>
          ))}
        </select>

        <div className="control-row compact">
          <button
            type="button"
            disabled={episodes.length === 0}
            onClick={() => {
              selectEpisodeByOffset(-1, false)
            }}
          >
            Prev
          </button>
          <button
            type="button"
            disabled={frames.length === 0}
            onClick={() => {
              setPlaying((value) => !value)
            }}
          >
            {playing ? 'Pause' : 'Play'}
          </button>
          <button
            type="button"
            disabled={episodes.length === 0}
            onClick={() => {
              selectEpisodeByOffset(1, true)
            }}
          >
            Next
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

        <label className="inline-label" htmlFor="replay-seek">
          Frame
        </label>
        <input
          id="replay-seek"
          type="range"
          min={0}
          max={Math.max(frames.length - 1, 0)}
          step={1}
          value={Math.max(0, Math.min(frameIndex, Math.max(frames.length - 1, 0)))}
          onChange={(event) => {
            setFrameIndex(parseInt(event.target.value, 10) || 0)
          }}
          disabled={frames.length === 0}
        />

        <button
          type="button"
          className="full"
          onClick={() => {
            setRefreshToken((value) => value + 1)
          }}
        >
          Reload
        </button>

        <pre className="status">{status}</pre>
        {error ? <p className="error">{error}</p> : null}
      </aside>
    </div>
  )
}

export default Replay
