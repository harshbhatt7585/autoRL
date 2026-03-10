export function firstScalar(value: unknown, fallback = 0): number {
  if (typeof value === 'number' && Number.isFinite(value)) {
    return value
  }
  if (Array.isArray(value) && value.length > 0) {
    return firstScalar(value[0], fallback)
  }
  return fallback
}

export function parseNumber(value: unknown, fallback = 0): number {
  if (typeof value === 'number' && Number.isFinite(value)) {
    return value
  }
  if (typeof value === 'string') {
    const parsed = Number(value)
    return Number.isFinite(parsed) ? parsed : fallback
  }
  return fallback
}

export function parseReward(value: unknown): number {
  if (typeof value === 'number' && Number.isFinite(value)) {
    return value
  }
  if (Array.isArray(value)) {
    let total = 0
    let found = false
    for (const row of value) {
      if (!row || typeof row !== 'object') {
        continue
      }
      const rewardVal = (row as Record<string, unknown>).reward
      if (typeof rewardVal === 'number' && Number.isFinite(rewardVal)) {
        total += rewardVal
        found = true
      }
    }
    return found ? total : 0
  }
  if (value && typeof value === 'object') {
    const rewardVal = (value as Record<string, unknown>).reward
    if (typeof rewardVal === 'number' && Number.isFinite(rewardVal)) {
      return rewardVal
    }
  }
  return 0
}

function as2DLayer(value: unknown): number[][] {
  if (!Array.isArray(value)) {
    return []
  }
  const out: number[][] = []
  for (const row of value) {
    if (!Array.isArray(row)) {
      continue
    }
    out.push(row.map((cell) => parseNumber(cell, 0)))
  }
  return out
}

export function as3DObservation(value: unknown): number[][][] {
  if (!Array.isArray(value)) {
    return []
  }
  return value.map((layer) => as2DLayer(layer))
}

export function resolveUrl(baseUrl: string, maybeRelative: string): string {
  if (maybeRelative.startsWith('http://') || maybeRelative.startsWith('https://')) {
    return maybeRelative
  }
  const base = baseUrl.endsWith('/') ? baseUrl.slice(0, -1) : baseUrl
  const rel = maybeRelative.startsWith('/') ? maybeRelative : `/${maybeRelative}`
  return `${base}${rel}`
}
