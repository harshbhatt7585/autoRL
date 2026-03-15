import { defineConfig, loadEnv } from 'vite'
import react from '@vitejs/plugin-react'

export default defineConfig(({ mode }) => {
  const env = loadEnv(mode, process.cwd(), '')

  const defaultApiTarget = 'http://127.0.0.1:8770'
  const snakeApiTarget = env.VITE_SNAKE_API_URL || defaultApiTarget
  const mazeApiTarget = env.VITE_MAZE_API_URL || env.VITE_SNAKE_API_URL || defaultApiTarget
  const battleGridApiTarget =
    env.VITE_BATTLE_GRID_API_URL || env.VITE_MAZE_API_URL || env.VITE_SNAKE_API_URL || defaultApiTarget

  return {
    plugins: [react()],
    server: {
      proxy: {
        '/snake': {
          target: snakeApiTarget,
          changeOrigin: true,
        },
        '/maze': {
          target: mazeApiTarget,
          changeOrigin: true,
        },
        '/battle-grid': {
          target: battleGridApiTarget,
          changeOrigin: true,
        },
      },
    },
  }
})
