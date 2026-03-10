import { Link } from 'react-router-dom'

function HomePage() {
  return (
    <main className="home-shell">
      <section className="home-card">
        <h1>Simverse Frontend</h1>
        <p>Open the renderer dashboard to stream live environments and replay episodes.</p>
        <Link className="home-link" to="/render">
          Open /render
        </Link>
      </section>
    </main>
  )
}

export default HomePage
