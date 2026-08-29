import { useEffect, useState } from 'react'

const ANALYZER_URL = 'http://127.0.0.1:5000'
const emptyAnalysis = { camera: 'starting', face_detected: false, ear: null, mar: null, blink_count: 0, yawn_count: 0, looking_away_count: 0, eye_status: null, mouth_status: null, drowsiness_status: null, attention_status: null, attention_percentage: null, interview_duration: 0, last_updated: null }

function Icon({ children }) { return <span className="icon" aria-hidden="true">{children}</span> }

function Header({ onEndCall, connected, ended }) {
  return <header className="topbar"><div className="brand-lockup"><div className="brand-mark"><span /></div><div><p className="brand-name">NEXUS <span>/</span> ANALYZER</p><p className="brand-subtitle">Live interview intelligence</p></div></div><div className="header-actions">{!ended && <div className="session-pill"><span className="live-dot" /> {connected ? 'Session active' : 'Waiting for analyzer'}</div>}{!ended && <button className="end-call compact-end" onClick={onEndCall}><Icon>□</Icon> End call</button>}<button className="avatar" aria-label="Account menu">JD</button></div></header>
}

function CameraStage({ analysis, connected }) {
  const active = connected && analysis.camera === 'active'
  return <section className="camera-section"><div className="section-heading"><div><p className="eyebrow">Candidate view</p><h1>Stay present, stay sharp.</h1></div><div className="secure-label"><Icon>⌁</Icon> Private session</div></div><div className={`camera-stage ${active ? 'active' : 'idle'}`}>{active ? <img src={`${ANALYZER_URL}/video`} alt="Live camera feed" className="camera-video" /> : <div className="camera-placeholder"><div className="placeholder-avatar">JD</div><p>{analysis.camera === 'error' ? 'Camera could not be read' : 'Start the analyzer to begin the call'}</p></div>}<div className="camera-topline"><span className="recording-dot" /> {active ? 'MONITORING' : 'OFFLINE'}</div><div className="camera-corner top-left" /><div className="camera-corner top-right" /><div className="camera-corner bottom-left" /><div className="camera-corner bottom-right" /><div className="camera-caption"><span>JD</span><span>Candidate</span></div></div><div className="camera-footer"><div className="signal"><span className="signal-bars"><i /><i /><i /><i /></span> Camera signal <strong>{active ? 'Good' : 'Unavailable'}</strong></div><div className="footer-note">Analysis runs locally on your device</div></div></section>
}

function formatDuration(seconds) { return seconds == null ? '--:--' : `${String(Math.floor(seconds / 60)).padStart(2, '0')}:${String(Math.floor(seconds % 60)).padStart(2, '0')}` }
function value(value, digits = 0) { return value == null ? '--' : digits ? value.toFixed(digits) : value }
function MetricCard({ label, value: metricValue, detail, icon }) { return <div className="metric-card"><div className="metric-icon"><Icon>{icon}</Icon></div><div className="metric-copy"><span>{label}</span><strong>{metricValue}</strong><small>{detail}</small></div></div> }

function AnalyticsPanel({ analysis }) {
  const hasFace = analysis.face_detected
  const metrics = [
    { label: 'Eye status', value: hasFace ? analysis.eye_status : '--', detail: `EAR ${value(analysis.ear, 2)}`, icon: '◉' },
    { label: 'Blink count', value: analysis.blink_count, detail: 'This session', icon: '◌' },
    { label: 'Mouth status', value: hasFace ? analysis.mouth_status : '--', detail: `MAR ${value(analysis.mar, 2)}`, icon: '⌣' },
    { label: 'Yawn count', value: analysis.yawn_count, detail: 'This session', icon: '◡' },
    { label: 'Drowsiness', value: hasFace ? analysis.drowsiness_status : '--', detail: 'Based on live EAR', icon: '☀' },
    { label: 'Looking away', value: analysis.looking_away_count, detail: 'This session', icon: '↗' },
  ]
  return <aside className="analytics-panel"><div className="panel-heading"><div><p className="eyebrow">Live signals</p><h2>Interview analytics</h2></div></div><div className="overview-row"><div className="overview-item"><span className="overview-label">Interview duration</span><strong>{formatDuration(analysis.interview_duration)}</strong><span className="trend">Live session</span></div><div className="overview-item attention-overview"><span className="overview-label">Attention</span><strong>{value(analysis.attention_percentage)}%</strong><div className="progress-track"><div className="progress-value" style={{ width: `${analysis.attention_percentage || 0}%` }} /></div></div></div><div className="divider" /><div className="status-banner"><div className="status-check">✓</div><div><span>Current attention status</span><strong>{hasFace ? analysis.attention_status : 'Waiting for face'}</strong></div><span className="status-live">{analysis.camera === 'active' ? 'LIVE' : 'OFFLINE'}</span></div><div className="metrics-grid">{metrics.map((metric) => <MetricCard key={metric.label} {...metric} />)}</div><div className="last-updated">Last updated: {analysis.last_updated ? new Date(analysis.last_updated * 1000).toLocaleTimeString() : '--'}</div></aside>
}

function SummaryCard({ label, metricValue, detail }) { return <div className="summary-card"><span className="overview-label">{label}</span><strong>{metricValue}</strong><small>{detail}</small></div> }
function CallSummary({ summary, onNewCall }) {
  const interpretation = summary.overall_drowsiness_status === 'DROWSY' ? 'Drowsiness was detected during this call.' : 'No drowsiness events were detected during this call.'
  return <main className="summary-page"><div className="summary-heading"><p className="eyebrow">Call ended</p><h1>Overall Statistics</h1><p>Final analysis from this call.</p></div><div className="summary-grid"><SummaryCard label="Call duration" metricValue={formatDuration(summary.call_duration)} detail="Total call time" /><SummaryCard label="Total blinks" metricValue={summary.total_blinks} detail="Detected in this call" /><SummaryCard label="Total yawns" metricValue={summary.total_yawns} detail="Detected in this call" /><SummaryCard label="Average EAR" metricValue={value(summary.average_ear, 2)} detail="Collected eye frames" /><SummaryCard label="Average MAR" metricValue={value(summary.average_mar, 2)} detail="Collected mouth frames" /><SummaryCard label="Drowsiness events" metricValue={summary.drowsiness_events} detail="Eyes closed over threshold" /><SummaryCard label="Eyes closed time" metricValue={`${value(summary.total_eyes_closed_time, 1)}s`} detail="Accumulated duration" /><SummaryCard label="Maximum MAR" metricValue={value(summary.maximum_mar, 2)} detail="Maximum observed" /></div><div className="summary-status"><div><span className="overview-label">Overall eye status</span><strong>{summary.overall_eye_status}</strong></div><div><span className="overview-label">Overall drowsiness status</span><strong>{summary.overall_drowsiness_status}</strong></div><p>{interpretation}</p></div><button className="secondary-button" onClick={onNewCall}>Start New Call</button></main>
}

function EndInterviewDialog({ onCancel, onConfirm }) { return <div className="dialog-backdrop" role="presentation"><div className="dialog" role="dialog" aria-modal="true" aria-labelledby="end-title"><div className="dialog-icon">!</div><p className="eyebrow">End session</p><h2 id="end-title">Ready to wrap up?</h2><p>Your live analysis will stop and the final statistics will be prepared.</p><div className="dialog-actions"><button className="secondary-button" onClick={onCancel}>Keep interviewing</button><button className="end-call" onClick={onConfirm}>End interview</button></div></div></div> }

export default function App() {
  const [analysis, setAnalysis] = useState(emptyAnalysis)
  const [summary, setSummary] = useState(null)
  const [ended, setEnded] = useState(false)
  const [connected, setConnected] = useState(false)
  const [showDialog, setShowDialog] = useState(false)

  useEffect(() => {
    if (ended) return undefined
    const source = new EventSource(`${ANALYZER_URL}/api/stream`)
    source.onopen = () => setConnected(true)
    source.onmessage = (event) => { const next = JSON.parse(event.data); console.debug('[analysis] received frame state', next); setAnalysis(next); if (next.camera === 'stopped') source.close() }
    source.onerror = () => setConnected(false)
    return () => source.close()
  }, [ended])

  async function endCall() {
    setShowDialog(false)
    const response = await fetch(`${ANALYZER_URL}/api/end`, { method: 'POST' })
    const finalSummary = await response.json()
    setSummary(finalSummary)
    setEnded(true)
    setConnected(false)
  }

  async function newCall() {
    setSummary(null)
    setAnalysis(emptyAnalysis)
    await fetch(`${ANALYZER_URL}/api/start`, { method: 'POST' })
    setEnded(false)
  }

  return <div className="app-shell"><Header ended={ended} connected={connected} onEndCall={() => setShowDialog(true)} />{ended ? <CallSummary summary={summary} onNewCall={newCall} /> : <main className="dashboard"><CameraStage analysis={analysis} connected={connected} /><AnalyticsPanel analysis={analysis} /></main>}{showDialog && <EndInterviewDialog onCancel={() => setShowDialog(false)} onConfirm={endCall} />}</div>
}
