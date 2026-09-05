import { useState } from 'react'

const defaults = { TransactionAmt_raw: '150', hour_of_day: '14', day_of_week: '3', txn_deviation_from_customer_avg: '200', V_V29: '0', V_V37: '0', V_V30: '0', V_V35: '0' }
const labels = { TransactionAmt_raw: 'Transaction amount', TransactionAmt_log1p: 'Transaction amount (log)', txn_deviation_from_customer_avg: 'Deviation from customer average', hour_of_day: 'Hour of day', day_of_week: 'Day index', is_high_value: 'High-value transaction', R_emaildomain: 'Recipient email domain', P_emaildomain: 'Purchaser email domain' }
const featureName = (name) => labels[name] || (/^card/.test(name) ? `Card attribute ${name.slice(4)}` : /^addr/.test(name) ? `Address attribute ${name.slice(4)}` : name)
const number = (value) => value.toLocaleString('en-US', { minimumFractionDigits: 2, maximumFractionDigits: 2 })
const actions = { APPROVE: 'Approve', HUMAN_REVIEW: 'Human review', ESCALATE: 'Escalate' }

function Factors({ title, factors, protective = false }) {
  return <section className="factors"><h3>{title}</h3>
    {factors.length ? <ul>{[...factors].sort((a, b) => b.contribution - a.contribution).map(factor => <li key={factor.feature}>
      <span title={factor.feature}>{featureName(factor.feature)}</span>
      <strong className={protective ? 'protective' : 'risk'}>{protective ? '−' : '+'}{(factor.contribution * 100).toFixed(2)} pp</strong>
    </li>)}</ul> : <p className="muted">No {protective ? 'protective' : 'risk'} factors returned for this transaction.</p>}
  </section>
}

export default function App() {
  const [form, setForm] = useState(defaults)
  const [result, setResult] = useState(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState('')
  const [submitted, setSubmitted] = useState('')
  const changed = result && submitted !== JSON.stringify(form)

  function field(name, label, options = {}) {
    return <label className="field" key={name} htmlFor={name}><span>{label}</span>
      <input id={name} type="number" required step="any" {...options} value={form[name]} onChange={event => setForm({ ...form, [name]: event.target.value })} />
    </label>
  }

  async function analyze(event) {
    event.preventDefault()
    const payload = Object.fromEntries(Object.entries(form).map(([key, value]) => [key, Number(value)]))
    if (Object.values(payload).some(value => !Number.isFinite(value))) {
      setError('Enter a valid number in every field.'); return
    }
    payload.TransactionAmt_log1p = Math.log1p(payload.TransactionAmt_raw)
    payload.is_high_value = payload.TransactionAmt_raw > 500 ? 1 : 0
    setLoading(true); setError(''); setResult(null)
    const controller = new AbortController()
    const timeout = setTimeout(() => controller.abort(), 30000)
    try {
      const response = await fetch('/api/predict', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(payload), signal: controller.signal })
      if (!response.ok) throw new Error(`Analysis failed (HTTP ${response.status}). Check that the backend is running on port 8000, then try again.`)
      const data = await response.json()
      if (!Number.isFinite(data.risk_score) || !data.expected_costs || !Array.isArray(data.top_risk_factors) || !Array.isArray(data.top_protective_factors)) throw new Error('The API returned an incomplete result. Check the backend and try again.')
      setResult(data); setSubmitted(JSON.stringify(form))
    } catch (err) {
      setError(err.name === 'AbortError' ? 'Analysis timed out. Check the backend and try again.' : err instanceof TypeError ? 'Cannot reach the API. Start the backend on port 8000 and try again.' : err.message)
    } finally { clearTimeout(timeout); setLoading(false) }
  }

  return <div className="app">
    <header className="brand-header"><div className="brand"><svg className="brand-mark" viewBox="0 0 40 44" fill="none" aria-hidden="true"><path d="M20 3 35 9v12c0 9-6 15-15 20C11 36 5 30 5 21V9L20 3Z" stroke="currentColor" strokeWidth="2.5"/><path d="m12 22 5 5 11-12" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round"/></svg><div><h1>ReturnGuard</h1><p>AI-Powered Transaction Risk Decision System</p></div></div><p className="oversight">AI recommends.<br/><strong>People make the final decision.</strong></p></header>
    <main>
      <section className="panel inputs" aria-labelledby="transaction-title">
        <h2 id="transaction-title">Transaction Analysis</h2>
        <p className="intro">Review the sample inputs, then analyze the transaction.</p>
        <form onSubmit={analyze}>
          <fieldset disabled={loading}>
            {field('TransactionAmt_raw', 'Transaction amount (currency units)', { min: 0, step: '0.01' })}
            <p className="derived">High value: <strong>{Number(form.TransactionAmt_raw) > 500 ? 'Yes' : 'No'}</strong> · Automatically set when amount is above 500.</p>
            <div className="field-grid">
              {field('hour_of_day', 'Hour of day (0–23)', { min: 0, max: 23, step: 1 })}
              {field('day_of_week', 'Day index (0–6)', { min: 0, max: 6, step: 1 })}
            </div>
            {field('txn_deviation_from_customer_avg', 'Deviation from customer average', { step: '0.01' })}
            <p className="hint">Amount minus the customer’s historical average, in currency units.</p>
            <details><summary>Additional model inputs</summary><p className="hint">Anonymized transaction signals. Other omitted API features use the existing zero defaults.</p>
              <div className="field-grid">{['V_V29', 'V_V37', 'V_V30', 'V_V35'].map(name => field(name, name))}</div>
            </details>
            <button type="submit">{loading ? 'Analyzing transaction…' : 'Analyze Transaction'}</button>
          </fieldset>
        </form>
        <p className="hint api-note">Analysis uses the live local prediction API.</p>
        {error && <p className="error" role="alert">{error}</p>}
      </section>
      <section className="panel results" aria-labelledby="result-title" aria-live="polite" aria-busy={loading}>
        <h2 id="result-title">Risk Decision</h2>
        {!result ? <div className="empty"><h3>{loading ? 'Analyzing your transaction…' : error ? 'Analysis unavailable' : 'Ready to analyze'}</h3><p>{loading ? 'Calculating risk, expected costs and contributing factors.' : error ? 'Your inputs are preserved. Resolve the connection issue and select Analyze Transaction to retry.' : 'Select Analyze Transaction to see the risk score, recommended action and explanation here.'}</p></div> : <>
          <p className="result-context">Analyzed amount: {number(result.transaction_amount)} currency units</p>
          {changed && <p className="stale" role="status">Inputs changed. Analyze again to update this result.</p>}
          <div className="decision">
            <div><span className="muted">Risk score</span><div className="score">{(result.risk_score * 100).toFixed(2)}<span>%</span></div><span className={`band ${result.risk_band.toLowerCase()}`}>{result.risk_band} RISK</span></div>
            <div><span className="muted">Recommended action</span><h3 className="action">{actions[result.recommended_action] || result.recommended_action}</h3><p className="hint">{result.recommended_action === 'HUMAN_REVIEW' ? 'Route to an investigator for review.' : result.recommended_action === 'ESCALATE' ? 'Escalate for further investigation.' : 'Approval recommended for human consideration.'}</p></div>
          </div>
          <section className="costs"><h3>Expected Costs</h3><p className="hint">Modeled loss in currency units for each action.</p>
            <dl>{Object.entries(result.expected_costs).map(([action, cost]) => <div key={action} className={action === ({ APPROVE: 'approve', HUMAN_REVIEW: 'review', ESCALATE: 'escalate' }[result.recommended_action]) ? 'recommended-cost' : ''}><dt>{action === 'review' ? 'Human review' : action === 'approve' ? 'Approve' : 'Escalate'}</dt><dd>{number(cost)}</dd></div>)}</dl>
          </section>
          <div className="factor-grid"><Factors title="Top risk factors" factors={result.top_risk_factors} /><Factors title="Top protective factors" factors={result.top_protective_factors} protective /></div>
          <p className="hint">Factor impacts show probability changes in percentage points (pp) relative to the explainer’s reference values; they are not additive.</p>
        </>}
      </section>
    </main>
    <footer>AI recommends. Human investigators make final decisions.</footer>
  </div>
}
