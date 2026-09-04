import React, { useState } from 'react'

function App() {
  // Form state with sensible default numeric values
  const [TransactionAmt_raw, setTransactionAmt_raw] = useState(150.0)
  const [hour_of_day, setHourOfDay] = useState(14)
  const [day_of_week, setDayOfWeek] = useState(3)
  const [is_high_value, setIsHighValue] = useState(0)
  const [txn_deviation_from_customer_avg, setTxnDeviation] = useState(200.0)
  const [V_V29, setVV29] = useState(0.0)
  const [V_V37, setVV37] = useState(0.0)
  const [V_V30, setVV30] = useState(0.0)
  const [V_V35, setVV35] = useState(0.0)

  // API state
  const [prediction, setPrediction] = useState(null)
  const [isLoading, setIsLoading] = useState(false)
  const [apiError, setApiError] = useState(null)

  // Analyze: POST form values to the FastAPI backend
  const analyzeTransaction = async () => {
    setIsLoading(true)
    setApiError(null)
    try {
      const body = {
        TransactionAmt_raw: Number(TransactionAmt_raw),
        hour_of_day: Number(hour_of_day),
        day_of_week: Number(day_of_week),
        is_high_value: Number(is_high_value),
        txn_deviation_from_customer_avg: Number(txn_deviation_from_customer_avg),
        V_V29: Number(V_V29),
        V_V37: Number(V_V37),
        V_V30: Number(V_V30),
        V_V35: Number(V_V35),
      }
      const res = await fetch('http://localhost:8000/predict', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(body),
      })
      if (!res.ok) {
        const errText = await res.text()
        throw new Error('API error: ' + res.status + ' ' + errText)
      }
      const data = await res.json()
      setPrediction(data)
    } catch (e) {
      setApiError(e.message || 'Unexpected error')
    } finally {
      setIsLoading(false)
    }
  }

  return (
    <div className="min-h-screen bg-gray-50 dark:bg-gray-900">
      <header className="border-y border-gray-200 dark:border-gray-700">
        <div className="max-w-7xl mx-auto px-4 py-6">
          <h1 className="text-3xl font-bold text-gray-900 dark:text-gray-100">
            ReturnGuard
          </h1>
          <p className="text-sm text-gray-600 dark:text-gray-400">
            AI-Powered Transaction Risk Decision System
          </p>
        </div>
      </header>

      <main className="max-w-7xl mx-auto p-4">
        <section className="mt-8">
          <h2 className="text-xl font-medium text-gray-900 dark:text-gray-100 mb-4">
            Transaction Analysis
          </h2>
          <div className="bg-white dark:bg-gray-800 rounded-lg p-6 border border-gray-200 dark:border-gray-700">
            {/* Transaction amount field */}
            <div className="grid grid-cols-1 gap-4 mb-4">
              <label className="text-sm text-gray-700 dark:text-gray-300 block">
                Transaction Amount (raw)
                <span className="text-xs text-gray-500 dark:text-gray-400 ml-2">
                  ($)
                </span>
              </label>
              <input
                type="number"
                step="0.01"
                value={TransactionAmt_raw}
                onChange={(e) => setTransactionAmt_raw(Number(e.target.value) || 0.0)}
                className="w-full px-3 py-2 border border-gray-300 dark:border-gray-600 rounded-md shadow-sm focus:outline-none focus:ring-blue-500 dark:focus:ring-blue-500 focus:border-blue-500"
                placeholder="150.00"
                aria-label="Transaction amount in currency units"
              />
              <p className="text-xs text-gray-500 dark:text-gray-400 mt-1">
                Total transaction amount before log transformation
              </p>
            </div>

            {/* Hour of day */}
            <div className="grid grid-cols-1 gap-4 mb-4">
              <label className="text-sm text-gray-700 dark:text-gray-300 block">
                Hour of Day
                <span className="text-xs text-gray-500 dark:text-gray-400 ml-2">
                  0-23
                </span>
              </label>
              <input
                type="number"
                min="0"
                max="23"
                value={hour_of_day}
                onChange={(e) => setHourOfDay(Number(e.target.value) || 0)}
                className="w-full px-3 py-2 border border-gray-300 dark:border-gray-600 rounded-md shadow-sm focus:outline-none focus:ring-blue-500 dark:focus:ring-blue-500 focus:border-blue-500"
                placeholder="14"
                aria-label="Hour of day (0-23)"
              />
              <p className="text-xs text-gray-500 dark:text-gray-400 mt-1">
                Hour in 24-hour format (e.g., 14 = 2PM)
              </p>
            </div>

            {/* Day of week */}
            <div className="grid grid-cols-1 gap-4 mb-4">
              <label className="text-sm text-gray-700 dark:text-gray-300 block">
                Day of Week
                <span className="text-xs text-gray-500 dark:text-gray-400 ml-2">
                  0-6
                </span>
              </label>
              <input
                type="number"
                min="0"
                max="6"
                value={day_of_week}
                onChange={(e) => setDayOfWeek(Number(e.target.value) || 0)}
                className="w-full px-3 py-2 border border-gray-300 dark:border-gray-600 rounded-md shadow-sm focus:outline-none focus:ring-blue-500 dark:focus:ring-blue-500 focus:border-blue-500"
                placeholder="3"
                aria-label="Day of week (0=Sunday, 6=Saturday)"
              />
              <p className="text-xs text-gray-500 dark:text-gray-400 mt-1">
                Day of week (0=Sunday, 1=Monday, ..., 6=Saturday)
              </p>
            </div>

            {/* Is high value */}
            <div className="grid grid-cols-1 gap-4 mb-4">
              <label className="text-sm text-gray-700 dark:text-gray-300 block">
                Is High Value
                <span className="text-xs text-gray-500 dark:text-gray-400 ml-2">
                  > 500
                </span>
              </label>
              <input
                type="number"
                min="0"
                max="1"
                value={is_high_value}
                onChange={(e) => setIsHighValue(Number(e.target.value) || 0)}
                className="w-full px-3 py-2 border border-gray-300 dark:border-gray-600 rounded-md shadow-sm focus:outline-none focus:ring-blue-500 dark:focus:ring-blue-500 focus:border-blue-500"
                placeholder="0 or 1"
                aria-label="Whether transaction is above $500 threshold"
              />
              <p className="text-xs text-gray-500 dark:text-gray-400 mt-1">
                Transaction amount > $500 (deterministic flag)
              </p>
            </div>

            {/* Transaction deviation from customer avg */}
            <div className="grid grid-cols-1 gap-4 mb-4">
              <label className="text-sm text-gray-700 dark:text-gray-300 block">
                Deviation from Customer Avg
                <span className="text-xs text-gray-500 dark:text-gray-400 ml-2">
                  ($)
                </span>
              </label>
              <input
                type="number"
                step="0.01"
                value={txn_deviation_from_customer_avg}
                onChange={(e) =>
                  setTxnDeviation(Number(e.target.value) || 0.0)
                }
                className="w-full px-3 py-2 border border-gray-300 dark:border-gray-600 rounded-md shadow-sm focus:outline-none focus:ring-blue-500 dark:focus:ring-blue-500 focus:border-blue-500"
                placeholder="200.00"
                aria-label="Transaction deviation from customer historical average"
              />
              <p className="text-xs text-gray-500 dark:text-gray-400 mt-1">
                Difference between transaction amount and customer's historical average
              </p>
            </div>

            {/* V_V29 */}
            <div className="grid grid-cols-1 gap-4 mb-4">
              <label className="text-sm text-gray-700 dark:text-gray-300 block">
                V_V29
                <span className="text-xs text-gray-500 dark:text-gray-400 ml-2">
                  value
                </span>
              </label>
              <input
                type="number"
                step="0.01"
                value={V_V29}
                onChange={(e) => setVV29(Number(e.target.value) || 0.0)}
                className="w-full px-3 py-2 border border-gray-300 dark:border-gray-600 rounded-md shadow-sm focus:outline-none focus:ring-blue-500 dark:focus:ring-blue-500 focus:border-blue-500"
                placeholder="0.00"
                aria-label="V feature V_V29"
              />
              <p className="text-xs text-gray-500 dark:text-gray-400 mt-1">
                V feature index 29 (enrichment feature)
              </p>
            </div>

            {/* V_V37 */}
            <div className="grid grid-cols-1 gap-4 mb-4">
              <label className="text-sm text-gray-700 dark:text-gray-300 block">
                V_V37
                <span className="text-xs text-gray-500 dark:text-gray-400 ml-2">
                  value
                </span>
              </label>
              <input
                type="number"
                step="0.01"
                value={V_V37}
                onChange={(e) => setVV37(Number(e.target.value) || 0.0)}
                className="w-full px-3 py-2 border border-gray-300 dark:border-gray-600 rounded-md shadow-sm focus:outline-none focus:ring-blue-500 dark:focus:ring-blue-500 focus:border-blue-500"
                placeholder="0.00"
                aria-label="V feature V_V37"
              />
              <p className="text-xs text-gray-500 dark:text-gray-400 mt-1">
                V feature index 37 (enrichment feature)
              </p>
            </div>

            {/* V_V30 */}
            <div className="grid grid-cols-1 gap-4 mb-4">
              <label className="text-sm text-gray-700 dark:text-gray-300 block">
                V_V30
                <span className="text-xs text-gray-500 dark:text-gray-400 ml-2">
                  value
                </span>
              </label>
              <input
                type="number"
                step="0.01"
                value={V_V30}
                onChange={(e) => setVV30(Number(e.target.value) || 0.0)}
                className="w-full px-3 py-2 border border-gray-300 dark:border-gray-600 rounded-md shadow-sm focus:outline-none focus:ring-blue-500 dark:focus:ring-blue-500 focus:border-blue-500"
                placeholder="0.00"
                aria-label="V feature V_V30"
              />
              <p className="text-xs text-gray-500 dark:text-gray-400 mt-1">
                V feature index 30 (enrichment feature)
              </p>
            </div>

            {/* V_V35 */}
            <div className="grid grid-cols-1 gap-4 mb-4">
              <label className="text-sm text-gray-700 dark:text-gray-300 block">
                V_V35
                <span className="text-xs text-gray-500 dark:text-gray-400 ml-2">
                  value
                </span>
              </label>
              <input
                type="number"
                step="0.01"
                value={V_V35}
                onChange={(e) => setVV35(Number(e.target.value) || 0.0)}
                className="w-full px-3 py-2 border border-gray-300 dark:border-gray-600 rounded-md shadow-sm focus:outline-none focus:ring-blue-500 dark:focus:ring-blue-500 focus:border-blue-500"
                placeholder="0.00"
                aria-label="V feature V_V35"
              />
              <p className="text-xs text-gray-500 dark:text-gray-400 mt-1">
                V feature index 35 (enrichment feature)
              </p>
            </div>
          </div>
        </section>

        {/* Analyze button */}
        <section className="mt-8">
          <button
            type="button"
            className="w-full px-6 py-3 border border-blue-600 text-blue-600 rounded-md font-medium text-sm hover:bg-blue-50 dark:hover:bg-blue-900 focus:outline-none focus:ring-2 focus:ring-blue-500 dark:focus:ring-blue-900"
            onClick={analyzeTransaction}
            disabled={isLoading}
          >
            {isLoading ? 'Analyzing...' : 'Analyze Transaction'}
          </button>
        </section>

        {/* Risk Decision section */}
        <section className="mt-8">
          <h2 className="text-xl font-medium text-gray-900 dark:text-gray-100 mb-4">
            Risk Decision
          </h2>
          {prediction ? (
            <div className="bg-white dark:bg-gray-800 rounded-lg p-6 border border-gray-200 dark:border-gray-700">
              <div className="grid grid-cols-2 gap-4">
                <div>
                  <p className="text-sm text-gray-700 dark:text-gray-300 mb-1">
                    Risk Score
                  </p>
                  <p className="text-2xl font-bold text-gray-900 dark:text-gray-100">
                    {(prediction.risk_score * 100).toFixed(1)}%
                  </p>
                </div>
                <div>
                  <p className="text-sm text-gray-700 dark:text-gray-300 mb-1">
                    Risk Band
                  </p>
                  <p className="text-xl font-medium text-gray-900 dark:text-gray-100">
                    {prediction.risk_band}
                  </p>
                </div>
              </div>
              <div className="mt-4">
                <p className="text-sm text-gray-700 dark:text-gray-300 mb-1">
                  Recommended Action
                </p>
                <p className="text-lg font-medium text-gray-900 dark:text-gray-100">
                  {prediction.recommended_action}
                </p>
              </div>
            </div>
          ) : (prediction === null && !apiError ? (
            <p className="text-gray-500 dark:text-gray-400 text-sm">
              Prediction results will appear here after analysis
            </p>
          ) : (apiError ? (
            <p className="text-red-600 dark:text-red-400 text-sm">
              Error: {apiError}
            </p>
          ) : null)}</section>

{/* Expected Costs section */}
        <section className="mt-8">
          <h2 className="text-xl font-medium text-gray-900 dark:text-gray-100 mb-4">
            Expected Costs
          </h2>
          {prediction ? (
            <div className="bg-white dark:bg-gray-800 rounded-lg p-6 border border-gray-200 dark:border-gray-700">
              <div className="grid grid-cols-3 gap-4">
                <div>
                  <p className="text-sm text-gray-700 dark:text-gray-300 mb-1">
                    Approve
                  </p>
                  <p className="text-lg font-medium text-gray-900 dark:text-gray-100">
                    {prediction.expected_costs.approve}
                  </p>
                </div>
                <div>
                  <p className="text-sm text-gray-700 dark:text-gray-300 mb-1">
                    Review
                  </p>
                  <p className="text-lg font-medium text-gray-900 dark:text-gray-100">
                    {prediction.expected_costs.review}
                  </p>
                </div>
                <div>
                  <p className="text-sm text-gray-700 dark:text-gray-300 mb-1">
                    Escalate
                  </p>
                  <p className="text-lg font-medium text-gray-900 dark:text-gray-100">
                    {prediction.expected_costs.escalate}
                  </p>
                </div>
              </div>
            </div>
          ) : (prediction === null && !apiError ? (
            <p className="text-gray-500 dark:text-gray-400 text-sm">              Modelled expected costs will appear here after analysis
            </p>
          ) : (apiError ? (
            <p className="text-red-600 dark:text-red-400 text-sm">              Error: {apiError}
            </p>
          ) : null)}</section>

{/* Why This Decision? section */}
        <section className="mt-8">
          <h2 className="text-xl font-medium text-gray-900 dark:text-gray-100 mb-4">
            Why This Decision?
          </h2>
          {prediction ? (
            <div className="bg-white dark:bg-gray-800 rounded-lg p-6 border border-gray-200 dark:border-gray-700">
              <div className="grid grid-cols-2 gap-4">
                <h3 className="text-sm font-medium text-gray-700 dark:text-gray-300 mb-3">Top Risk Factors</h3>
                <ul className="space-y-2 text-sm text-gray-700 dark:text-gray-300">
                  {prediction.top_risk_factors.map((factor, idx) => (
                    <li key={idx}>
                      <span className="font-medium">{factor.feature}</span>: {factor.contribution > 0 ? '+' + factor.contribution : factor.contribution}{factor.description ? \". \" + factor.description : ''}
                    </li>
                  ))}
                </ul>
                <h3 className="text-sm font-medium text-gray-700 dark:text-gray-300 mb-3">Protective Factors</h3>
                <ul className="space-y-2 text-sm text-gray-700 dark:text-gray-300">
                  {prediction.top_protective_factors.map((factor, idx) => (
                    <li key={idx}>
                      <span className="font-medium">{factor.feature}</span>: {factor.contribution > 0 ? '+' + factor.contribution : factor.contribution}{factor.description ? \". \" + factor.description : ''}
                    </li>
                  ))}
                </ul>
              </div>
            </div>
          ) : (prediction === null && !apiError ? (
            <p className="text-gray-500 dark:text-gray-400 text-sm">              Risk and protective factors will appear here after analysis
            </p>
          ) : (apiError ? (
            <p className="text-red-600 dark:text-red-400 text-sm">              Error: {apiError}
            </p>
          ) : null)}</section>
      </main>

      <footer className="mt-8 border-t border-gray-200 dark:border-gray-700 pt-8">
        <div className="max-w-7xl mx-auto px-4">
          <p className="text-center text-xs text-gray-500 dark:text-gray-400">
            AI recommends. Human investigators make final decisions.
          </p>
        </div>
      </footer>
    </div>
  )
}

export default App