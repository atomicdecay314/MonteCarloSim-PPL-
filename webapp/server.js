const express = require('express');
const path = require('path');
const YahooFinance = require('yahoo-finance2').default;

const yf = new YahooFinance({ suppressNotices: ['yahooSurvey'] });
const app = express();
const PORT = 3000;

app.use(express.static(path.join(__dirname, 'public')));

// ── Stock data endpoint ────────────────────────────────────────────────────────
app.get('/api/stock/:ticker', async (req, res) => {
  const ticker = req.params.ticker.toUpperCase();
  try {
    const quote = await yf.quote(ticker);
    const price = quote.regularMarketPrice;
    if (!price) return res.status(404).json({ error: 'Ticker not found' });

    // Try ATM implied volatility from nearest options expiry
    let iv = null;
    try {
      const optData = await yf.options(ticker);
      const calls = optData?.options?.[0]?.calls ?? [];
      if (calls.length) {
        const atm = calls.reduce((best, c) =>
          Math.abs(c.strike - price) < Math.abs(best.strike - price) ? c : best,
          calls[0]
        );
        if (atm?.impliedVolatility > 0) iv = atm.impliedVolatility;
      }
    } catch (_) { /* options unavailable */ }

    // Fallback: summary detail IV
    if (!iv) {
      try {
        const summary = await yf.quoteSummary(ticker, { modules: ['summaryDetail'] });
        iv = summary?.summaryDetail?.impliedVolatility ?? null;
      } catch (_) { /* ignore */ }
    }

    res.json({
      ticker,
      price: +price.toFixed(2),
      iv: iv ? +(iv).toFixed(4) : null,
    });
  } catch (err) {
    res.status(500).json({ error: err.message });
  }
});

app.listen(PORT, () => {
  console.log(`Monte Carlo Simulator running at http://localhost:${PORT}`);
});
