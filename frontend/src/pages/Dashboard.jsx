
import { useEffect, useState } from "react";
import Navbar from "../components/Navbar";
import {
  CartesianGrid,
  Line,
  LineChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import "./Dashboard.css";

const API_BASE = (
  import.meta.env.VITE_API_BASE_URL || "http://127.0.0.1:8000"
).replace(/\/$/, "");

const CROPS = [
  { value: "soybean", label: "Soybean" },
  { value: "onion", label: "Onion" },
  { value: "wheat", label: "Wheat" },
  { value: "tomato", label: "Tomato" },
];

async function fetchJson(path, signal) {
  const response = await fetch(`${API_BASE}${path}`, { signal });
  const data = await response.json();

  if (!response.ok) {
    throw new Error(
      data.detail || `API request failed (${response.status})`
    );
  }

  return data;
}

function formatPrice(value) {
  if (value === null || value === undefined) return "Not reported";

  return new Intl.NumberFormat("en-IN", {
    style: "currency",
    currency: "INR",
    maximumFractionDigits: 0,
  }).format(value);
}

function FreshnessNotice({ data }) {
  if (!data) return null;

  const status = data.freshness_status || "unknown";
  const age = data.data_age_days;
  const date = data.last_observation_date || "Unavailable";

  const messages = {
    fresh: "The archived observations are relatively recent.",
    aging: "These observations are aging. Check a current market source before making a selling decision.",
    stale: "These observations are stale. Do not rely on them alone for a current selling decision.",
    future_dated: "The latest observation has a future date. Verify the archive.",
    unknown: "The freshness of these observations could not be established.",
  };

  return (
    <section className={`freshness-notice freshness-${status}`}>
      <div>
        <span className="freshness-label">
          DATA FRESHNESS: {status.replace("_", " ").toUpperCase()}
        </span>
        <p>{messages[status] || messages.unknown}</p>
      </div>

      <div className="freshness-meta">
        <strong>{date}</strong>
        <span>
          {Number.isInteger(age) && age >= 0
            ? `${age} day${age === 1 ? "" : "s"} old`
            : "Age unavailable"}
        </span>
      </div>
    </section>
  );
}

function MetricCard({ label, value, detail }) {
  return (
    <div className="mandi-metric">
      <span>{label}</span>
      <strong>{value}</strong>
      <small>{detail}</small>
    </div>
  );
}

export default function Dashboard() {
  const [crop, setCrop] = useState("soybean");
  const [markets, setMarkets] = useState([]);
  const [marketId, setMarketId] = useState("1599");

  const [latest, setLatest] = useState(null);
  const [history, setHistory] = useState(null);
  const [comparison, setComparison] = useState(null);

  const [loadingMarkets, setLoadingMarkets] = useState(true);
  const [loadingData, setLoadingData] = useState(false);
  const [marketError, setMarketError] = useState("");
  const [dataError, setDataError] = useState("");
  const [refreshKey, setRefreshKey] = useState(0);

  // Load the list of Dharashiv markets once.
  useEffect(() => {
    const controller = new AbortController();

    async function loadMarkets() {
      setLoadingMarkets(true);
      setMarketError("");

      try {
        const data = await fetchJson("/markets/local", controller.signal);
        const availableMarkets = data.markets || [];

        setMarkets(availableMarkets);

        setMarketId((current) => {
          const stillAvailable = availableMarkets.some(
            (market) => market.market_id === current
          );

          return stillAvailable
            ? current
            : availableMarkets[0]?.market_id || "";
        });
      } catch (error) {
        if (error.name !== "AbortError") {
          setMarketError(error.message);
        }
      } finally {
        if (!controller.signal.aborted) {
          setLoadingMarkets(false);
        }
      }
    }

    loadMarkets();

    return () => controller.abort();
  }, []);

  // Reload the selected crop and market whenever either changes.
  useEffect(() => {
    if (!crop || !marketId) return;

    const controller = new AbortController();

    async function loadDashboard() {
      setLoadingData(true);
      setDataError("");

      try {
        const [latestData, historyData, comparisonData] =
          await Promise.all([
            fetchJson(
              `/mandi/prices/latest?crop=${encodeURIComponent(crop)}`,
              controller.signal
            ),
            fetchJson(
              `/mandi/prices/history?crop=${encodeURIComponent(crop)}&market_id=${encodeURIComponent(marketId)}`,
              controller.signal
            ),
            fetchJson(
              `/mandi/prices/compare?crop=${encodeURIComponent(crop)}`,
              controller.signal
            ),
          ]);

        setLatest(latestData);
        setHistory(historyData);
        setComparison(comparisonData);
      } catch (error) {
        if (error.name !== "AbortError") {
          setDataError(error.message);
        }
      } finally {
        if (!controller.signal.aborted) {
          setLoadingData(false);
        }
      }
    }

    loadDashboard();

    return () => controller.abort();
  }, [crop, marketId, refreshKey]);

  const selectedMarket = markets.find(
    (market) => market.market_id === marketId
  );

  const records = latest?.records || [];
  const numericCount = latest?.numeric_price_count ?? 0;
  const missingCount = latest?.data_quality?.missing_or_invalid_price_count ?? 0;

  const chartData = (history?.records || []).map((record) => ({
    date: record.observation_date,
    reportedPrice:
      record.reported_price === null
        ? null
        : Number(record.reported_price),
    variety: record.variety || "Unspecified",
  }));

  const comparisonGroups = comparison?.comparisons || [];

  const freshnessData = latest
    ? {
        freshness_status: latest.freshness_status,
        last_observation_date: latest.last_observation_date,
        data_age_days: latest.data_age_days,
      }
    : null;

  return (
    <>
      <Navbar />

      <main className="mandi-dashboard">
        <header className="dashboard-heading">
          <div>
            <p className="dashboard-eyebrow">CROPBALANCE AI / MARKET INTELLIGENCE</p>
            <h1>Dharashiv Mandi Dashboard</h1>
            <p className="dashboard-subtitle">
              Explore archived crop observations, market history and
              comparable reported values.
            </p>
          </div>

          <span className="archive-tag">ARCHIVED DATA · NOT LIVE</span>
        </header>

        <section className="dashboard-controls">
          <label>
            Select crop
            <select value={crop} onChange={(e) => setCrop(e.target.value)}>
              {CROPS.map((item) => (
                <option key={item.value} value={item.value}>
                  {item.label}
                </option>
              ))}
            </select>
          </label>

          <label>
            Select market for history
            <select
              value={marketId}
              onChange={(e) => setMarketId(e.target.value)}
              disabled={loadingMarkets || markets.length === 0}
            >
              {markets.map((market) => (
                <option key={market.market_id} value={market.market_id}>
                  {market.market}
                </option>
              ))}
            </select>
          </label>

          <button
            className="dashboard-refresh"
            onClick={() => setRefreshKey((value) => value + 1)}
            disabled={loadingData}
          >
            {loadingData ? "Loading..." : "Refresh"}
          </button>
        </section>

        {marketError && (
          <div className="dashboard-error">
            Could not load markets: {marketError}
          </div>
        )}

        {dataError && (
          <div className="dashboard-error">
            Could not load mandi information: {dataError}
          </div>
        )}

        {loadingData && !latest && (
          <div className="dashboard-loading">Loading market observations...</div>
        )}

        {latest && (
          <>
            <FreshnessNotice data={freshnessData} />

            <section className="mandi-metrics">
              <MetricCard
                label="Markets covered"
                value={markets.length}
                detail="Dharashiv district"
              />
              <MetricCard
                label="Latest observations"
                value={latest.record_count ?? records.length}
                detail="One latest observation per market"
              />
              <MetricCard
                label="Numeric reported values"
                value={numericCount}
                detail="Not independently verified"
              />
              <MetricCard
                label="Missing / invalid"
                value={missingCount}
                detail="NR is not treated as zero"
              />
            </section>

            <section className="dashboard-section">
              <div className="section-heading">
                <div>
                  <p className="dashboard-eyebrow">MARKET OVERVIEW</p>
                  <h2>Latest {crop} observations</h2>
                </div>
                <span>{latest.district || "Dharashiv"}, Maharashtra</span>
              </div>

              <div className="market-grid">
                {records.map((record) => (
                  <article
                    className={`market-card ${record.market_id === marketId ? "market-card-selected" : ""}`}
                    key={`${record.market_id}-${record.variety}`}
                  >
                    <div className="market-card-top">
                      <h3>{record.market}</h3>
                      <span>{record.market_id}</span>
                    </div>

                    <strong className="market-price">
                      {record.reported_price === null
                        ? "Not reported"
                        : formatPrice(record.reported_price)}
                    </strong>

                    <p>{record.variety || "Unspecified variety"}</p>
                    <small>
                        {record.observation_date} · {
                          record.price_unit &&
                          !["NR", "N.R.", "NA", "N/A"].includes(
                            record.price_unit.trim().toUpperCase()
                          )
                            ? record.price_unit
                            : "Unit unavailable"
                        }
                    </small>

                    {record.reported_price === null && (
                      <span className="not-reported-tag">
                        {record.price_status === "not_reported"
                          ? "NR"
                          : "Unavailable"}
                      </span>
                    )}

                    <button
                      className="market-select-link"
                      onClick={() => setMarketId(record.market_id)}
                    >
                      View history
                    </button>
                  </article>
                ))}
              </div>
            </section>

            <section className="dashboard-section">
              <div className="section-heading">
                <div>
                  <p className="dashboard-eyebrow">HISTORICAL OBSERVATIONS</p>
                  <h2>{selectedMarket?.market || history?.market || "Selected market"}</h2>
                </div>
                <span>{history?.record_count ?? 0} records</span>
              </div>

              <div className="chart-panel">
                {chartData.some((point) => point.reportedPrice !== null) ? (
                  <ResponsiveContainer width="100%" height={320}>
                    <LineChart data={chartData}>
                      <CartesianGrid stroke="#263b50" strokeDasharray="4 4" />
                      <XAxis
                        dataKey="date"
                        stroke="#a8b7c8"
                        tick={{ fontSize: 12 }}
                      />
                      <YAxis
                        stroke="#a8b7c8"
                        tick={{ fontSize: 12 }}
                        tickFormatter={(value) => `₹${value}`}
                        domain={["auto", "auto"]}
                      />
                      <Tooltip
                        contentStyle={{
                          background: "#10243a",
                          border: "1px solid #34516c",
                          borderRadius: 10,
                          color: "#fff",
                        }}
                        formatter={(value) =>
                          value === null ? ["Not reported", "Reported value"] :
                            [formatPrice(value), "Reported value"]
                        }
                      />
                      <Line
                        type="monotone"
                        dataKey="reportedPrice"
                        name="Reported value"
                        stroke="#62d39b"
                        strokeWidth={3}
                        dot={{ r: 4, fill: "#62d39b" }}
                        connectNulls={false}
                      />
                    </LineChart>
                  </ResponsiveContainer>
                ) : (
                  <div className="empty-state">
                    No numeric historical observations are available for this market.
                  </div>
                )}

                <p className="chart-caption">
                  Values are archived AGMARKNET reports. Missing observations
                  remain missing; the line does not represent a live quote.
                </p>
              </div>
            </section>

            <section className="dashboard-section">
              <div className="section-heading">
                <div>
                  <p className="dashboard-eyebrow">COMPARABLE OBSERVATIONS</p>
                  <h2>Market comparison</h2>
                </div>
                <span>{comparisonGroups.length} comparison groups</span>
              </div>

              {comparisonGroups.length === 0 ? (
                <div className="empty-state">
                  {comparison?.message ||
                    "There are not enough matching observations to compare markets."}
                </div>
              ) : (
                <div className="comparison-grid">
                  {comparisonGroups.map((group) => (
                    <article
                      className="comparison-card"
                      key={`${group.observation_date}-${group.variety}-${group.price_unit}`}
                    >
                      <div className="comparison-card-heading">
                        <div>
                          <h3>{group.variety || "Unspecified variety"}</h3>
                          <span>{group.observation_date}</span>
                        </div>
                        <span>{group.market_count} markets</span>
                      </div>

                      <p className="comparison-highest">
                        {formatPrice(group.highest_reported_value)}
                        <small>Highest reported value</small>
                      </p>

                      <p className="comparison-winner">
                        {group.highest_reported_markets.join(", ")}
                      </p>

                      <div className="comparison-list">
                        {group.records_ranked_high_to_low.map((record) => (
                          <div
                            className="comparison-row"
                            key={record.market_id}
                          >
                            <span>{record.market}</span>
                            <strong>{formatPrice(record.reported_price)}</strong>
                          </div>
                        ))}
                      </div>
                    </article>
                  ))}
                </div>
              )}

              <p className="dashboard-caveat">
                A higher reported value does not establish the best place to sell.
                Confirm today's price, crop grade, quantity, transport costs and
                market fees before making a selling decision.
              </p>
            </section>
          </>
        )}

        <footer className="dashboard-footer">
          CropBalance AI · Local Mandi Intelligence · Dharashiv pilot
        </footer>
      </main>
    </>
  );
}
