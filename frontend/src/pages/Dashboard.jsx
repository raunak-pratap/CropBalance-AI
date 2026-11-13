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
import { dashboardTranslations } from "../i18n/dashboard";

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
  const data = await response.json().catch(() => null);

  if (!response.ok) {
    throw new Error(
      data?.detail || `API request failed (${response.status})`
    );
  }

  if (!data || typeof data !== "object") {
    throw new Error("The API returned an invalid response.");
  }

  return data;
}

function formatPrice(value, notReportedText, language) {
  if (value === null || value === undefined) {
    return notReportedText;
  }

  const localeByLanguage = {
    en: "en-IN",
    hi: "hi-IN",
    mr: "mr-IN",
  };

  return new Intl.NumberFormat(localeByLanguage[language] || "en-IN", {
    style: "currency",
    currency: "INR",
    maximumFractionDigits: 0,
  }).format(value);
}

function FreshnessNotice({ data, t }) {
  if (!data) return null;

  const status = data.freshness_status || "unknown";
  const safeStatus = t.freshnessStatuses[status] ? status : "unknown";
  const age = data.data_age_days;
  const date = data.last_observation_date || t.unavailable;

  return (
    <section className={`freshness-notice freshness-${safeStatus}`}>
      <div>
        <span className="freshness-label">
          {t.freshnessTitle}: {t.freshnessStatuses[safeStatus]}
        </span>
        <p>{t.freshnessMessages[safeStatus]}</p>
      </div>

      <div className="freshness-meta">
        <strong>{date}</strong>
        <span>
          {Number.isInteger(age) && age >= 0
            ? t.ageOld(age)
            : t.ageUnavailable}
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
  const [language, setLanguage] = useState(() => {
    try {
      const saved = window.localStorage.getItem(
        "cropbalance-dashboard-language"
      );
      return saved && dashboardTranslations[saved] ? saved : "en";
    } catch {
      return "en";
    }
  });

  const t = dashboardTranslations[language] || dashboardTranslations.en;

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

  // Remember the user's dashboard language on this browser.
  useEffect(() => {
    try {
      window.localStorage.setItem(
        "cropbalance-dashboard-language",
        language
      );
    } catch {
      // The dashboard remains usable when browser storage is unavailable.
    }
  }, [language]);

  // Load the list of Dharashiv markets once.
  useEffect(() => {
    const controller = new AbortController();

    async function loadMarkets() {
      setLoadingMarkets(true);
      setMarketError("");

      try {
        const data = await fetchJson("/markets/local", controller.signal);
        const availableMarkets = Array.isArray(data.markets)
          ? data.markets
          : [];

        setMarkets(availableMarkets);
        setMarketId((current) => {
          const stillAvailable = availableMarkets.some(
            (market) => String(market.market_id) === String(current)
          );

          return stillAvailable
            ? current
            : String(availableMarkets[0]?.market_id || "");
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

  // Reload observations whenever the selected crop, market, or refresh key changes.
  useEffect(() => {
    if (!crop || !marketId) return undefined;

    const controller = new AbortController();

    async function loadDashboard() {
      setLoadingData(true);
      setDataError("");

      try {
        const [latestData, historyData, comparisonData] = await Promise.all([
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
    (market) => String(market.market_id) === String(marketId)
  );

  const records = latest?.records || [];
  const numericCount = latest?.numeric_price_count ?? 0;
  const missingCount =
    latest?.data_quality?.missing_or_invalid_price_count ?? 0;
  const comparisonGroups = comparison?.comparisons || [];

  const chartData = (history?.records || []).map((record) => ({
    date: record.observation_date,
    reportedPrice:
      record.reported_price === null || record.reported_price === undefined
        ? null
        : Number(record.reported_price),
  }));

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
            <p className="dashboard-eyebrow">{t.eyebrow}</p>
            <h1>{t.title}</h1>
            <p className="dashboard-subtitle">{t.subtitle}</p>
          </div>
          <span className="archive-tag">{t.archiveTag}</span>
        </header>

        <section className="dashboard-controls">
          <label>
            {t.language}
          <select
              value={language}
              onChange={(e) => {
                const selectedLanguage = e.target.value;

                setLanguage(selectedLanguage);

                window.localStorage.setItem(
                  "cropbalance-dashboard-language",
                  selectedLanguage
                );
              }}
            >
              <option value="en">English</option>
              <option value="hi">हिन्दी</option>
              <option value="mr">मराठी</option>
            </select>
          </label>

          <label>
            {t.crop}
            <select
              value={crop}
              onChange={(event) => setCrop(event.target.value)}
              aria-label={t.crop}
            >
              {CROPS.map((item) => (
                <option key={item.value} value={item.value}>
                  {t.cropNames[item.value] || item.label}
                </option>
              ))}
            </select>
          </label>

          <label>
            {t.marketForHistory}
            <select
              value={marketId}
              onChange={(event) => setMarketId(event.target.value)}
              disabled={loadingMarkets || markets.length === 0}
              aria-label={t.marketForHistory}
            >
              {markets.map((market) => (
                <option key={market.market_id} value={market.market_id}>
                  {market.market}
                </option>
              ))}
            </select>
          </label>

          <button
            type="button"
            className="dashboard-refresh"
            onClick={() => setRefreshKey((value) => value + 1)}
            disabled={loadingData || !marketId}
          >
            {loadingData ? t.loading : t.refresh}
          </button>
        </section>

        {marketError && (
          <div className="dashboard-error" role="alert">
            {t.loadingMarketsError}: {marketError}
          </div>
        )}

        {dataError && (
          <div className="dashboard-error" role="alert">
            {t.loadingDataError}: {dataError}
          </div>
        )}

        {loadingData && !latest && (
          <div className="dashboard-loading" role="status">
            {t.loading}
          </div>
        )}

        {latest && (
          <>
            <FreshnessNotice data={freshnessData} t={t} />

            <section className="mandi-metrics">
              <MetricCard
                label={t.marketsCovered}
                value={markets.length}
                detail={t.district}
              />
              <MetricCard
                label={t.latestObservations}
                value={latest.record_count ?? records.length}
                detail={t.onePerMarket}
              />
              <MetricCard
                label={t.numericValues}
                value={numericCount}
                detail={t.notVerified}
              />
              <MetricCard
                label={t.missingInvalid}
                value={missingCount}
                detail={t.nrNotZero}
              />
            </section>

            <section className="dashboard-section">
              <div className="section-heading">
                <div>
                  <p className="dashboard-eyebrow">{t.marketOverview}</p>
                  <h2>
                    {t.latestCropObservations(t.cropNames[crop] || crop)}
                  </h2>
                </div>
                <span>{latest.district || t.districtLocation}</span>
              </div>

              <div className="market-grid">
                {records.map((record) => {
                  const unit = (record.price_unit || "").trim();
                  const usableUnit = unit &&
                    !["NR", "N.R.", "NA", "N/A"].includes(
                      unit.toUpperCase()
                    );

                  return (
                    <article
                      className={`market-card ${String(record.market_id) === String(marketId) ? "market-card-selected" : ""}`}
                      key={`${record.market_id}-${record.variety || "unknown"}`}
                    >
                      <div className="market-card-top">
                        <h3>{record.market}</h3>
                        <span>{record.market_id}</span>
                      </div>

                      <strong className="market-price">
                        {formatPrice(
                          record.reported_price,
                          t.notReported,
                          language
                        )}
                      </strong>

                      <p>{record.variety || t.unspecifiedVariety}</p>
                      <small>
                        {record.observation_date} · {usableUnit ? unit : t.unitUnavailable}
                      </small>

                      {record.reported_price === null && (
                        <span className="not-reported-tag">
                          {record.price_status === "not_reported"
                            ? t.nr
                            : t.unavailable}
                        </span>
                      )}

                      <button
                        type="button"
                        className="market-select-link"
                        onClick={() => setMarketId(String(record.market_id))}
                      >
                        {t.viewHistory}
                      </button>
                    </article>
                  );
                })}
              </div>
            </section>

            <section className="dashboard-section">
              <div className="section-heading">
                <div>
                  <p className="dashboard-eyebrow">
                    {t.historicalObservations}
                  </p>
                  <h2>
                    {selectedMarket?.market || history?.market || t.selectedMarket}
                  </h2>
                </div>
                <span>{t.records(history?.record_count ?? 0)}</span>
              </div>

              <div className="chart-panel">
                {chartData.some((point) => point.reportedPrice !== null) ? (
                  <ResponsiveContainer width="100%" height={320}>
                    <LineChart data={chartData}>
                      <CartesianGrid
                        stroke="#263b50"
                        strokeDasharray="4 4"
                      />
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
                          value === null || value === undefined
                            ? [t.notReported, t.chartReportedValue]
                            : [
                                formatPrice(value, t.notReported, language),
                                t.chartReportedValue,
                              ]
                        }
                      />
                      <Line
                        type="monotone"
                        dataKey="reportedPrice"
                        name={t.chartReportedValue}
                        stroke="#62d39b"
                        strokeWidth={3}
                        dot={{ r: 4, fill: "#62d39b" }}
                        connectNulls={false}
                      />
                    </LineChart>
                  </ResponsiveContainer>
                ) : (
                  <div className="empty-state">{t.noNumericHistory}</div>
                )}

                <p className="chart-caption">{t.chartCaption}</p>
              </div>
            </section>

            <section className="dashboard-section">
              <div className="section-heading">
                <div>
                  <p className="dashboard-eyebrow">
                    {t.comparableObservations}
                  </p>
                  <h2>{t.marketComparison}</h2>
                </div>
                <span>{t.comparisonGroups(comparisonGroups.length)}</span>
              </div>

              {comparisonGroups.length === 0 ? (
                <div className="empty-state">{t.noComparisons}</div>
              ) : (
                <div className="comparison-grid">
                  {comparisonGroups.map((group) => (
                    <article
                      className="comparison-card"
                      key={`${group.observation_date}-${group.variety}-${group.price_unit}`}
                    >
                      <div className="comparison-card-heading">
                        <div>
                          {/* Preserve the original AGMARKNET variety label. */}
                          <h3>{group.variety || t.unspecifiedVariety}</h3>
                          <span>{group.observation_date}</span>
                        </div>
                        <span>{t.markets(group.market_count)}</span>
                      </div>

                      <p className="comparison-highest">
                        {formatPrice(
                          group.highest_reported_value,
                          t.notReported,
                          language
                        )}
                        <small>{t.highestReported}</small>
                      </p>

                      <p className="comparison-winner">
                        {(group.highest_reported_markets || []).join(", ")}
                      </p>

                      <div className="comparison-list">
                        {(group.records_ranked_high_to_low || []).map(
                          (record) => (
                            <div
                              className="comparison-row"
                              key={record.market_id}
                            >
                              <span>{record.market}</span>
                              <strong>
                                {formatPrice(
                                  record.reported_price,
                                  t.notReported,
                                  language
                                )}
                              </strong>
                            </div>
                          )
                        )}
                      </div>
                    </article>
                  ))}
                </div>
              )}

              <p className="dashboard-caveat">{t.dataCaveat}</p>
            </section>
          </>
        )}

        <footer className="dashboard-footer">{t.footer}</footer>
      </main>
    </>
  );
}
