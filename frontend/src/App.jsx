import React, { useState, useEffect } from 'react';
import { motion } from 'framer-motion';
import {
  LineChart, Line, BarChart, Bar, PieChart, Pie, Cell,
  XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer
} from 'recharts';
import { Sun, Wind, Zap, AlertTriangle, CheckCircle, TrendingUp, DollarSign, Target } from 'lucide-react';
import './App.css';

const COLORS = {
  solar: '#FDB813',
  wind: '#00A9CE',
  shortage: '#EF4444',
  success: '#10B981',
  profit: '#8B5CF6'
};

// Pricing Constants (₹ per MW) - Indian Rupees
const SOLAR_PRICE = 10000;  // What we sell solar energy for (₹10,000/MW)
const WIND_PRICE = 7500;    // What we sell wind energy for (₹7,500/MW)
const SOLAR_COST = 3500;    // What it costs us to generate solar (₹3,500/MW)
const WIND_COST = 2500;     // What it costs us to generate wind (₹2,500/MW)

function App() {
  const [demand, setDemand] = useState('');
  const [budget, setBudget] = useState('');
  const [prediction, setPrediction] = useState(null);
  const [optimization, setOptimization] = useState(null);
  const [loading, setLoading] = useState(false);
  const [historicalData, setHistoricalData] = useState({ solar: [], wind: [] });
  const [error, setError] = useState(null);

  // Fetch historical data on mount
  useEffect(() => {
    fetchHistoricalData();
  }, []);

  const fetchHistoricalData = async () => {
    try {
      const response = await fetch('http://localhost:5000/api/historical?hours=73');
      const data = await response.json();
      setHistoricalData(data);
    } catch (err) {
      console.error('Error fetching historical data:', err);
    }
  };

  // Linear Programming Optimization
  const optimizeSales = (solarAvail, windAvail, demandMW, budgetAmount) => {
    const totalAvailable = solarAvail + windAvail;
    
    // Case 1: Energy shortage - Cannot meet demand
    if (totalAvailable < demandMW) {
      const totalRevenue = (solarAvail * SOLAR_PRICE) + (windAvail * WIND_PRICE);
      const totalCost = (solarAvail * SOLAR_COST) + (windAvail * WIND_COST);
      const totalProfit = totalRevenue - totalCost;
      const shortage = demandMW - totalAvailable;
      
      return {
        solarSold: solarAvail,
        windSold: windAvail,
        totalSold: totalAvailable,
        totalRevenue: totalRevenue,
        totalCost: totalCost,
        totalProfit: totalProfit,
        demandMet: false,
        budgetExceeded: false,
        energyShortage: shortage,
        optimized: false,
        verdict: 'Energy Shortage Detected! Cannot meet demand',
        shortageMessage: `Short by ${shortage.toFixed(2)} MW. Available: ${totalAvailable.toFixed(2)} MW, Required: ${demandMW.toFixed(2)} MW`
      };
    }
    
    // Case 2: Can meet demand exactly (total available == demand)
    if (totalAvailable === demandMW) {
      const totalRevenue = (solarAvail * SOLAR_PRICE) + (windAvail * WIND_PRICE);
      const totalCost = (solarAvail * SOLAR_COST) + (windAvail * WIND_COST);
      const totalProfit = totalRevenue - totalCost;
      
      // Check if budget is sufficient
      if (totalRevenue > budgetAmount) {
        return {
          solarSold: 0,
          windSold: 0,
          totalSold: 0,
          totalRevenue: 0,
          totalCost: 0,
          totalProfit: 0,
          demandMet: false,
          budgetExceeded: true,
          budgetShortage: totalRevenue - budgetAmount,
          optimized: false,
          verdict: 'Budget Insufficient! Cannot afford required energy',
          budgetMessage: `Need ₹${totalRevenue.toLocaleString('en-IN')} but budget is only ₹${budgetAmount.toLocaleString('en-IN')}. Short by ₹${(totalRevenue - budgetAmount).toLocaleString('en-IN')}`
        };
      }
      
      return {
        solarSold: solarAvail,
        windSold: windAvail,
        totalSold: totalAvailable,
        totalRevenue: totalRevenue,
        totalCost: totalCost,
        totalProfit: totalProfit,
        demandMet: true,
        budgetExceeded: false,
        optimized: false,
        verdict: 'Perfect Match! Selling all available energy within budget'
      };
    }

    // Optimization needed: total available > demand
    const solarProfitPerMW = SOLAR_PRICE - SOLAR_COST;
    const windProfitPerMW = WIND_PRICE - WIND_COST;

    let bestSolution = null;
    let maxProfit = -Infinity;

    // Check corner points of feasible region
    const candidates = [];

    // Corner 1: Maximum solar within budget
    const maxSolarInBudget = Math.min(solarAvail, budgetAmount / SOLAR_PRICE);
    const windNeededForMaxSolar = Math.max(0, demandMW - maxSolarInBudget);
    
    if (windNeededForMaxSolar <= windAvail) {
      const revenue1 = maxSolarInBudget * SOLAR_PRICE + windNeededForMaxSolar * WIND_PRICE;
      if (revenue1 <= budgetAmount && maxSolarInBudget + windNeededForMaxSolar >= demandMW) {
        candidates.push({
          solar: maxSolarInBudget,
          wind: windNeededForMaxSolar,
          revenue: revenue1
        });
      }
    }

    // Corner 2: Maximum wind within budget
    const maxWindInBudget = Math.min(windAvail, budgetAmount / WIND_PRICE);
    const solarNeededForMaxWind = Math.max(0, demandMW - maxWindInBudget);
    
    if (solarNeededForMaxWind <= solarAvail) {
      const revenue2 = solarNeededForMaxWind * SOLAR_PRICE + maxWindInBudget * WIND_PRICE;
      if (revenue2 <= budgetAmount && solarNeededForMaxWind + maxWindInBudget >= demandMW) {
        candidates.push({
          solar: solarNeededForMaxWind,
          wind: maxWindInBudget,
          revenue: revenue2
        });
      }
    }

    // Corner 3: Meet demand exactly with maximum solar
    const solarForDemand = Math.min(solarAvail, demandMW);
    const windForDemand = Math.max(0, demandMW - solarForDemand);
    
    if (windForDemand <= windAvail) {
      const revenue3 = solarForDemand * SOLAR_PRICE + windForDemand * WIND_PRICE;
      if (revenue3 <= budgetAmount) {
        candidates.push({
          solar: solarForDemand,
          wind: windForDemand,
          revenue: revenue3
        });
      }
    }

    // Corner 4: Budget-constrained optimal mix
    if (SOLAR_PRICE !== WIND_PRICE) {
      const wind4 = (budgetAmount - demandMW * SOLAR_PRICE) / (WIND_PRICE - SOLAR_PRICE);
      const solar4 = demandMW - wind4;
      
      if (solar4 >= 0 && solar4 <= solarAvail && wind4 >= 0 && wind4 <= windAvail) {
        candidates.push({
          solar: solar4,
          wind: wind4,
          revenue: budgetAmount
        });
      }
    }

    // Corner 5: All available solar + wind up to demand
    const solar5 = Math.min(solarAvail, demandMW);
    const wind5 = Math.min(windAvail, demandMW - solar5);
    const revenue5 = solar5 * SOLAR_PRICE + wind5 * WIND_PRICE;
    
    if (revenue5 <= budgetAmount && solar5 + wind5 >= demandMW) {
      candidates.push({
        solar: solar5,
        wind: wind5,
        revenue: revenue5
      });
    }

    // Evaluate all candidates and find maximum profit
    candidates.forEach(candidate => {
      const cost = candidate.solar * SOLAR_COST + candidate.wind * WIND_COST;
      const profit = candidate.revenue - cost;
      
      if (profit > maxProfit) {
        maxProfit = profit;
        bestSolution = {
          solarSold: candidate.solar,
          windSold: candidate.wind,
          totalRevenue: candidate.revenue,
          totalCost: cost,
          totalProfit: profit
        };
      }
    });

    if (!bestSolution) {
      // No feasible solution - budget too low
      const minRevenue = demandMW * Math.min(SOLAR_PRICE, WIND_PRICE);
      return {
        solarSold: 0,
        windSold: 0,
        totalSold: 0,
        totalRevenue: 0,
        totalCost: 0,
        totalProfit: 0,
        demandMet: false,
        budgetExceeded: true,
        budgetShortage: minRevenue - budgetAmount,
        optimized: false,
        verdict: 'Budget Insufficient! Cannot meet demand within budget',
        minBudgetNeeded: minRevenue,
        budgetMessage: `Minimum budget needed: ₹${minRevenue.toLocaleString('en-IN')}. Current budget: ₹${budgetAmount.toLocaleString('en-IN')}. Short by: ₹${(minRevenue - budgetAmount).toLocaleString('en-IN')}`
      };
    }

    return {
      ...bestSolution,
      totalSold: bestSolution.solarSold + bestSolution.windSold,
      demandMet: bestSolution.solarSold + bestSolution.windSold >= demandMW,
      budgetExceeded: false,
      optimized: true,
      verdict: 'Optimized for maximum profit within constraints'
    };
  };

  const handlePredict = async (e) => {
    e.preventDefault();
    setLoading(true);
    setError(null);

    try {
      const response = await fetch('http://localhost:5000/api/predict', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ demand: parseFloat(demand) })
      });

      if (!response.ok) {
        const errorText = await response.text();
        throw new Error(`Prediction failed: ${errorText}`);
      }

      const data = await response.json();
      
      // Check if there's an error in the response
      if (data.error) {
        throw new Error(data.error);
      }
      
      console.log('Prediction data received:', data);
      
      // Transform the response to match expected format
      const transformedData = {
        solar_pred: data.solar_pred || 0,
        wind_pred: data.wind_pred || 0,
        total_available: data.total_available || (data.solar_pred + data.wind_pred),
        can_meet_demand: data.can_meet_demand,
        shortage: data.shortage || 0,
        timestamp: data.timestamp
      };
      
      setPrediction(transformedData);

      // Run optimization
      const budgetAmount = parseFloat(budget) || Infinity;
      const optimizationResult = optimizeSales(
        transformedData.solar_pred,
        transformedData.wind_pred,
        parseFloat(demand),
        budgetAmount
      );
      
      setOptimization(optimizationResult);
    } catch (err) {
      console.error('Prediction error:', err);
      setError(err.message || 'Failed to get prediction. Please check if the backend is running.');
    } finally {
      setLoading(false);
    }
  };

  // Prepare chart data
  const prepareTimeSeriesData = () => {
    if (!historicalData.solar || !historicalData.wind) return [];
    
    const maxLen = Math.min(historicalData.solar.length, historicalData.wind.length);
    const data = [];
    
    for (let i = 0; i < maxLen; i++) {
      data.push({
        time: new Date(historicalData.solar[i].timestamp).toLocaleTimeString([], { 
          hour: '2-digit', 
          minute: '2-digit' 
        }),
        solar: historicalData.solar[i].value,
        wind: historicalData.wind[i].value,
        isForecast: false
      });
    }
    
    if (prediction) {
      data.push({
        time: 'Forecast',
        solar: prediction.solar_pred,
        wind: prediction.wind_pred,
        isForecast: true
      });
    }
    
    return data;
  };

  const preparePieData = () => {
    if (!optimization) return [];
    return [
      { name: 'Solar', value: optimization.solarSold, color: COLORS.solar },
      { name: 'Wind', value: optimization.windSold, color: COLORS.wind }
    ];
  };

  const prepareBarData = () => {
    if (!prediction || !optimization) return [];
    return [
      {
        name: 'Energy',
        demand: parseFloat(demand),
        sold: optimization.totalSold,
        available: prediction.total_available
      }
    ];
  };

  const prepareProfitData = () => {
    if (!optimization) return [];
    return [
      {
        name: 'Solar',
        revenue: optimization.solarSold * SOLAR_PRICE,
        cost: optimization.solarSold * SOLAR_COST,
        profit: optimization.solarSold * (SOLAR_PRICE - SOLAR_COST)
      },
      {
        name: 'Wind',
        revenue: optimization.windSold * WIND_PRICE,
        cost: optimization.windSold * WIND_COST,
        profit: optimization.windSold * (WIND_PRICE - WIND_COST)
      }
    ];
  };

  return (
    <div className="app">
      <div className="gradient-bg" />
      
      <div className="container">
        {/* Header */}
        <motion.header
          initial={{ opacity: 0, y: -20 }}
          animate={{ opacity: 1, y: 0 }}
          className="header"
        >
          <div className="header-content">
            <Zap className="header-icon" size={40} />
            <div>
              <h1 className="header-title">Renewable Energy Forecasting & Optimization</h1>
              <p className="header-subtitle">Smart Solar & Wind Energy Sales with Profit Maximization</p>
            </div>
          </div>
        </motion.header>

        <div className="main-grid">
          {/* Left Section - Input & Metrics */}
          <motion.div
            initial={{ opacity: 0, x: -20 }}
            animate={{ opacity: 1, x: 0 }}
            transition={{ delay: 0.2 }}
            className="left-section"
          >
            {/* Input Card */}
            <div className="card input-card">
              <h2 className="card-title">
                <TrendingUp size={24} />
                Client Requirements
              </h2>
              <form onSubmit={handlePredict}>
                <div className="input-group">
                  <label htmlFor="demand">Energy Demand (MW)</label>
                  <input
                    type="number"
                    id="demand"
                    value={demand}
                    onChange={(e) => setDemand(e.target.value)}
                    placeholder="e.g., 5000"
                    required
                    min="0"
                    step="0.01"
                  />
                </div>
                <div className="input-group">
                  <label htmlFor="budget">Client Budget (₹)</label>
                  <input
                    type="number"
                    id="budget"
                    value={budget}
                    onChange={(e) => setBudget(e.target.value)}
                    placeholder="e.g., 40000000 (optional)"
                    min="0"
                    step="0.01"
                  />
                  <small style={{color: 'rgba(255,255,255,0.6)', fontSize: '0.85rem'}}>
                    Leave empty for no budget constraint
                  </small>
                </div>
                <button type="submit" className="btn-primary" disabled={loading}>
                  {loading ? (
                    <span className="loading-spinner" />
                  ) : (
                    'Calculate Optimal Sales'
                  )}
                </button>
              </form>
            </div>

            {/* Pricing Info Card */}
            <div className="card pricing-card">
              <h3 className="card-title">
                <DollarSign size={20} />
                Pricing Structure
              </h3>
              <div className="pricing-grid">
                <div className="pricing-item">
                  <span className="pricing-label">Solar Price:</span>
                  <span className="pricing-value solar-text">₹{SOLAR_PRICE.toLocaleString('en-IN')}/MW</span>
                </div>
                <div className="pricing-item">
                  <span className="pricing-label">Solar Cost:</span>
                  <span className="pricing-value">₹{SOLAR_COST.toLocaleString('en-IN')}/MW</span>
                </div>
                <div className="pricing-item">
                  <span className="pricing-label">Solar Profit:</span>
                  <span className="pricing-value success">₹{(SOLAR_PRICE - SOLAR_COST).toLocaleString('en-IN')}/MW</span>
                </div>
                <div className="pricing-item">
                  <span className="pricing-label">Wind Price:</span>
                  <span className="pricing-value wind-text">₹{WIND_PRICE.toLocaleString('en-IN')}/MW</span>
                </div>
                <div className="pricing-item">
                  <span className="pricing-label">Wind Cost:</span>
                  <span className="pricing-value">₹{WIND_COST.toLocaleString('en-IN')}/MW</span>
                </div>
                <div className="pricing-item">
                  <span className="pricing-label">Wind Profit:</span>
                  <span className="pricing-value success">₹{(WIND_PRICE - WIND_COST).toLocaleString('en-IN')}/MW</span>
                </div>
              </div>
            </div>

            {/* Error Display */}
            {error && (
              <motion.div
                initial={{ opacity: 0, scale: 0.9 }}
                animate={{ opacity: 1, scale: 1 }}
                className="alert alert-error"
              >
                <AlertTriangle size={20} />
                <div>
                  <strong>Error:</strong> {error}
                  <br />
                  <small style={{marginTop: '0.5rem', display: 'block'}}>
                    Make sure the backend is running on http://localhost:5000
                  </small>
                </div>
              </motion.div>
            )}

            {/* Prediction Cards */}
            {prediction && (
              <>
                <motion.div
                  initial={{ opacity: 0, y: 20 }}
                  animate={{ opacity: 1, y: 0 }}
                  className="metrics-grid"
                >
                  <div className="metric-card solar-card">
                    <Sun className="metric-icon" size={32} />
                    <div className="metric-content">
                      <p className="metric-label">Solar Available</p>
                      <p className="metric-value">{prediction.solar_pred.toFixed(2)} MW</p>
                    </div>
                  </div>

                  <div className="metric-card wind-card">
                    <Wind className="metric-icon" size={32} />
                    <div className="metric-content">
                      <p className="metric-label">Wind Available</p>
                      <p className="metric-value">{prediction.wind_pred.toFixed(2)} MW</p>
                    </div>
                  </div>

                  <div className="metric-card total-card">
                    <Zap className="metric-icon" size={32} />
                    <div className="metric-content">
                      <p className="metric-label">Total Available</p>
                      <p className="metric-value">{prediction.total_available.toFixed(2)} MW</p>
                    </div>
                  </div>
                </motion.div>

                {/* Optimized Sales Card */}
                {optimization && (
                  <motion.div
                    initial={{ opacity: 0, y: 20 }}
                    animate={{ opacity: 1, y: 0 }}
                    transition={{ delay: 0.1 }}
                    className="card optimization-card"
                  >
                    <h3 className="card-title">
                      <Target size={24} />
                      Optimized Sales
                    </h3>
                    <div className="optimization-grid">
                      <div className="optimization-item">
                        <span className="optimization-label">Solar to Sell:</span>
                        <span className="optimization-value solar-text">
                          {optimization.solarSold.toFixed(2)} MW
                        </span>
                      </div>
                      <div className="optimization-item">
                        <span className="optimization-label">Wind to Sell:</span>
                        <span className="optimization-value wind-text">
                          {optimization.windSold.toFixed(2)} MW
                        </span>
                      </div>
                      <div className="optimization-item">
                        <span className="optimization-label">Total Sold:</span>
                        <span className="optimization-value">
                          {optimization.totalSold.toFixed(2)} MW
                        </span>
                      </div>
                    </div>
                  </motion.div>
                )}

                {/* Optimization Summary Card */}
                {optimization && (
                  <motion.div
                    initial={{ opacity: 0, y: 20 }}
                    animate={{ opacity: 1, y: 0 }}
                    transition={{ delay: 0.2 }}
                    className={`card result-card ${
                      optimization.demandMet && !optimization.budgetExceeded ? 'success' : 
                      optimization.energyShortage || optimization.budgetShortage ? 'error' : 'warning'
                    }`}
                  >
                    {optimization.demandMet && !optimization.budgetExceeded ? (
                      <CheckCircle className="result-icon" size={48} />
                    ) : (
                      <AlertTriangle className="result-icon" size={48} />
                    )}
                    
                    <h3 className="result-title">{optimization.verdict}</h3>
                    
                    {/* Energy Shortage Message */}
                    {optimization.energyShortage && (
                      <div className="shortage-alert energy-shortage">
                        <AlertTriangle size={24} />
                        <div>
                          <p className="shortage-title">⚡ Energy Shortage Detected!</p>
                          <p className="shortage-detail">{optimization.shortageMessage}</p>
                          <p className="shortage-info">
                            Selling all available: {optimization.totalSold.toFixed(2)} MW
                          </p>
                        </div>
                      </div>
                    )}
                    
                    {/* Budget Shortage Message */}
                    {optimization.budgetShortage && (
                      <div className="shortage-alert budget-shortage">
                        <AlertTriangle size={24} />
                        <div>
                          <p className="shortage-title">💰 Budget Insufficient!</p>
                          <p className="shortage-detail">{optimization.budgetMessage}</p>
                        </div>
                      </div>
                    )}
                    
                    <div className="distribution-info">
                      <div className="distribution-item">
                        <span className="distribution-label">Total Revenue:</span>
                        <span className="distribution-value success">
                          ₹{optimization.totalRevenue.toLocaleString('en-IN', {minimumFractionDigits: 2, maximumFractionDigits: 2})}
                        </span>
                      </div>
                      <div className="distribution-item">
                        <span className="distribution-label">Total Cost:</span>
                        <span className="distribution-value">
                          ₹{optimization.totalCost.toLocaleString('en-IN', {minimumFractionDigits: 2, maximumFractionDigits: 2})}
                        </span>
                      </div>
                      <div className="distribution-item profit-highlight">
                        <span className="distribution-label">Total Profit:</span>
                        <span className="distribution-value profit-text">
                          ₹{optimization.totalProfit.toLocaleString('en-IN', {minimumFractionDigits: 2, maximumFractionDigits: 2})}
                        </span>
                      </div>
                      {budget && !optimization.budgetShortage && (
                        <div className="distribution-item">
                          <span className="distribution-label">Budget Remaining:</span>
                          <span className="distribution-value">
                            ₹{(parseFloat(budget) - optimization.totalRevenue).toLocaleString('en-IN', {minimumFractionDigits: 2, maximumFractionDigits: 2})}
                          </span>
                        </div>
                      )}
                      {optimization.energyShortage && (
                        <div className="distribution-item shortage-item">
                          <span className="distribution-label">Energy Shortage:</span>
                          <span className="distribution-value shortage-text">
                            {optimization.energyShortage.toFixed(2)} MW
                          </span>
                        </div>
                      )}
                      {optimization.minBudgetNeeded && optimization.budgetShortage && (
                        <div className="distribution-item shortage-item">
                          <span className="distribution-label">Minimum Budget Needed:</span>
                          <span className="distribution-value shortage-text">
                            ₹{optimization.minBudgetNeeded.toLocaleString('en-IN', {minimumFractionDigits: 2, maximumFractionDigits: 2})}
                          </span>
                        </div>
                      )}
                    </div>
                  </motion.div>
                )}
              </>
            )}
          </motion.div>

          {/* Right Section - Charts */}
          <motion.div
            initial={{ opacity: 0, x: 20 }}
            animate={{ opacity: 1, x: 0 }}
            transition={{ delay: 0.3 }}
            className="right-section"
          >
            {/* Time Series Chart */}
            <div className="card chart-card">
              <h3 className="chart-title">72-Hour Historical Data + Forecast</h3>
              <ResponsiveContainer width="100%" height={300}>
                <LineChart data={prepareTimeSeriesData()}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#e0e0e0" />
                  <XAxis 
                    dataKey="time" 
                    stroke="#666"
                    tick={{ fontSize: 12 }}
                    interval="preserveStartEnd"
                  />
                  <YAxis stroke="#666" tick={{ fontSize: 12 }} />
                  <Tooltip 
                    contentStyle={{ 
                      backgroundColor: 'rgba(255, 255, 255, 0.95)',
                      border: '1px solid #ddd',
                      borderRadius: '8px'
                    }}
                  />
                  <Legend />
                  <Line 
                    type="monotone" 
                    dataKey="solar" 
                    stroke={COLORS.solar} 
                    strokeWidth={2}
                    dot={{ fill: COLORS.solar, r: 2 }}
                    name="Solar (MW)"
                  />
                  <Line 
                    type="monotone" 
                    dataKey="wind" 
                    stroke={COLORS.wind} 
                    strokeWidth={2}
                    dot={{ fill: COLORS.wind, r: 2 }}
                    name="Wind (MW)"
                  />
                </LineChart>
              </ResponsiveContainer>
            </div>

            {/* Distribution Charts */}
            {optimization && (
              <div className="charts-grid">
                {/* Sales Distribution Pie Chart */}
                <div className="card chart-card">
                  <h3 className="chart-title">Sales Distribution</h3>
                  <ResponsiveContainer width="100%" height={250}>
                    <PieChart>
                      <Pie
                        data={preparePieData()}
                        cx="50%"
                        cy="50%"
                        labelLine={false}
                        label={(entry) => `${entry.name}: ${entry.value.toFixed(1)} MW`}
                        outerRadius={80}
                        fill="#8884d8"
                        dataKey="value"
                      >
                        {preparePieData().map((entry, index) => (
                          <Cell key={`cell-${index}`} fill={entry.color} />
                        ))}
                      </Pie>
                      <Tooltip />
                    </PieChart>
                  </ResponsiveContainer>
                </div>

                {/* Demand vs Supply Bar Chart */}
                <div className="card chart-card">
                  <h3 className="chart-title">Demand vs Supply vs Sales</h3>
                  <ResponsiveContainer width="100%" height={250}>
                    <BarChart data={prepareBarData()}>
                      <CartesianGrid strokeDasharray="3 3" stroke="#e0e0e0" />
                      <XAxis dataKey="name" stroke="#666" />
                      <YAxis stroke="#666" />
                      <Tooltip 
                        contentStyle={{ 
                          backgroundColor: 'rgba(255, 255, 255, 0.95)',
                          border: '1px solid #ddd',
                          borderRadius: '8px'
                        }}
                      />
                      <Legend />
                      <Bar dataKey="demand" fill="#9333EA" name="Demand (MW)" />
                      <Bar dataKey="sold" fill="#10B981" name="Sold (MW)" />
                      <Bar dataKey="available" fill="#F59E0B" name="Available (MW)" />
                    </BarChart>
                  </ResponsiveContainer>
                </div>

                {/* Profit Breakdown Chart */}
                <div className="card chart-card">
                  <h3 className="chart-title">Revenue, Cost & Profit Breakdown</h3>
                  <ResponsiveContainer width="100%" height={250}>
                    <BarChart data={prepareProfitData()}>
                      <CartesianGrid strokeDasharray="3 3" stroke="#e0e0e0" />
                      <XAxis dataKey="name" stroke="#666" />
                      <YAxis stroke="#666" />
                      <Tooltip 
                        contentStyle={{ 
                          backgroundColor: 'rgba(255, 255, 255, 0.95)',
                          border: '1px solid #ddd',
                          borderRadius: '8px'
                        }}
                        formatter={(value) => `₹${value.toLocaleString('en-IN')}`}
                      />
                      <Legend />
                      <Bar dataKey="revenue" fill="#10B981" name="Revenue (₹)" />
                      <Bar dataKey="cost" fill="#EF4444" name="Cost (₹)" />
                      <Bar dataKey="profit" fill="#8B5CF6" name="Profit (₹)" />
                    </BarChart>
                  </ResponsiveContainer>
                </div>
              </div>
            )}
          </motion.div>
        </div>
      </div>
    </div>
  );
}

export default App;