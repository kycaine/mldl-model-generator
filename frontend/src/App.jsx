import React, { useState } from 'react';
import axios from 'axios';
import { Upload, Database, Settings, Activity, Download, CheckCircle, ChevronRight, Play } from 'lucide-react';
import {
  ResponsiveContainer,
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  BarChart,
  Bar
} from 'recharts';

const API_URL = 'http://localhost:8000';

function App() {
  const [step, setStep] = useState(1);
  const [fileId, setFileId] = useState(null);
  const [schema, setSchema] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  // Selections
  const [targetColumn, setTargetColumn] = useState('');
  const [featureColumns, setFeatureColumns] = useState([]);
  const [modelType, setModelType] = useState('classification');
  const [modelName, setModelName] = useState('Logistic Regression');
  const [metrics, setMetrics] = useState(null);
  const [plotData, setPlotData] = useState([]);
  const [downloadFormat, setDownloadFormat] = useState('joblib');

  const handleFileUpload = async (e) => {
    const file = e.target.files[0];
    if (!file) return;

    setLoading(true);
    setError(null);
    const formData = new FormData();
    formData.append('file', file);

    try {
      const res = await axios.post(`${API_URL}/upload`, formData);
      setFileId(res.data.file_id);

      const schemaRes = await axios.get(`${API_URL}/dataset-columns`, { params: { file_id: res.data.file_id } });
      setSchema(schemaRes.data);
      setStep(2);
    } catch (err) {
      setError(err.response?.data?.detail || 'Upload failed');
    } finally {
      setLoading(false);
    }
  };

  const handleCleanData = async () => {
    setLoading(true);
    try {
      await axios.post(`${API_URL}/clean-data`, { file_id: fileId });
      setStep(3);
    } catch (err) {
      setError(err.response?.data?.detail || 'Cleaning failed');
    } finally {
      setLoading(false);
    }
  };

  const handleFeatureEngineering = async () => {
    if (!targetColumn) {
      setError('Please select a target column');
      return;
    }
    if (featureColumns.length === 0) {
      setError('Please select at least one feature column');
      return;
    }

    setLoading(true);
    try {
      await axios.post(`${API_URL}/feature-engineering`, {
        file_id: fileId,
        target_column: targetColumn,
        feature_columns: featureColumns
      });
      setStep(4);
    } catch (err) {
      setError(err.response?.data?.detail || 'Feature engineering failed');
    } finally {
      setLoading(false);
    }
  };

  const handleTrainModel = async () => {
    setLoading(true);
    setError(null);
    try {
      const res = await axios.post(`${API_URL}/train`, {
        file_id: fileId,
        model_type: modelType,
        model_name: modelName,
        target_column: targetColumn,
        feature_columns: featureColumns
      });

      if (res.data && res.data.metrics) {
        setMetrics(res.data.metrics);
        setPlotData(res.data.plot_data || []);
        setStep(5);
        window.scrollTo(0, 0);
      } else {
        throw new Error('No metrics returned from server');
      }
    } catch (err) {
      setError(err.response?.data?.detail || err.message || 'Training failed');
    } finally {
      setLoading(false);
    }
  };

  const downloadModel = () => {
    window.open(`${API_URL}/download-model?file_id=${fileId}&model_name=${modelName}&format=${downloadFormat}`, '_blank');
  };

  const toggleFeature = (colName) => {
    if (featureColumns.includes(colName)) {
      setFeatureColumns(prev => prev.filter(c => c !== colName));
    } else {
      setFeatureColumns(prev => [...prev, colName]);
    }
  };

  const handleStartOver = () => {
    setStep(1);
    setFileId(null);
    setSchema(null);
    setTargetColumn('');
    setFeatureColumns([]);
    setModelType('classification');
    setModelName('Logistic Regression');
    setMetrics(null);
    setPlotData([]);
    setError(null);
  };

  const renderStepIndicator = () => {
    return (
      <div className="step-indicator">
        {[1, 2, 3, 4, 5].map(num => {
          const isCompleted = step > num || (step === 5 && num === 5);
          const isActive = step === num && num !== 5;
          return (
            <div key={num} className={`step ${isActive ? 'active' : isCompleted ? 'completed' : ''}`}>
              {isCompleted ? <CheckCircle size={20} /> : num}
            </div>
          );
        })}
      </div>
    );
  };

  return (
    <div className="app-container">
      <header className="header">
        <h1>ML/DL Generator</h1>
        <p>Your beautiful, seamless machine learning platform.</p>
      </header>

      {renderStepIndicator()}

      <div className="main-content">
        {error && (
          <div style={{ backgroundColor: 'var(--error)', padding: '15px', borderRadius: '8px', marginBottom: '20px', color: 'white' }}>
            {error}
          </div>
        )}

        {/* STEP 1: UPLOAD */}
        {step === 1 && (
          <div className="card">
            <h2><Upload /> Upload Dataset</h2>
            <p style={{ marginBottom: '20px', color: 'var(--text-secondary)' }}>Upload your CSV or Excel file to begin building your model.</p>

            <label className="upload-area">
              <Upload className="upload-icon" size={48} />
              <h3>Drag & Drop or Click to Upload</h3>
              <p style={{ marginTop: '10px', color: 'var(--text-secondary)' }}>Supports .csv, .xls, .xlsx</p>
              <input type="file" accept=".csv,.xls,.xlsx" onChange={handleFileUpload} disabled={loading} />
            </label>

            {loading && <p style={{ textAlign: 'center', marginTop: '20px', color: 'var(--primary)' }}>Analyzing file...</p>}
          </div>
        )}

        {/* STEP 2: SCHEMA & CLEANING */}
        {step === 2 && schema && (
          <div className="card">
            <h2><Database /> Dataset Overview</h2>
            <div className="stat-badges">
              <div className="badge numeric">Rows: {schema.total_rows}</div>
              <div className="badge categorical">Columns: {schema.total_columns}</div>
            </div>

            <div style={{ overflowX: 'auto' }}>
              <table className="data-table">
                <thead>
                  <tr>
                    <th>Column Name</th>
                    <th>Type</th>
                    <th>Missing Values</th>
                    <th>Unique Values</th>
                  </tr>
                </thead>
                <tbody>
                  {schema.columns.map(col => (
                    <tr key={col.name}>
                      <td><strong>{col.name}</strong></td>
                      <td>
                        <span className={`badge ${col.type}`}>{col.type}</span>
                      </td>
                      <td style={{ color: col.missing_values > 0 ? 'var(--error)' : 'inherit' }}>
                        {col.missing_values}
                      </td>
                      <td>{col.unique_values}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>

            <div style={{ marginTop: '30px', display: 'flex', justifyContent: 'flex-end' }}>
              <button className="btn btn-success" onClick={handleCleanData} disabled={loading}>
                <Settings size={18} /> Auto-Clean Data <ChevronRight size={18} />
              </button>
            </div>
          </div>
        )}

        {/* STEP 3: FEATURE ENGINEERING */}
        {step === 3 && schema && (
          <div className="card">
            <h2><Settings /> Feature Selection & Engineering</h2>
            <p style={{ marginBottom: '20px', color: 'var(--text-secondary)' }}>
              Select your prediction target and features to use. Automatic encoding and scaling will be applied.
            </p>

            <div className="form-group">
              <label>Target Column (What you want to predict)</label>
              <select
                className="form-select"
                value={targetColumn}
                onChange={e => {
                  setTargetColumn(e.target.value);
                  setFeatureColumns(prev => prev.filter(c => c !== e.target.value));
                  
                  // Auto-suggest model type based on schema
                  const selectedCol = schema.columns.find(c => c.name === e.target.value);
                  if (selectedCol) {
                    if (selectedCol.type === 'numeric' && selectedCol.unique_values > 10) {
                      setModelType('regression');
                      setModelName('Linear Regression');
                    } else {
                      setModelType('classification');
                      setModelName('Logistic Regression');
                    }
                  }
                  setError(null);
                }}
              >
                <option value="">Select Target...</option>
                {schema.columns.map(col => (
                  <option key={col.name} value={col.name}>{col.name}</option>
                ))}
              </select>
            </div>

            <div className="form-group" style={{ marginTop: '20px' }}>
              <label>Feature Columns (Data used to predict)</label>
              <div className="checkbox-grid">
                {schema.columns.filter(c => c.name !== targetColumn).map(col => (
                  <div
                    key={col.name}
                    className={`checkbox-item ${featureColumns.includes(col.name) ? 'selected' : ''}`}
                    onClick={() => toggleFeature(col.name)}
                  >
                    <input
                      type="checkbox"
                      checked={featureColumns.includes(col.name)}
                      readOnly
                    />
                    <span>{col.name}</span>
                  </div>
                ))}
              </div>
            </div>

            <div style={{ marginTop: '30px', display: 'flex', justifyContent: 'flex-end' }}>
              <button className="btn btn-success" onClick={handleFeatureEngineering} disabled={loading || !targetColumn}>
                <Activity size={18} /> Process Features <ChevronRight size={18} />
              </button>
            </div>
          </div>
        )}

        {/* STEP 4: TRAINING */}
        {step === 4 && (
          <div className="card">
            <h2><Play /> Train Model</h2>

            <div className="form-group">
              <label>Model Type</label>
              <select
                className="form-select"
                value={modelType}
                onChange={e => {
                  setModelType(e.target.value);
                  setModelName(e.target.value === 'classification' ? 'Logistic Regression' : 'Linear Regression');
                }}
              >
                <option value="classification">Classification (Predict a category/class)</option>
                <option value="regression">Regression (Predict a continuous number)</option>
              </select>
            </div>

            <div className="form-group" style={{ marginTop: '20px' }}>
              <label>Algorithm</label>
              <select className="form-select" value={modelName} onChange={e => setModelName(e.target.value)}>
                {modelType === 'classification' ? (
                  <>
                    <option value="Logistic Regression">Logistic Regression</option>
                    <option value="Random Forest Classifier">Random Forest</option>
                  </>
                ) : (
                  <>
                    <option value="Linear Regression">Linear Regression</option>
                    <option value="Random Forest Regressor">Random Forest</option>
                  </>
                )}
              </select>
            </div>

            <div style={{ marginTop: '30px', display: 'flex', justifyContent: 'flex-end' }}>
              <button className="btn btn-success" onClick={handleTrainModel} disabled={loading}>
                <Play size={18} /> Start Training <ChevronRight size={18} />
              </button>
            </div>
          </div>
        )}

        {/* STEP 5: EVALUATION */}
        {step === 5 && metrics && (
          <div className="card">
            <h2><Activity /> Model Evaluation</h2>
            <p style={{ color: 'var(--text-secondary)' }}>Model '{modelName}' trained successfully.</p>

            <div className="metrics-grid">
              {Object.entries(metrics).map(([key, value]) => (
                <div className="metric-card" key={key}>
                  <div className="metric-value">
                    {typeof value === 'number' ? (value > 1 && key !== 'rmse' && key !== 'mae' ? value : value.toFixed(4)) : value}
                  </div>
                  <div className="metric-label">{key}</div>
                </div>
              ))}
            </div>

            <div className="chart-container">
              <h3 className="chart-title"><Activity size={18} /> Actual vs Predicted (Sample)</h3>
              <ResponsiveContainer width="100%" height="80%">
                {modelType === 'regression' ? (
                  <LineChart data={plotData}>
                    <CartesianGrid strokeDasharray="3 3" stroke="#334155" />
                    <XAxis dataKey="index" stroke="#94a3b8" />
                    <YAxis stroke="#94a3b8" />
                    <Tooltip
                      contentStyle={{ backgroundColor: '#1e293b', border: '1px solid #334155', borderRadius: '8px' }}
                      itemStyle={{ color: '#f8fafc' }}
                    />
                    <Legend />
                    <Line type="monotone" dataKey="actual" stroke="#818cf8" strokeWidth={2} dot={{ r: 4 }} activeDot={{ r: 6 }} />
                    <Line type="monotone" dataKey="predicted" stroke="#10b981" strokeWidth={2} dot={{ r: 4 }} activeDot={{ r: 6 }} />
                  </LineChart>
                ) : (
                  <BarChart data={plotData}>
                    <CartesianGrid strokeDasharray="3 3" stroke="#334155" />
                    <XAxis dataKey="index" stroke="#94a3b8" />
                    <YAxis stroke="#94a3b8" />
                    <Tooltip
                      contentStyle={{ backgroundColor: '#1e293b', border: '1px solid #334155', borderRadius: '8px' }}
                    />
                    <Bar dataKey="actual" fill="#818cf8" radius={[4, 4, 0, 0]} />
                    <Bar dataKey="predicted" fill="#10b981" radius={[4, 4, 0, 0]} />
                  </BarChart>
                )}
              </ResponsiveContainer>
            </div>

            <div style={{ marginTop: '40px', display: 'flex', justifyContent: 'center', alignItems: 'center', gap: '15px', flexWrap: 'wrap' }}>
            <button className="btn btn-secondary" onClick={handleStartOver}>
              Start Over
            </button>
            
            <div style={{ display: 'flex', alignItems: 'center', gap: '10px', background: 'rgba(141, 110, 99, 0.08)', padding: '5px 15px', borderRadius: '8px', border: '1px solid var(--border-color)' }}>
              <span style={{ fontSize: '0.9rem', color: 'var(--text-secondary)' }}>Format:</span>
              <select 
                className="form-select" 
                style={{ width: 'auto', padding: '5px 10px', fontSize: '0.9rem', border: 'none', background: 'transparent' }}
                value={downloadFormat}
                onChange={(e) => setDownloadFormat(e.target.value)}
              >
                <option value="joblib">Joblib (.joblib)</option>
                <option value="onnx">ONNX (.onnx)</option>
                <option value="safetensors">Safetensors (.safotensors)</option>
              </select>
            </div>

            <button className="btn btn-success" onClick={downloadModel}>
              <Download size={18} /> Download Model
            </button>
          </div>
          </div>
        )}
      </div>

    </div>
  );
}

export default App;
