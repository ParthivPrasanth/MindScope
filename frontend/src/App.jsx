import { useState } from "react";

const API_BASE = "http://127.0.0.1:8000";

const initialForm = {
  age: "",
  gender: "",
  cgpa: "",
  sleep_duration: "",
  academic_pressure: "3",
  study_satisfaction: "3",
  financial_stress: "3",
  work_study_hours: "",
  dietary_habits: "",
  family_history: "",
  suicidal_thoughts: "",
  free_text: "",
};

const getResultContent = (data) => {
  const { result, severity } = data;

  if (result === "not_depressed") {
    return {
      icon: "🌱",
      title: "Looking Good",
      color: "#4A90D9",
      text: "Based on your responses and what you've shared with us, our assessment doesn't detect significant signs of depression right now. That said, everyone's mental health shifts over time — it's worth checking in with yourself regularly. If things ever feel heavier than usual, reaching out to someone you trust is always a good step.",
    };
  }

  if (result === "uncertain") {
    return {
      icon: "🌤️",
      title: "Worth Paying Attention To",
      color: "#6B9FD4",
      text: "Your responses suggest you may be going through a challenging time. We don't have enough information to give a confident assessment — but that's okay. What matters is that you're reflecting on how you feel. Talking to a counsellor or a trusted person can make a real difference.",
    };
  }

  const severityMessages = {
    "Minimal / Normal": {
      icon: "🌤️",
      title: "Mild Indicators Detected",
      color: "#5B9BD5",
      text: "Your responses suggest you may be experiencing some mild signs of low mood or stress. This doesn't mean something is seriously wrong — but it's worth being kind to yourself and checking in on how you're doing. Small steps like rest, connection, and talking about how you feel can help.",
    },
    "Mild": {
      icon: "🌤️",
      title: "Mild Indicators Detected",
      color: "#5B9BD5",
      text: "Your responses suggest you may be experiencing some mild signs of low mood or stress. This doesn't mean something is seriously wrong — but it's worth being kind to yourself. Small steps like rest, connection, and talking about how you feel can make a difference.",
    },
    "Moderate": {
      icon: "💙",
      title: "Moderate Indicators Detected",
      color: "#4A7FBF",
      text: "Your responses suggest you may be experiencing moderate symptoms that are affecting your day-to-day life. You don't have to manage this alone. Speaking to a counsellor or mental health professional can provide real support and clarity.",
    },
    "Severe": {
      icon: "💙",
      title: "Significant Indicators Detected",
      color: "#3A6FA8",
      text: "Your responses suggest you may be going through a difficult period with significant emotional distress. Please know that what you're feeling is real and you deserve support. Reaching out to a professional can be a meaningful first step.",
    },
    "Extremely Severe": {
      icon: "💙",
      title: "Please Reach Out",
      color: "#2C5F8A",
      text: "Your responses suggest you may be experiencing a high level of distress. You don't have to face this alone. Please consider talking to someone — a counsellor, a helpline, or a trusted person in your life. Taking that step takes courage, and support is available.",
    },
  };

  return severityMessages[severity] || severityMessages["Moderate"];
};

const getSeverityPosition = (data) => {
  if (data.result === "not_depressed") return 0.08;
  if (data.result === "uncertain") return 0.42;
  const map = {
    "Minimal / Normal": 0.22,
    "Mild": 0.38,
    "Moderate": 0.56,
    "Severe": 0.75,
    "Extremely Severe": 0.93,
  };
  return map[data.severity] ?? 0.56;
};

function ArcGauge({ position }) {
  const probability = position;
  const r = 70;
  const cx = 100;
  const cy = 95;

  const bgPath = `M ${cx - r} ${cy} A ${r} ${r} 0 0 1 ${cx + r} ${cy}`;

  const theta = Math.PI - probability * Math.PI;
  const ex = cx + r * Math.cos(theta);
  const ey = cy - r * Math.sin(theta);
  const largeArc = probability > 0.5 ? 1 : 0;
  const activePath = `M ${cx - r} ${cy} A ${r} ${r} 0 ${largeArc} 1 ${ex} ${ey}`;

  const needleLen = r - 12;
  const nx = cx + needleLen * Math.cos(theta);
  const ny = cy - needleLen * Math.sin(theta);

  return (
    <svg viewBox="0 0 200 108" className="arc-gauge">
      <path d={bgPath} fill="none" stroke="#D6E8F7" strokeWidth="14" strokeLinecap="round" />
      <path d={activePath} fill="none" stroke="#4A90D9" strokeWidth="14" strokeLinecap="round" />
      <line x1={cx} y1={cy} x2={nx} y2={ny} stroke="#2C5F8A" strokeWidth="3" strokeLinecap="round" />
      <circle cx={cx} cy={cy} r="5" fill="#2C5F8A" />
      <text x="28" y="108" fontSize="9" fill="#9aafcc" textAnchor="middle">Minimal</text>
      <text x="172" y="108" fontSize="9" fill="#9aafcc" textAnchor="middle">Significant</text>
    </svg>
  );
}

function ProgressBar({ step }) {
  const steps = ["Personal", "Lifestyle", "Mental Health", "Feelings", "Result"];
  return (
    <div className="progress-bar">
      {steps.map((s, i) => (
        <div key={s} className={`progress-step${i < step ? " done" : ""}${i === step ? " active" : ""}`}>
          <div className="progress-dot">{i < step ? "✓" : i + 1}</div>
          <span className="progress-label">{s}</span>
          {i < steps.length - 1 && (
            <div className={`progress-line${i < step ? " done" : ""}`} />
          )}
        </div>
      ))}
    </div>
  );
}

function SliderField({ label, name, value, onSlide }) {
  const numValue = Number(value) || 3;
  const fillPct = ((numValue - 1) / 4) * 100;

  return (
    <div className="field-row">
      <label className="form-label">{label}</label>
      <div className="slider-wrapper">
        <input
          type="range"
          name={name}
          min="1"
          max="5"
          value={numValue}
          onChange={(e) => onSlide(name, e.target.value)}
          className="slider-input"
          style={{ "--fill": `${fillPct}%` }}
        />
        <div className="slider-labels">
          <span className="slider-edge-label">Low</span>
          <span className="slider-value-badge">{numValue}</span>
          <span className="slider-edge-label">High</span>
        </div>
      </div>
    </div>
  );
}

function FormInput({ label, name, type = "text", value, onChange, options = null, required = true }) {
  return (
    <div className="field-row">
      <label htmlFor={name} className="form-label">{label}</label>
      {options ? (
        <select id={name} name={name} value={value} onChange={onChange} required={required}>
          <option value=""></option>
          {options.map((opt) => (
            <option key={opt} value={opt}>{opt}</option>
          ))}
        </select>
      ) : (
        <input
          id={name}
          name={name}
          type={type}
          step={type === "number" ? "0.1" : undefined}
          min={type === "number" ? "0" : undefined}
          value={value}
          onChange={onChange}
          required={required}
        />
      )}
    </div>
  );
}

function ResultCard({ data, onReset }) {
  const content = getResultContent(data);
  const confidencePct = Math.round(data.depression_probability * 100);
  const showICall =
    data.result === "depressed" &&
    (data.severity === "Severe" || data.severity === "Extremely Severe");

  return (
    <div className="result-card">
      <div className="result-icon">{content.icon}</div>
      <h2 className="result-title" style={{ color: content.color }}>{content.title}</h2>

      <ArcGauge position={getSeverityPosition(data)} />

      <p className="result-text">{content.text}</p>

      {data.severity && (
        <div className="severity-badge">
          Assessment level: <strong>{data.severity}</strong>
        </div>
      )}

      <div className="confidence-row">
        Our confidence in this assessment: <strong>{confidencePct}%</strong>
      </div>

      {showICall && (
        <div className="icall-card">
          <div className="icall-icon">💬</div>
          <div>
            <div className="icall-title">If you'd like to talk to someone</div>
            <div className="icall-text">iCall offers free, confidential support.</div>
            <div className="icall-number">9152987821</div>
          </div>
        </div>
      )}

      <p className="result-disclaimer">{data.disclaimer}</p>

      <button className="reset-btn" onClick={onReset}>Take this again</button>
    </div>
  );
}

function App() {
  const [form, setForm] = useState(initialForm);
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null);
  const [error, setError] = useState("");

  const currentStep = result
    ? 4
    : form.free_text.length > 0
    ? 3
    : form.family_history
    ? 2
    : form.sleep_duration && form.work_study_hours && form.dietary_habits
    ? 1
    : 0;

  const onChange = (e) => {
    const { name, value } = e.target;
    setForm((prev) => ({ ...prev, [name]: value }));
  };

  const onSlide = (name, value) => {
    setForm((prev) => ({ ...prev, [name]: String(value) }));
  };

  const onReset = () => {
    setForm(initialForm);
    setResult(null);
    setError("");
  };

  const onSubmit = async (e) => {
    e.preventDefault();
    setLoading(true);
    setError("");
    setResult(null);

    try {
      const suicidalValue =
        !form.suicidal_thoughts || form.suicidal_thoughts === "Prefer not to answer"
          ? "No"
          : form.suicidal_thoughts;

      const payload = {
        survey_dict: {
          age: Number(form.age),
          gender: form.gender,
          cgpa: Number(form.cgpa),
          sleep_duration: form.sleep_duration,
          academic_pressure: Number(form.academic_pressure),
          study_satisfaction: Number(form.study_satisfaction),
          financial_stress: Number(form.financial_stress),
          work_study_hours: Number(form.work_study_hours),
          dietary_habits: form.dietary_habits,
          family_history: form.family_history,
          suicidal_thoughts: suicidalValue,
        },
        free_text: form.free_text,
      };

      const res = await fetch(`${API_BASE}/predict`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });

      if (!res.ok) {
        const message = await res.text();
        throw new Error(message || "Prediction request failed");
      }

      const data = await res.json();
      setResult(data);
    } catch (err) {
      setError(err.message || "Something went wrong");
    } finally {
      setLoading(false);
    }
  };

  return (
    <main>
      <section className="hero">
        <h1>🧠 MindScope Assessment</h1>
        <p>Comprehensive mental health prediction using AI-powered multi-modal analysis</p>
      </section>

      <ProgressBar step={currentStep} />

      {result ? (
        <ResultCard data={result} onReset={onReset} />
      ) : (
        <form className="form-container" onSubmit={onSubmit}>
          {/* Personal Info */}
          <div className="card-section">
            <h2 className="card-title">👤 Personal Information</h2>
            <div className="fields-grid cols-2">
              <FormInput label="Age" name="age" type="number" value={form.age} onChange={onChange} />
              <FormInput
                label="Gender"
                name="gender"
                value={form.gender}
                onChange={onChange}
                options={["Male", "Female", "Non-binary", "Prefer not to say"]}
              />
              <FormInput label="CGPA" name="cgpa" type="number" value={form.cgpa} onChange={onChange} />
            </div>
          </div>

          {/* Lifestyle */}
          <div className="card-section">
            <h2 className="card-title">🏃 Lifestyle & Academics</h2>
            <div className="fields-grid cols-2">
              <FormInput
                label="Sleep Duration"
                name="sleep_duration"
                value={form.sleep_duration}
                onChange={onChange}
                options={["Less than 5 hours", "5-6 hours", "6-7 hours", "7-8 hours", "More than 8 hours"]}
              />
              <FormInput
                label="Work/Study Hours"
                name="work_study_hours"
                type="number"
                value={form.work_study_hours}
                onChange={onChange}
              />
              <FormInput
                label="Dietary Habits"
                name="dietary_habits"
                value={form.dietary_habits}
                onChange={onChange}
                options={["Healthy", "Moderate", "Unhealthy"]}
              />
            </div>
          </div>

          {/* Mental Health */}
          <div className="card-section">
            <h2 className="card-title">🧠 Mental Health Indicators</h2>
            <div className="fields-grid">
              <SliderField
                label="Academic Pressure"
                name="academic_pressure"
                value={form.academic_pressure}
                onSlide={onSlide}
              />
              <SliderField
                label="Study Satisfaction"
                name="study_satisfaction"
                value={form.study_satisfaction}
                onSlide={onSlide}
              />
              <SliderField
                label="Financial Stress"
                name="financial_stress"
                value={form.financial_stress}
                onSlide={onSlide}
              />
              <div className="fields-grid cols-2" style={{ marginTop: "0.5rem" }}>
                <FormInput
                  label="Family History of Mental Illness"
                  name="family_history"
                  value={form.family_history}
                  onChange={onChange}
                  options={["No", "Yes", "I'm not sure"]}
                />
                <div className="field-row">
                  <label className="form-label">
                    Have you ever experienced thoughts of not wanting to be here?
                  </label>
                  <select
                    name="suicidal_thoughts"
                    value={form.suicidal_thoughts}
                    onChange={onChange}
                  >
                    <option value=""></option>
                    <option value="No">No</option>
                    <option value="Yes">Yes</option>
                    <option value="Prefer not to answer">Prefer not to answer</option>
                  </select>
                  <p className="optional-note">
                    This question is optional. Your answer does not affect whether you receive a result.
                  </p>
                </div>
              </div>
            </div>
          </div>

          {/* Free Text */}
          <div className="card-section">
            <label htmlFor="free_text" className="card-title">📝 How are you feeling?</label>
            <textarea
              id="free_text"
              name="free_text"
              value={form.free_text}
              onChange={onChange}
              required
              placeholder="Share your thoughts and feelings..."
            />
          </div>

          <button type="submit" disabled={loading}>
            {loading ? (
              <>
                <span className="spinner" />
                Analysing...
              </>
            ) : (
              "Get Prediction"
            )}
          </button>

          {error && (
            <div className="error">
              <strong>Error:</strong> {error}
            </div>
          )}
        </form>
      )}
    </main>
  );
}

export default App;
