import logging
import json
from datetime import datetime, timezone
import os
import joblib
import pandas as pd
# Prediction monitoring
logging.basicConfig(level=logging.INFO)
prediction_logger = logging.getLogger("BotSentinel.Predictions")

from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field, ConfigDict


# ============================================================
# CONFIGURATION
# ============================================================

MODEL_PATH = "model/model.joblib"

NUMERIC_FEATURES = [
    "sport", "dport", "pkts", "bytes", "dur",
    "mean", "stddev", "sum", "min", "max",
    "spkts", "dpkts", "sbytes", "dbytes",
    "rate", "srate", "drate"
]

CATEGORICAL_FEATURES = [
    "flgs", "proto", "state"
]

ALL_FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES

model = None


# ============================================================
# APPLICATION LIFESPAN
# ============================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    global model

    if os.path.exists(MODEL_PATH):
        try:
            model = joblib.load(MODEL_PATH)
            print(f"BotSentinel model loaded: {MODEL_PATH}")
        except Exception as error:
            print(f"Model loading failed: {error}")
            model = None
    else:
        print(f"Model file not found: {MODEL_PATH}")

    yield

    model = None


app = FastAPI(
    title="BotSentinel API",
    description=(
        "Explainable AI-Based Real-Time Botnet "
        "Detection for IoT Networks"
    ),
    version="1.0.0",
    lifespan=lifespan
)


# ============================================================
# INPUT VALIDATION
# ============================================================

class TrafficInput(BaseModel):

    model_config = ConfigDict(extra="forbid")

    # Network ports
    sport: int = Field(ge=0, le=65535)
    dport: int = Field(ge=0, le=65535)

    # Packet and byte statistics
    pkts: float = Field(ge=0)
    bytes: float = Field(ge=0)
    dur: float = Field(ge=0)

    # Statistical traffic features
    mean: float = Field(ge=0)
    stddev: float = Field(ge=0)
    sum: float = Field(ge=0)
    min: float = Field(ge=0)
    max: float = Field(ge=0)

    # Source and destination statistics
    spkts: float = Field(ge=0)
    dpkts: float = Field(ge=0)
    sbytes: float = Field(ge=0)
    dbytes: float = Field(ge=0)

    # Traffic rate
    rate: float = Field(ge=0)
    srate: float = Field(ge=0)
    drate: float = Field(ge=0)

    # Protocol information
    flgs: str = Field(min_length=1, max_length=30)
    proto: str = Field(min_length=1, max_length=30)
    state: str = Field(min_length=1, max_length=30)


# ============================================================
# HEALTH CHECK
# ============================================================

@app.get("/health", tags=["Monitoring"])
def health():

    return {
        "application": "BotSentinel",
        "status": "healthy" if model is not None else "degraded",
        "model_loaded": model is not None
    }


# ============================================================
# PREDICTION ENGINE
# ============================================================

@app.post("/predict", tags=["Detection"])
def predict(data: TrafficInput):

    if model is None:
        raise HTTPException(
            status_code=503,
            detail="Trained model is not available."
        )

    try:

        input_data = pd.DataFrame(
            [data.model_dump()],
            columns=ALL_FEATURES
        )

        prediction = int(model.predict(input_data)[0])

        probabilities = model.predict_proba(input_data)[0]

        classes = list(model.classes_)
        class_index = classes.index(prediction)

        confidence = float(probabilities[class_index])

        label = (
            "Attack"
            if prediction == 1
            else "Normal"
        )

        if prediction == 1:

            recommendations = [
                "Investigate the source and destination of this traffic.",
                "Check packet rate and connection frequency.",
                "Review firewall and network monitoring logs.",
                "Consider isolating suspicious devices if an attack is confirmed."
            ]

        else:

            recommendations = [
                "No attack detected by the model.",
                "Continue routine network monitoring.",
                "Maintain regular security checks."
            ]

        # Structured prediction log for monitoring and future drift analysis.
        # Avoid logging client IP addresses or other identifying metadata.
        prediction_logger.info(
            json.dumps({
                "event": "prediction",
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "features": data.model_dump(),
                "prediction": prediction,
                "label": label,
                "confidence": round(confidence, 4)
            }, ensure_ascii=False)
        )

        return {
            "application": "BotSentinel",
            "prediction": prediction,
            "label": label,
            "confidence": round(confidence, 4),
            "confidence_percentage": round(confidence * 100, 2),
            "recommendations": recommendations
        }

    except HTTPException:
        raise

    except Exception as error:

        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {str(error)}"
        )


# ============================================================
# DASHBOARD
# ============================================================

@app.get("/", response_class=HTMLResponse, tags=["Dashboard"])
def home():

    return r"""
<!DOCTYPE html>
<html lang="en">

<head>

<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">

<title>BotSentinel | AI Network Security</title>

<style>

:root {
    --bg: #080d18;
    --panel: #101a2b;
    --panel2: #142137;
    --border: #263750;
    --text: #e5edf8;
    --muted: #8da1bc;
    --blue: #38bdf8;
    --green: #4ade80;
    --red: #fb7185;
}

* {
    box-sizing: border-box;
    margin: 0;
    padding: 0;
}

body {
    background: var(--bg);
    color: var(--text);
    font-family: Arial, Helvetica, sans-serif;
    min-height: 100vh;
}

header {
    height: 75px;
    padding: 0 6%;
    display: flex;
    justify-content: space-between;
    align-items: center;
    border-bottom: 1px solid var(--border);
    background: #0b1423;
}

.logo {
    font-size: 23px;
    font-weight: 800;
    color: var(--blue);
    letter-spacing: 0.5px;
}

.logo span {
    color: white;
}

.header-right {
    display: flex;
    gap: 20px;
    align-items: center;
}

.api-status {
    font-size: 12px;
    color: var(--green);
    background: #123323;
    padding: 9px 13px;
    border-radius: 30px;
}

main {
    max-width: 1200px;
    margin: 40px auto;
    padding: 0 22px;
}

.hero {
    margin-bottom: 30px;
}

.eyebrow {
    color: var(--blue);
    text-transform: uppercase;
    letter-spacing: 2px;
    font-size: 11px;
    font-weight: bold;
    margin-bottom: 12px;
}

h1 {
    font-size: clamp(27px, 4vw, 38px);
    margin-bottom: 12px;
}

.subtitle {
    color: var(--muted);
    line-height: 1.7;
    font-size: 14px;
}

.stats {
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 17px;
    margin-bottom: 25px;
}

.stat-card {
    background: var(--panel);
    border: 1px solid var(--border);
    border-radius: 11px;
    padding: 20px;
}

.stat-label {
    color: var(--muted);
    font-size: 12px;
    margin-bottom: 12px;
}

.stat-value {
    font-size: 21px;
    font-weight: bold;
}

.blue {
    color: var(--blue);
}

.green {
    color: var(--green);
}

.panel {
    background: var(--panel);
    border: 1px solid var(--border);
    border-radius: 13px;
    padding: 25px;
    margin-bottom: 22px;
}

.panel-heading {
    display: flex;
    justify-content: space-between;
    align-items: center;
    gap: 12px;
    margin-bottom: 10px;
}

h2 {
    font-size: 19px;
}

.section-description {
    color: var(--muted);
    font-size: 13px;
    line-height: 1.6;
    margin-bottom: 22px;
}

.group-title {
    color: var(--blue);
    font-size: 12px;
    text-transform: uppercase;
    letter-spacing: 1px;
    margin: 24px 0 14px;
}

.fields {
    display: grid;
    grid-template-columns: repeat(4, minmax(0, 1fr));
    gap: 15px;
}

.field label {
    display: block;
    color: #a9bad0;
    font-size: 12px;
    margin-bottom: 7px;
}

.field input {
    width: 100%;
    background: #091323;
    border: 1px solid #34455e;
    border-radius: 7px;
    color: white;
    padding: 12px;
    outline: none;
    font-size: 13px;
}

.field input:focus {
    border-color: var(--blue);
    box-shadow: 0 0 0 2px #38bdf820;
}

.actions {
    display: flex;
    flex-wrap: wrap;
    gap: 12px;
    margin-top: 25px;
}

button {
    border: none;
    border-radius: 7px;
    padding: 13px 22px;
    font-size: 13px;
    font-weight: bold;
    cursor: pointer;
    transition: 0.2s;
}

.primary {
    background: #0284c7;
    color: white;
}

.primary:hover {
    background: #0369a1;
}

.secondary {
    background: #25354d;
    color: #dbeafe;
}

.secondary:hover {
    background: #334b6a;
}

button:disabled {
    opacity: 0.6;
    cursor: wait;
}

.result {
    display: none;
}

.result-top {
    display: flex;
    justify-content: space-between;
    align-items: center;
    flex-wrap: wrap;
    gap: 20px;
}

.result-label {
    font-size: 27px;
    font-weight: bold;
    margin-top: 8px;
}

.attack {
    color: var(--red);
}

.normal {
    color: var(--green);
}

.confidence {
    text-align: right;
}

.confidence strong {
    display: block;
    font-size: 25px;
    margin-top: 8px;
    color: var(--blue);
}

.progress {
    height: 9px;
    background: #263750;
    border-radius: 20px;
    overflow: hidden;
    margin-top: 20px;
}

.progress-bar {
    height: 100%;
    width: 0%;
    background: var(--blue);
    transition: width 0.5s;
}

.recommendations {
    margin: 20px 0 0 20px;
    color: #bdcce0;
    line-height: 2;
    font-size: 13px;
}

.error {
    color: var(--red);
    font-size: 13px;
    margin-top: 15px;
}

footer {
    border-top: 1px solid var(--border);
    padding: 27px 15px;
    text-align: center;
    color: #72849e;
    font-size: 12px;
    line-height: 1.8;
}

footer a {
    color: var(--blue);
    text-decoration: none;
}

@media(max-width: 850px) {
    .fields {
        grid-template-columns: repeat(2, 1fr);
    }
}

@media(max-width: 600px) {
    header {
        padding: 0 18px;
    }

    .stats {
        grid-template-columns: 1fr;
    }

    .fields {
        grid-template-columns: 1fr;
    }

    .panel {
        padding: 18px;
    }

    .confidence {
        text-align: left;
    }
}

</style>

</head>

<body>

<header>

    <div class="logo">◈ Bot<span>Sentinel</span></div>

    <div class="header-right">
        <div class="api-status" id="apiStatus">
            ● Checking API...
        </div>
    </div>

</header>

<main>

    <section class="hero">

        <div class="eyebrow">
            Artificial Intelligence · IoT Security
        </div>

        <h1>Network Traffic Intelligence</h1>

        <p class="subtitle">
            Analyze network traffic using a machine learning model
            to identify potentially malicious activity.
        </p>

    </section>


    <section class="stats">

        <div class="stat-card">
            <div class="stat-label">Detection Engine</div>
            <div class="stat-value blue">Random Forest</div>
        </div>

        <div class="stat-card">
            <div class="stat-label">Input Features</div>
            <div class="stat-value">20 Features</div>
        </div>

        <div class="stat-card">
            <div class="stat-label">Detection Classes</div>
            <div class="stat-value green">Normal / Attack</div>
        </div>

    </section>


    <section class="panel">

        <div class="panel-heading">
            <h2>Traffic Analysis</h2>
        </div>

        <p class="section-description">
            Enter the network traffic measurements below.
            The trained model will classify the supplied traffic.
        </p>

        <form id="trafficForm">

            <div class="group-title">
                Network Ports and Protocol
            </div>

            <div class="fields" id="networkFields"></div>

            <div class="group-title">
                Packet and Byte Statistics
            </div>

            <div class="fields" id="packetFields"></div>

            <div class="group-title">
                Traffic Statistical Features
            </div>

            <div class="fields" id="statFields"></div>

            <div class="actions">

                <button
                    type="submit"
                    class="primary"
                    id="analyzeButton">
                    Analyze Traffic →
                </button>

                <button
                    type="button"
                    class="secondary"
                    id="resetButton">
                    Reset Values
                </button>

            </div>

        </form>

        <div class="error" id="errorMessage"></div>

    </section>


    <section class="panel result" id="resultPanel">

        <div class="panel-heading">
            <h2>Detection Report</h2>
        </div>

        <div class="result-top">

            <div>
                <div class="stat-label">Classification Result</div>
                <div class="result-label" id="predictionLabel">
                    --
                </div>
            </div>

            <div class="confidence">
                <div class="stat-label">Model Confidence</div>
                <strong id="confidenceValue">0%</strong>
            </div>

        </div>

        <div class="progress">
            <div class="progress-bar" id="confidenceBar"></div>
        </div>

        <div class="group-title">
            Security Recommendations
        </div>

        <ul class="recommendations" id="recommendationList"></ul>

        <p class="section-description" style="margin-top:20px">
            This result is a model prediction, not a guarantee
            that the traffic is safe or malicious. Verify suspicious
            activity using additional network evidence.
        </p>

    </section>

</main>


<footer>

    <strong>BotSentinel</strong> · AI-Based Botnet Detection
    and Attack Classification for IoT Networks.

    <br>

    Research and academic project · API Documentation:
    <a href="/docs">Swagger UI</a>

</footer>


<script>

// ==========================================================
// INPUT DEFINITIONS
// ==========================================================

const defaults = {

    sport: 51514,
    dport: 80,

    pkts: 10,
    bytes: 1200,
    dur: 2.5,

    mean: 120,
    stddev: 20,
    sum: 1200,
    min: 80,
    max: 200,

    spkts: 6,
    dpkts: 4,
    sbytes: 800,
    dbytes: 400,

    rate: 4,
    srate: 2.4,
    drate: 1.6,

    flgs: "e",
    proto: "tcp",
    state: "CON"

};

const groups = {

    networkFields: [
        "sport", "dport", "flgs", "proto", "state"
    ],

    packetFields: [
        "pkts", "bytes", "spkts", "dpkts",
        "sbytes", "dbytes"
    ],

    statFields: [
        "dur", "mean", "stddev", "sum",
        "min", "max", "rate", "srate", "drate"
    ]

};


// ==========================================================
// BUILD INPUT FORM
// ==========================================================

function createFields() {

    Object.entries(groups).forEach(([containerId, names]) => {

        const container = document.getElementById(containerId);

        names.forEach(name => {

            const wrapper = document.createElement("div");
            wrapper.className = "field";

            const label = document.createElement("label");
            label.textContent = name.toUpperCase();

            const input = document.createElement("input");

            input.name = name;
            input.value = defaults[name];
            input.required = true;

            if (typeof defaults[name] === "number") {

                input.type = "number";
                input.step = "any";
                input.min = "0";

                if (name === "sport" || name === "dport") {
                    input.step = "1";
                    input.max = "65535";
                }

            } else {

                input.type = "text";
                input.maxLength = 30;

            }

            label.appendChild(input);
            wrapper.appendChild(label);
            container.appendChild(wrapper);

        });

    });

}

createFields();


// ==========================================================
// API HEALTH STATUS
// ==========================================================

async function checkAPI() {

    const status = document.getElementById("apiStatus");

    try {

        const response = await fetch("/health");
        const data = await response.json();

        if (response.ok && data.model_loaded) {

            status.textContent = "● API Online";
            status.style.color = "#4ade80";

        } else {

            status.textContent = "● Model Unavailable";
            status.style.color = "#fb7185";

        }

    } catch (error) {

        status.textContent = "● API Offline";
        status.style.color = "#fb7185";

    }

}

checkAPI();


// ==========================================================
// RESET FORM
// ==========================================================

document.getElementById("resetButton").addEventListener(
    "click", () => {

        document.getElementById("trafficForm").reset();

        Object.entries(defaults).forEach(([name, value]) => {

            document.querySelector(
                `[name="${name}"]`
            ).value = value;

        });

        document.getElementById("resultPanel").style.display = "none";
        document.getElementById("errorMessage").textContent = "";

    }
);


// ==========================================================
// SUBMIT TRAFFIC FOR PREDICTION
// ==========================================================

document.getElementById("trafficForm").addEventListener(
    "submit", async (event) => {

        event.preventDefault();

        const button = document.getElementById("analyzeButton");
        const errorBox = document.getElementById("errorMessage");
        const resultPanel = document.getElementById("resultPanel");

        button.disabled = true;
        button.textContent = "Analyzing Traffic...";

        errorBox.textContent = "";
        resultPanel.style.display = "none";

        try {

            const data = {};

            Object.entries(defaults).forEach(([name, defaultValue]) => {

                const value = document.querySelector(
                    `[name="${name}"]`
                ).value;

                data[name] = typeof defaultValue === "number"
                    ? Number(value)
                    : value.trim();

            });

            const response = await fetch("/predict", {

                method: "POST",

                headers: {
                    "Content-Type": "application/json"
                },

                body: JSON.stringify(data)

            });

            const output = await response.json();

            if (!response.ok) {

                throw new Error(
                    typeof output.detail === "string"
                        ? output.detail
                        : JSON.stringify(output.detail)
                );

            }

            // Display classification
            const predictionLabel = document.getElementById(
                "predictionLabel"
            );

            predictionLabel.textContent = output.label;

            predictionLabel.className =
                "result-label " +
                (output.prediction === 1 ? "attack" : "normal");

            // Display confidence
            const confidence = output.confidence_percentage;

            document.getElementById(
                "confidenceValue"
            ).textContent = confidence.toFixed(2) + "%";

            document.getElementById(
                "confidenceBar"
            ).style.width = confidence + "%";

            // Display recommendations safely
            const list = document.getElementById(
                "recommendationList"
            );

            list.replaceChildren();

            output.recommendations.forEach(tip => {

                const item = document.createElement("li");

                item.textContent = tip;

                list.appendChild(item);

            });

            resultPanel.style.display = "block";

            resultPanel.scrollIntoView({
                behavior: "smooth",
                block: "start"
            });

        } catch (error) {

            errorBox.textContent =
                "Prediction failed: " + error.message;

        } finally {

            button.disabled = false;
            button.textContent = "Analyze Traffic →";

        }

    }
);

</script>

</body>
</html>
"""


# ============================================================
# RUN DIRECTLY
# ============================================================

if __name__ == "__main__":

    import uvicorn

    uvicorn.run(
        "app:app",
        host="127.0.0.1",
        port=8000,
        reload=True
    )
