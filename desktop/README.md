# Nexus Shadow-Quant — Frontend

The React + TypeScript frontend for the **Nexus Shadow-Quant** desktop application.  
Built with **Vite**, **Electron**, and **react-grid-layout** for a fully local, drag-and-drop trading dashboard.

## Stack

| Layer | Technology |
|:------|:-----------|
| Framework | React 19 + TypeScript |
| Bundler | Vite 6 |
| Shell | Electron 40 |
| Layout | react-grid-layout (drag-and-drop, resize, JSON presets) |
| Charts | TradingView Lightweight Charts |
| Styles | Vanilla CSS + CSS variables (light/dark theme) |
| Realtime | WebSocket (push) + REST (localhost:8420) |

## Development

```powershell
# From the project root
npm install
npm run dev         # Electron + Vite dev server (HMR)
npm run build       # Production bundle
```

## Key Components

```
src/
├── components/
│   ├── PriceCard/        # Live BTC price + signal badge
│   ├── QuantPanel/       # 16-model quant intelligence panel
│   ├── TradingView/      # Chart + AI trajectory overlay
│   ├── DrNexus/          # AI analyst chat panel
│   ├── PaperTrader/      # Position stats + equity curve
│   ├── WorldClock/       # 6 financial hub clocks
│   ├── SwissWeather/     # Live Zürich weather
│   └── HardwareMonitor/  # GPU/CPU utilization + VRAM
└── App.tsx               # Grid layout orchestrator
```

## Backend

The Python backend runs on `localhost:8420`. See [python_backend/](python_backend/) for setup.  
See the [root README](../README.md) for full project documentation.
