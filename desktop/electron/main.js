/**
 * Nexus Shadow-Quant — Electron Main Process
 * ============================================
 * Manages: window lifecycle, Python backend spawn, splash screen, system tray
 * Supports both development mode and installed app (embedded Python)
 */

const { app, BrowserWindow, Tray, Menu, nativeImage, ipcMain } = require('electron');
const path = require('path');
const fs = require('fs');
const { spawn } = require('child_process');

let mainWindow = null;
let splashWindow = null;
let tray = null;
let pythonProcess = null;

const API_PORT = 8420;
const isDev = process.env.NODE_ENV === 'development' || !app.isPackaged;

// ─── Path Resolution ─────────────────────────────────

function getPythonPath() {
    if (isDev) return 'python';
    return path.join(process.resourcesPath, 'python_embedded', 'python.exe');
}

function getScriptPath(script) {
    if (isDev) return path.join(__dirname, '..', '..', script);
    return path.join(process.resourcesPath, 'python_backend', script);
}

function getBackendCwd() {
    if (isDev) return path.join(__dirname, '..', '..');
    return path.join(process.resourcesPath, 'python_backend');
}

function getUserDataPath() {
    // In production, data is stored in AppData
    if (isDev) return path.join(__dirname, '..', '..');
    return path.join(app.getPath('userData'));
}

// ─── Python Backend ──────────────────────────────────

function startPython() {
    const pythonPath = getPythonPath();
    const scriptPath = getScriptPath('api_server.py');
    const cwd = getBackendCwd();

    console.log(`[Electron] Starting Python: ${pythonPath} ${scriptPath}`);
    console.log(`[Electron] CWD: ${cwd}`);

    pythonProcess = spawn(pythonPath, [scriptPath], {
        env: {
            ...process.env,
            NEXUS_API_PORT: String(API_PORT),
            PYTHONPATH: cwd,
            NEXUS_USER_DATA: getUserDataPath(),
        },
        stdio: ['pipe', 'pipe', 'pipe'],
        cwd: cwd,
    });

    pythonProcess.stdout.on('data', (data) => {
        console.log(`[Python] ${data.toString().trim()}`);
    });

    pythonProcess.stderr.on('data', (data) => {
        console.error(`[Python] ${data.toString().trim()}`);
    });

    pythonProcess.on('error', (err) => {
        console.error('Failed to start Python backend:', err);
    });

    pythonProcess.on('exit', (code) => {
        console.log(`Python backend exited with code ${code}`);
    });
}

function stopPython() {
    if (!pythonProcess) return;

    // 1. Ask the backend to shut down gracefully
    const http = require('http');
    const req = http.request(
        { hostname: '127.0.0.1', port: API_PORT, path: '/api/shutdown', method: 'POST', timeout: 2000 },
        () => { }
    );
    req.on('error', () => { }); // Ignore — process may already be gone
    req.end();

    // 2. Wait up to 3s for clean exit, then force kill
    const proc = pythonProcess;
    pythonProcess = null;

    const forceKillTimer = setTimeout(() => {
        try {
            if (process.platform === 'win32') {
                // SIGTERM doesn't work on Windows — use taskkill for reliable termination
                require('child_process').execSync(`taskkill /PID ${proc.pid} /T /F`, { stdio: 'ignore' });
            } else {
                proc.kill('SIGTERM');
            }
        } catch { /* already exited */ }
    }, 3000);

    proc.on('exit', () => clearTimeout(forceKillTimer));
}

// ─── Wait for API ────────────────────────────────────

function waitForApi(retries = 60) {
    return new Promise((resolve, reject) => {
        let attempts = 0;
        const check = () => {
            const http = require('http');
            const req = http.get(`http://127.0.0.1:${API_PORT}/api/boot-status`, (res) => {
                let body = '';
                res.on('data', (chunk) => body += chunk);
                res.on('end', () => {
                    try {
                        const data = JSON.parse(body);
                        // Forward status to splash window
                        if (splashWindow && !splashWindow.isDestroyed()) {
                            splashWindow.webContents.send('boot-progress', data);
                        }
                        if (data.stage === 'ready' || data.stage === 'error') {
                            resolve(data);
                        } else {
                            attempts++;
                            if (attempts < retries) setTimeout(check, 1000);
                            else resolve(data);
                        }
                    } catch {
                        attempts++;
                        if (attempts < retries) setTimeout(check, 1000);
                        else reject(new Error('Timeout waiting for API'));
                    }
                });
            });
            req.on('error', () => {
                attempts++;
                if (attempts < retries) setTimeout(check, 1000);
                else reject(new Error('API failed to start'));
            });
            req.end();
        };
        check();
    });
}

// ─── Splash Screen ───────────────────────────────────

function createSplash() {
    splashWindow = new BrowserWindow({
        width: 520,
        height: 380,
        frame: false,
        transparent: true,
        resizable: false,
        center: true,
        alwaysOnTop: true,
        skipTaskbar: true,
        webPreferences: {
            preload: path.join(__dirname, 'splashPreload.js'),
            contextIsolation: true,
            nodeIntegration: false,
        },
    });

    splashWindow.loadFile(path.join(__dirname, 'splash.html'));
}

// ─── Main Window ─────────────────────────────────────

function createMainWindow() {
    mainWindow = new BrowserWindow({
        width: 1440,
        height: 900,
        minWidth: 1100,
        minHeight: 700,
        show: false,
        frame: false,
        titleBarStyle: 'hidden',
        backgroundColor: '#080B12',
        webPreferences: {
            preload: path.join(__dirname, 'preload.js'),
            contextIsolation: true,
            nodeIntegration: false,
        },
    });

    // Load React app
    if (isDev) {
        mainWindow.loadURL('http://localhost:5173');
    } else {
        mainWindow.loadFile(path.join(__dirname, '..', 'dist', 'index.html'));
    }

    mainWindow.once('ready-to-show', () => {
        if (splashWindow && !splashWindow.isDestroyed()) {
            splashWindow.close();
        }
        mainWindow.show();
        mainWindow.focus();
    });

    mainWindow.on('closed', () => {
        mainWindow = null;
    });

    // Window controls via IPC
    ipcMain.on('window-minimize', () => mainWindow?.minimize());
    ipcMain.on('window-maximize', () => {
        if (mainWindow?.isMaximized()) mainWindow.unmaximize();
        else mainWindow?.maximize();
    });
    ipcMain.on('window-close', () => mainWindow?.close());
}

// ─── System Tray ─────────────────────────────────────

function createTray() {
    const iconPath = path.join(__dirname, '..', 'public', 'icon.png');
    try {
        const icon = nativeImage.createFromPath(iconPath).resize({ width: 20, height: 20 });
        tray = new Tray(icon);
    } catch {
        // Fallback: create a simple icon
        tray = new Tray(nativeImage.createEmpty());
    }

    const contextMenu = Menu.buildFromTemplate([
        { label: 'Show Nexus', click: () => mainWindow?.show() },
        { type: 'separator' },
        { label: 'Quit', click: () => app.quit() },
    ]);

    tray.setToolTip('Nexus Shadow-Quant');
    tray.setContextMenu(contextMenu);

    tray.on('click', () => {
        if (mainWindow) {
            if (mainWindow.isVisible()) mainWindow.focus();
            else mainWindow.show();
        }
    });
}

// ─── App Lifecycle ───────────────────────────────────

app.whenReady().then(async () => {
    createSplash();
    createTray();
    startPython();

    // Wait for Python API to be ready
    try {
        const bootResult = await waitForApi();

        // Check if system requirements failed
        if (bootResult && bootResult.stage === 'error') {
            const { dialog } = require('electron');
            const msg = bootResult.message || 'Unknown system error';

            // Keep splash visible with error, show native dialog
            dialog.showMessageBoxSync({
                type: 'error',
                title: 'Nexus Shadow-Quant — System Requirements',
                message: 'System requirements not met',
                detail: msg + '\n\nThe application requires:\n• NVIDIA GPU (RTX 3060 or higher)\n• 20 GB free disk space\n• CUDA-compatible drivers',
                buttons: ['Exit'],
            });

            stopPython();
            app.quit();
            return;
        }
    } catch (e) {
        console.error('API wait failed:', e);
    }

    createMainWindow();
});

app.on('window-all-closed', () => {
    stopPython();
    app.quit();
});

app.on('before-quit', () => {
    stopPython();
});
