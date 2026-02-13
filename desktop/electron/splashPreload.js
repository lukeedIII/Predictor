/**
 * Splash Preload — Secure IPC Bridge
 * Exposes only the boot-progress listener to the splash renderer.
 * This replaces the old nodeIntegration:true pattern (C-1 fix).
 */
const { contextBridge, ipcRenderer } = require('electron');

contextBridge.exposeInMainWorld('electronAPI', {
    onBootProgress: (cb) => ipcRenderer.on('boot-progress', (_e, data) => cb(data)),
});
