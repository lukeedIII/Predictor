declare global {
    interface Window {
        electronAPI?: {
            minimize: () => void;
            maximize: () => void;
            close: () => void;
            onBootProgress: (callback: (data: { stage: string; progress: number; message: string }) => void) => void;
        };
    }
}

export { };
