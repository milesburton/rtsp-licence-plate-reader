import { parentPort } from 'worker_threads';
import { processFrame } from '../controllers/frameController.js';
import { createWorker } from 'tesseract.js';

const workerInstance = createWorker({
    logger: m => console.log(`[Tesseract]: ${m.status} (${Math.round(m.progress * 100)}%)`)
});

(async () => {
    await workerInstance.load();
    await workerInstance.loadLanguage('eng');
    await workerInstance.initialize('eng');
})();

parentPort.on('message', async (packet) => {
    try {
        const frameBuffer = Buffer.from(packet.frameBuffer);
        const metadata = packet.metadata;

        // Process the frame
        const plateDetections = await processFrame(frameBuffer, metadata, workerInstance);

        // Send detections back to the main thread
        parentPort.postMessage({ success: true, data: plateDetections });
    } catch (error) {
        parentPort.postMessage({ success: false, error: error.message });
    }
});