import { Worker } from 'worker_threads';
import path from 'path';
import logger from '../utils/logger';
import type { PlateDetection } from '../utils/types';

// Initialize Worker with ES Module
const worker = new Worker(path.resolve(__dirname, 'worker.js'), {
    type: 'module',
});

// Listen for messages from the worker
worker.on('message', (message) => {
    if (message.success) {
        const detections: PlateDetection[] = message.data;
        detections.forEach(detection => {
            logger.info(`🆔 License Plate Detected: ${detection.text} (${detection.confidence}%) in frame #${detection.frameNumber} at ${new Date(detection.timestamp).toISOString()}`);
            // Further actions like storing the detection or triggering alerts can be added here
        });
    } else {
        logger.error(`❌ Worker encountered an error: ${message.error}`);
    }
});

// Handle worker errors
worker.on('error', (error) => {
    logger.error(`❌ Worker error: ${error.message}`);
});

// Handle worker exit
worker.on('exit', (code) => {
    if (code !== 0) {
        logger.error(`❌ Worker stopped with exit code ${code}`);
    } else {
        logger.info('✅ Worker exited gracefully.');
    }
});

/**
 * Sends a frame packet to the worker for processing.
 *
 * @param packet - The frame packet to send.
 */
export function sendFrameToWorker(packet: { frameBuffer: ArrayBuffer; metadata: { timestamp: number; frameNumber: number } }) {
    try {
        worker.postMessage(packet, [packet.frameBuffer]); // Transfer the ArrayBuffer
    } catch (error) {
        logger.error(`❌ Failed to send packet to worker: ${error}`);
    }
}