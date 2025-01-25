import { bufferToArrayBuffer } from './serialization';
import logger from './logger';
import { sendFrameToWorker } from '../services/workerManager';

interface FrameMetadata {
    timestamp: number;
    frameNumber: number;
}

interface FramePacket {
    frameBuffer: ArrayBuffer;
    metadata: FrameMetadata;
}

class FrameQueue {
    private queue: FramePacket[] = [];
    private isProcessing: boolean = false;

    /**
     * Enqueues a frame packet after ensuring it's serializable.
     *
     * @param frameBuffer - The Buffer containing frame data.
     * @param metadata - Metadata associated with the frame.
     */
    public enqueue(frameBuffer: Buffer, metadata: FrameMetadata) {
        const arrayBuffer = bufferToArrayBuffer(frameBuffer);
        const packet: FramePacket = {
            frameBuffer: arrayBuffer,
            metadata,
        };
        this.queue.push(packet);
        logger.debug(`Enqueued frame #${metadata.frameNumber}`);
        this.processQueue();
    }

    /**
     * Processes the queue by dispatching frames to the worker.
     */
    private async processQueue() {
        if (this.isProcessing || this.queue.length === 0) return;

        this.isProcessing = true;
        const packet = this.queue.shift()!;

        try {
            sendFrameToWorker(packet);
            logger.debug(`Dispatched frame #${packet.metadata.frameNumber} to worker`);
        } catch (error) {
            logger.error(`Failed to dispatch frame #${packet.metadata.frameNumber}: ${error}`);
            // Optionally re-enqueue or handle the error
        } finally {
            this.isProcessing = false;
            this.processQueue(); // Continue processing remaining frames
        }
    }
}

export const frameQueue = new FrameQueue();
