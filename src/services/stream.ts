import ffmpeg from 'fluent-ffmpeg';
import * as fs from 'fs';
import * as path from 'path';
import { logger } from '../utils/logger';
import { CONFIG, TEMP_DIR } from '../config';
import { processFrame } from './frame-processing';
import { EMOJIS } from '../utils/logger';
import { isShuttingDown } from './cleanup';

export class FrameQueue {
    private queue: Buffer[] = [];
    private processing = false;
    private maxSize: number;

    constructor(maxSize = 10) {
        this.maxSize = maxSize;
    }

    async enqueue(buffer: Buffer) {
        if (isShuttingDown) return;

        if (this.queue.length >= this.maxSize) {
            logger.warn(`${EMOJIS.FRAME_WRITE} Queue is full (size: ${this.queue.length}). Discarding oldest frame.`);
            this.queue.shift();
        }
        this.queue.push(buffer);
        if (!this.processing) {
            this.processQueue();
        }
    }

    private async processQueue() {
        if (this.queue.length === 0 || isShuttingDown) {
            this.processing = false;
            return;
        }

        this.processing = true;
        const buffer = this.queue.shift()!;

        try {
            await processFrame(buffer);
        } catch (error) {
            logger.error(`${EMOJIS.NO_PLATE} Error processing frame:`, error);
            if (error instanceof Error) {
                logger.error(`${EMOJIS.NO_PLATE} Stack trace:`, error.stack);
            }
        }

        setImmediate(() => this.processQueue());
    }
}

export const frameQueue: FrameQueue = new FrameQueue(CONFIG.FRAME_QUEUE_SIZE);

export function startStreamProcessing(retries = CONFIG.MAX_RETRIES, delay = CONFIG.RETRY_DELAY) {
    let attempt = 0;

    const processStream = () => {
        if (isShuttingDown) return;

        try {
            fs.accessSync(TEMP_DIR, fs.constants.W_OK);
        } catch (error) {
            logger.error(`${EMOJIS.NO_PLATE} Directory not writable:`, error);
            process.exit(1);
        }
        logger.info(`${EMOJIS.DIR_CHECK} Directory checks passed`);

        logger.info(`${EMOJIS.RTSP_CONNECT} Attempting to connect to RTSP stream (Attempt ${attempt + 1}/${retries})...`);

        const stream = ffmpeg(CONFIG.RTSP_URL)
            .inputOptions([
                '-rtsp_transport tcp',
                '-stimeout 5000000',
                '-fflags nobuffer',
                '-flags low_delay'
            ])
            .fps(CONFIG.FPS)
            .format('image2pipe')
            .outputOptions([
                '-vcodec mjpeg',
                '-pix_fmt yuvj420p'
            ])
            .on('start', (cmdline) => {
                const sanitizedCmd = sanitizeFFmpegCommand(cmdline);
                logger.info(`${EMOJIS.FFMPEG} FFmpeg command: ${sanitizedCmd}`);
            })
            .on('error', (err) => {
                if (isShuttingDown) return;
                logger.error(`${EMOJIS.NO_PLATE} FFmpeg stream error:`, err);
                if (err instanceof Error) {
                    logger.error(`${EMOJIS.NO_PLATE} Stack trace:`, err.stack);
                }
                if (attempt < retries - 1) {
                    attempt++;
                    logger.info(`${EMOJIS.RTSP_CONNECT} Retrying in ${delay / 1000} seconds...`);
                    setTimeout(processStream, delay);
                } else {
                    logger.error(`${EMOJIS.NO_PLATE} Max retries reached. Exiting...`);
                    process.exit(1);
                }
            })
            .on('end', () => {
                if (isShuttingDown) return;
                logger.info(`${EMOJIS.FRAME_WRITE} Stream ended unexpectedly, attempting to restart...`);
                setTimeout(processStream, delay);
            })
            .pipe();

        stream.on('data', async (buffer) => {
            await frameQueue.enqueue(buffer);
        });
    };

    processStream();
}

export function sanitizeFFmpegCommand(cmd: string): string {
    const rtspUrlRegex = /(-i\s+)(rtsp:\/\/[^@]+@[^\s]+)/;
    const sanitizedCmd = cmd.replace(rtspUrlRegex, (match, p1, p2) => {
        try {
            const url = new URL(p2);
            return `${p1}rtsp://*****:*****@${url.hostname}${url.pathname}`;
        } catch {
            return match;
        }
    });
    return sanitizedCmd;
}
