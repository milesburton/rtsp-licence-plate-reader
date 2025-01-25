import ffmpeg from 'fluent-ffmpeg';
import { PassThrough } from 'stream';
import { frameQueue } from '../utils/queue';
import logger from '../utils/logger';
import CONFIG from '../config';

export class StreamProcessor {
    private command: ffmpeg.FfmpegCommand | null = null;
    private stream: PassThrough | null = null;
    private retryCount: number = 0;
    private readonly maxRetries: number = CONFIG.MAX_RETRIES;
    private readonly initialRetryDelay: number = CONFIG.RETRY_DELAY; // in milliseconds
    private readonly backoffFactor: number = 2; // Exponential backoff multiplier
    private frameCounter: number = 0; // To uniquely identify frames

    /**
     * Starts processing the RTSP stream.
     */
    public startStream(): void {
        logger.info(`📡 Starting RTSP stream processing for URL: ${CONFIG.RTSP_URL}`);
        this.connectStream();
    }

    /**
     * Connects to the RTSP stream using FFmpeg.
     */
    private connectStream(): void {
        // Initialize PassThrough stream to handle FFmpeg output
        this.stream = new PassThrough();

        // Set up FFmpeg command
        this.command = ffmpeg(CONFIG.RTSP_URL)
            .addInputOption('-rtsp_transport', 'tcp') // Use TCP for RTSP transport
            .inputFormat('rtsp')
            .outputFormat('image2pipe') // Output as a pipe of images
            .outputOptions('-vcodec', 'mjpeg') // Use MJPEG codec
            .on('start', (commandLine) => {
                logger.info(`🎬 FFmpeg started with command: ${commandLine}`);
                this.retryCount = 0; // Reset retry count on successful start
            })
            .on('error', (err, stdout, stderr) => {
                logger.error(`❌ FFmpeg error: ${err.message}`);
                logger.debug(`FFmpeg stdout: ${stdout}`);
                logger.debug(`FFmpeg stderr: ${stderr}`);
                this.handleStreamError();
            })
            .on('end', () => {
                logger.warn('⚠️ FFmpeg stream ended unexpectedly.');
                this.handleStreamError();
            })
            .pipe(this.stream, { end: true });

        // Handle incoming data (frames) from FFmpeg
        this.stream.on('data', async (chunk: Buffer) => {
            try {
                await this.handleFrame(chunk);
            } catch (error) {
                logger.error(`❌ Error handling frame: ${error}`);
            }
        });

        this.stream.on('error', (err) => {
            logger.error(`❌ Stream error: ${err.message}`);
            this.handleStreamError();
        });
    }

    /**
     * Handles individual frames extracted from the RTSP stream.
     *
     * @param chunk - The buffer containing frame data.
     */
    private async handleFrame(chunk: Buffer): Promise<void> {
        // Increment frame counter for unique identification
        this.frameCounter += 1;

        logger.debug(`🖼️ Received frame #${this.frameCounter} of size: ${chunk.length} bytes`);

        // Metadata can include timestamp and frame number
        const metadata = {
            timestamp: Date.now(),
            frameNumber: this.frameCounter,
        };

        // Enqueue the frame for processing
        frameQueue.enqueue(chunk, metadata);
    }

    /**
     * Handles stream errors by attempting to reconnect with exponential backoff.
     */
    private handleStreamError(): void {
        if (this.retryCount < this.maxRetries) {
            const delay = this.initialRetryDelay * Math.pow(this.backoffFactor, this.retryCount);
            logger.warn(`⚠️ Attempting to reconnect... (${this.retryCount + 1}/${this.maxRetries}) in ${delay / 1000}s`);

            setTimeout(() => {
                this.retryCount += 1;
                this.connectStream();
            }, delay);
        } else {
            logger.error('❌ Max reconnection attempts reached. Exiting application.');
            process.exit(1);
        }
    }

    /**
     * Stops processing the RTSP stream gracefully.
     */
    public stopStream(): void {
        logger.info('🛑 Stopping RTSP stream processing...');
        if (this.command) {
            this.command.kill('SIGKILL');
            this.command = null;
        }
        if (this.stream) {
            this.stream.end();
            this.stream = null;
        }
    }
}
