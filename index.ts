import { env } from 'bun';
import ffmpeg from 'fluent-ffmpeg';
import * as fs from 'fs';
import * as path from 'path';
import winston from 'winston';
import chalk from 'chalk'; // For colorizing console output
import sharp from 'sharp';
import { createWorker, Worker } from 'tesseract.js';
import * as tf from '@tensorflow/tfjs-node';

// Custom logging format with context-specific emojis and colors
const customFormat = winston.format.printf(({ level, message, timestamp }) => {
    let emoji = '';
    let color = chalk.white;

    // Map log levels to emojis and colors
    switch (level) {
        case 'info':
            emoji = 'ℹ️';
            color = chalk.blue;
            break;
        case 'warn':
            emoji = '⚠️';
            color = chalk.yellow;
            break;
        case 'error':
            emoji = '❌';
            color = chalk.red;
            break;
        case 'debug':
            emoji = '🐛';
            color = chalk.green;
            break;
        default:
            emoji = '🔍';
            break;
    }

    // Format the log message
    return `${color(`${emoji} [${timestamp}] ${level.toUpperCase()}:`)} ${message}`;
});

function sanitizeFFmpegCommand(cmd: string): string {
    // Regex to match the RTSP URL with credentials
    const rtspUrlRegex = /(-i\s+)(rtsp:\/\/[^@]+@[^\s]+)/;

    // Replace the credentials with "*****"
    const sanitizedCmd = cmd.replace(rtspUrlRegex, (match, p1, p2) => {
        const url = new URL(p2);
        return `${p1}rtsp://*****:*****@${url.hostname}${url.pathname}`;
    });

    return sanitizedCmd;
}

// Configure the logger with the custom format
const logger = winston.createLogger({
    level: 'info',
    format: winston.format.combine(
        winston.format.timestamp({ format: 'YYYY-MM-DD HH:mm:ss' }), // Add timestamp
        winston.format.errors({ stack: true }), // Include error stacks
        customFormat // Use the custom format
    ),
    transports: [
        new winston.transports.Console(), // Log to console
        new winston.transports.File({ filename: 'error.log', level: 'error' }), // Log errors to a file
        new winston.transports.File({ filename: 'app.log' }) // Log everything to a file
    ],
});

// Context-specific emojis
const EMOJIS = {
    INIT: '🚀', // System initialisation
    DIR_CHECK: '📂', // Directory checks
    RTSP_CONNECT: '📡', // RTSP connection attempts
    FFMPEG: '🎥', // FFmpeg commands
    FRAME_WRITE: '🖼️', // Writing frames
    FRAME_PROCESS: '🔍', // Processing frames
    PEOPLE_DETECT: '👤', // Detecting people
    VEHICLE_DETECT: '🚗', // Detecting vehicles
    PLATE_DETECT: '🚘', // Detecting licence plates
    NO_PLATE: '🚫', // No licence plates detected
};

const CONFIG = {
    RTSP_URL: env.RTSP_URL, // RTSP stream URL (excluded from logs for security)
    FPS: parseInt(env.FPS || '15'), // Default to 15 FPS if not set
    FRAME_WIDTH: parseInt(env.FRAME_WIDTH || '1920'), // Default to 1920 if not set
    FRAME_HEIGHT: parseInt(env.FRAME_HEIGHT || '1080'), // Default to 1080 if not set
    MAX_RETRIES: parseInt(env.MAX_RETRIES || '3'), // Default to 3 retries if not set
    RETRY_DELAY: parseInt(env.RETRY_DELAY || '5000'), // Default to 5000 ms if not set
    DEBUG_MODE: env.DEBUG_MODE === 'true', // Convert to boolean
    MIN_VEHICLE_AREA: parseInt(env.MIN_VEHICLE_AREA || '5000'), // Default to 5000 if not set
    MAX_VEHICLE_AREA: parseInt(env.MAX_VEHICLE_AREA || '120000'), // Default to 120000 if not set
    MIN_PERSON_AREA: parseInt(env.MIN_PERSON_AREA || '5000'), // Default to 5000 if not set
    MAX_PERSON_AREA: parseInt(env.MAX_PERSON_AREA || '50000'), // Default to 50000 if not set
    PLATE_PATTERNS: {
        UK: /^[A-Z]{2}[0-9]{2}[A-Z]{3}$/,
        US: /^[A-Z0-9]{5,8}$/,
        EU: /^[A-Z]{1,2}[0-9]{1,4}[A-Z]{1,2}$/
    }
};

if (!CONFIG.RTSP_URL) {
    logger.error(`${EMOJIS.NO_PLATE} RTSP_URL environment variable is not set.`);
    process.exit(1);
}

class FrameQueue {
    private queue: string[] = [];
    private processing = false;
    private maxSize: number;

    constructor(maxSize = 10) {
        this.maxSize = maxSize;
    }

    async enqueue(framePath: string) {
        if (this.queue.length >= this.maxSize) {
            const oldFrame = this.queue.shift();
            if (oldFrame && fs.existsSync(oldFrame)) {
                await fs.promises.unlink(oldFrame);
            }
        }
        this.queue.push(framePath);
        if (!this.processing) {
            this.processQueue();
        }
    }

    private async processQueue() {
        if (this.queue.length === 0) {
            this.processing = false;
            return;
        }

        this.processing = true;
        const framePath = this.queue.shift()!;

        try {
            await processFrame(framePath);
            if (fs.existsSync(framePath)) {
                await fs.promises.unlink(framePath);
            }
        } catch (error) {
            logger.error(`${EMOJIS.NO_PLATE} Error processing frame:`, error);
        }

        setImmediate(() => this.processQueue());
    }
}

const frameQueue: FrameQueue = new FrameQueue();
const TEMP_DIR = './debug_output';
let worker: Worker | null = null;

async function initialise() {
    if (!fs.existsSync(TEMP_DIR)) {
        fs.mkdirSync(TEMP_DIR, { recursive: true });
    }
    worker = await createWorker('eng');
    logger.info(`${EMOJIS.INIT} System initialised with OCR capabilities.`);
}

interface Region {
    x: number;
    y: number;
    width: number;
    height: number;
    confidence: number;
}

interface DetectionConfig {
    minArea: number;
    maxArea: number;
    minAspectRatio: number;
    maxAspectRatio: number;
}

async function detectPeople(imagePath: string): Promise<Region[]> {
    try {
        const { data, info } = await sharp(imagePath)
            .grayscale()
            .blur(1.5)
            .threshold(140)
            .raw()
            .toBuffer({ resolveWithObject: true });

        const regions = findRegions(data, info.width, info.height, {
            minArea: CONFIG.MIN_PERSON_AREA,
            maxArea: CONFIG.MAX_PERSON_AREA,
            minAspectRatio: 0.25,
            maxAspectRatio: 0.7
        });

        if (CONFIG.DEBUG_MODE && regions.length > 0) {
            const original = sharp(imagePath);
            const svgBuffer = Buffer.from(`
                <svg width="${info.width}" height="${info.height}">
                    ${regions.map(r => `
                        <rect x="${r.x}" y="${r.y}" 
                              width="${r.width}" height="${r.height}"
                              fill="none" stroke="green" stroke-width="3"/>
                    `).join('')}
                </svg>
            `);

            await original
                .composite([{
                    input: svgBuffer,
                    top: 0,
                    left: 0
                }])
                .toFile(path.join(TEMP_DIR, `detected_people_${Date.now()}.jpg`));
        }

        return regions;
    } catch (error) {
        logger.error(`${EMOJIS.NO_PLATE} Error in person detection:`, error);
        return [];
    }
}

async function detectVehicles(imagePath: string): Promise<Region[]> {
    try {
        const { data, info } = await sharp(imagePath)
            .grayscale()
            .blur(1.0) // Reduced blur radius
            .threshold(100) // Adjusted threshold
            .raw()
            .toBuffer({ resolveWithObject: true });

        const regions = findRegions(data, info.width, info.height, {
            minArea: CONFIG.MIN_VEHICLE_AREA,
            maxArea: CONFIG.MAX_VEHICLE_AREA,
            minAspectRatio: 0.5,
            maxAspectRatio: 2.5
        });

        if (CONFIG.DEBUG_MODE) {
            const original = sharp(imagePath);
            const svgBuffer = Buffer.from(`
                <svg width="${info.width}" height="${info.height}">
                    ${regions.map(r => `
                        <rect x="${r.x}" y="${r.y}" 
                              width="${r.width}" height="${r.height}"
                              fill="none" stroke="red" stroke-width="3"/>
                    `).join('')}
                </svg>
            `);

            await original
                .composite([{
                    input: svgBuffer,
                    top: 0,
                    left: 0
                }])
                .toFile(path.join(TEMP_DIR, `detected_vehicles_${Date.now()}.jpg`));
        }

        return regions;
    } catch (error) {
        logger.error(`${EMOJIS.NO_PLATE} Error in vehicle detection:`, error);
        return [];
    }
}

function findRegions(data: Buffer, width: number, height: number, config: DetectionConfig): Region[] {
    const visited = new Set<number>();
    const regions: Region[] = [];

    for (let y = 0; y < height; y++) {
        for (let x = 0; x < width; x++) {
            const pos = y * width + x;
            if (data[pos] === 0 && !visited.has(pos)) {
                const region = floodFill(data, width, height, x, y, visited);
                const bounds = getBoundingBox(region, width);
                
                const aspectRatio = bounds.width / bounds.height;
                const area = bounds.width * bounds.height;
                
                if (area >= config.minArea && 
                    area <= config.maxArea &&
                    aspectRatio >= config.minAspectRatio && 
                    aspectRatio <= config.maxAspectRatio) {
                    // Calculate confidence as a percentage of the maximum area
                    bounds.confidence = (area / config.maxArea) * 100; 
                    regions.push(bounds);
                }
            }
        }
    }

    return filterOverlappingRegions(regions);
}

function floodFill(data: Buffer, width: number, height: number, x: number, y: number, visited: Set<number>): Set<number> {
    const region = new Set<number>();
    const stack = [{x, y}];

    while (stack.length > 0) {
        const {x: cx, y: cy} = stack.pop()!;
        const pos = cy * width + cx;

        if (cx < 0 || cx >= width || cy < 0 || cy >= height || 
            visited.has(pos) || data[pos] !== 0) {
            continue;
        }

        visited.add(pos);
        region.add(pos);

        stack.push(
            {x: cx + 1, y: cy},
            {x: cx - 1, y: cy},
            {x: cx, y: cy + 1},
            {x: cx, y: cy - 1}
        );
    }

    return region;
}

function filterOverlappingRegions(regions: Region[]): Region[] {
    return regions.filter((region, index) => {
        for (let i = 0; i < regions.length; i++) {
            if (i !== index) {
                const overlap = calculateOverlap(region, regions[i]);
                if (overlap > 0.5) {
                    return false;
                }
            }
        }
        return true;
    });
}

function calculateOverlap(r1: Region, r2: Region): number {
    const xOverlap = Math.max(0, Math.min(r1.x + r1.width, r2.x + r2.width) - Math.max(r1.x, r2.x));
    const yOverlap = Math.max(0, Math.min(r1.y + r1.height, r2.y + r2.height) - Math.max(r1.y, r2.y));
    const overlapArea = xOverlap * yOverlap;
    const r1Area = r1.width * r1.height;
    const r2Area = r2.width * r2.height;
    return overlapArea / Math.min(r1Area, r2Area);
}

function getBoundingBox(region: Set<number>, width: number): Region {
    let minX = Infinity, minY = Infinity, maxX = -Infinity, maxY = -Infinity;

    for (const pos of region) {
        const x = pos % width;
        const y = Math.floor(pos / width);
        minX = Math.min(minX, x);
        maxX = Math.max(maxX, x);
        minY = Math.min(minY, y);
        maxY = Math.max(maxY, y);
    }

    const widthBox = maxX - minX;
    const heightBox = maxY - minY;
    const area = widthBox * heightBox;

    return {
        x: minX,
        y: minY,
        width: widthBox,
        height: heightBox,
        confidence: 0 // Placeholder, will be updated in findRegions
    };
}

interface OCRWord {
    text: string;
    confidence: number;
}

async function performOCR(imagePath: string): Promise<OCRWord[]> {
    try {
        const { data } = await worker.recognize(imagePath);
        
        // Map words to a strictly typed array
        const words: OCRWord[] = data.words.map((word: { text: string; confidence: number }) => ({
            text: word.text,
            confidence: word.confidence
        })).filter((word: OCRWord) => {
            return Object.values(CONFIG.PLATE_PATTERNS).some(pattern => 
                pattern.test(word.text.toUpperCase())
            );
        });
        
        return words;
    } catch (error) {
        logger.error(`${EMOJIS.NO_PLATE} Error performing OCR:`, error);
        return [];
    }
}

async function processFrame(framePath: string) {
    try {
        logger.info(`${EMOJIS.FRAME_PROCESS} Processing frame: ${framePath}`);
        if (!fs.existsSync(framePath)) {
            logger.error(`${EMOJIS.NO_PLATE} Frame file does not exist`);
            return;
        }

        // Detect people
        const people = await detectPeople(framePath);
        if (people.length > 0) {
            const peopleConfidences = people
                .map(p => `👤 Person at (${p.x}, ${p.y}): ${p.confidence.toFixed(1)}% confidence`)
                .join('\n');
            logger.info(`${EMOJIS.PEOPLE_DETECT} Detected ${people.length} people in frame:\n${peopleConfidences}`);
        }

        // Process vehicles and plates
        const imageBuffer = await fs.promises.readFile(framePath);
        const tensor = tf.node.decodeImage(imageBuffer);
        
        const vehicles = await detectVehicles(framePath);
        if (vehicles.length === 0) {
            // No vehicles detected: 100% confidence
            logger.info(`${EMOJIS.VEHICLE_DETECT} Detected 0 vehicles in frame with 100% confidence`);
        } else {
            // Vehicles detected: log confidences
            const vehicleConfidences = vehicles
                .map(v => `🚗 Vehicle at (${v.x}, ${v.y}): ${v.confidence.toFixed(1)}% confidence`)
                .join('\n');
            logger.info(`${EMOJIS.VEHICLE_DETECT} Detected ${vehicles.length} vehicles in frame:\n${vehicleConfidences}`);
        }
        
        if (vehicles.length === 0 && people.length === 0) {
            logger.info(`${EMOJIS.NO_PLATE} No objects detected in this frame`);
            tensor.dispose();
            return;
        }
        
        let plateDetected = false;
        for (const vehicle of vehicles) {
            try {
                let vehicleTensor = tf.slice(tensor, [vehicle.y, vehicle.x, 0], [vehicle.height, vehicle.width, 3]);

                // Ensure the tensor is 3D (height, width, channels)
                vehicleTensor = vehicleTensor.squeeze(); // Remove extra dimensions (e.g., batch dimension)

                // Cast to Tensor3D to satisfy TypeScript
                const vehicleBuffer = await tf.node.encodePng(vehicleTensor as tf.Tensor3D);
                const vehiclePath = path.join(TEMP_DIR, `vehicle_${Date.now()}.jpg`);
                await fs.promises.writeFile(vehiclePath, vehicleBuffer);
                
                const plates = await performOCR(vehiclePath);
                if (plates.length > 0) {
                    const plateConfidences = plates
                        .map(p => `🚘 Detected plate: ${p.text} (${(p.confidence * 100).toFixed(1)}% confidence)`)
                        .join('\n');
                    logger.info(`${EMOJIS.PLATE_DETECT} Detected licence plates:\n${plateConfidences}`);
                    plateDetected = true;
                }
                
                await fs.promises.unlink(vehiclePath);
                vehicleTensor.dispose();
            } catch (error) {
                logger.error(`${EMOJIS.NO_PLATE} Error processing vehicle region:`, error);
            }
        }
        
        if (!plateDetected && vehicles.length > 0) {
            logger.info(`${EMOJIS.NO_PLATE} No licence plates detected in any vehicle regions`);
        }
        
        tensor.dispose();
    } catch (error) {
        logger.error(`${EMOJIS.NO_PLATE} Error processing frame:`, error);
    }
}

async function startStreamProcessing(retries = CONFIG.MAX_RETRIES, delay = CONFIG.RETRY_DELAY) {
    let attempt = 0;

    const processStream = () => {
        try {
            fs.accessSync(TEMP_DIR, fs.constants.W_OK);
        } catch (error) {
            logger.error(`${EMOJIS.NO_PLATE} Directory not writable: ${error}`);
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
            .outputOptions([
                '-frames:v 1',
                '-update 1',
                '-y'
            ])
            .on('start', (cmdline) => {
                // Sanitize the FFmpeg command before logging
                const sanitizedCmd = sanitizeFFmpegCommand(cmdline);
                logger.info(`${EMOJIS.FFMPEG} FFmpeg command: ${sanitizedCmd}`);
                logger.info(`${EMOJIS.FRAME_WRITE} Writing frames to: ${path.join(TEMP_DIR, 'frame.jpg')}`);
            })
            .on('error', (err) => {
                logger.error(`${EMOJIS.NO_PLATE} Error processing RTSP stream: ${err}`);

                if (attempt < retries - 1) {
                    attempt++;
                    logger.info(`${EMOJIS.RTSP_CONNECT} Retrying in ${delay / 1000} seconds...`);
                    setTimeout(processStream, delay);
                } else {
                    logger.error(`${EMOJIS.NO_PLATE} Max retries reached. Exiting...`);
                    process.exit(1);
                }
            })
            .output(path.join(TEMP_DIR, 'frame.jpg'))
            .on('end', () => {
                processFrame(path.join(TEMP_DIR, 'frame.jpg'));
            });

        stream.run();
    };

    processStream();
}

async function cleanup() {
    logger.info(`${EMOJIS.NO_PLATE} Cleaning up...`);
    
    if (worker) {
        await worker.terminate();
    }
    
    if (fs.existsSync(TEMP_DIR)) {
        await fs.promises.rm(TEMP_DIR, { recursive: true, force: true });
    }
    
    process.exit(0);
}

process.on('SIGINT', cleanup);
process.on('SIGTERM', cleanup);
process.on('uncaughtException', (error) => {
    logger.error(`${EMOJIS.NO_PLATE} Uncaught exception: ${error}`);
    cleanup();
});

function logConfiguration(config: typeof CONFIG) {
    logger.info(`${EMOJIS.INIT} Application configuration:`);
    logger.info(`${EMOJIS.INIT} ----------------------------------------`);
    logger.info(`${EMOJIS.INIT} FPS: ${config.FPS}`);
    logger.info(`${EMOJIS.INIT} Frame Width: ${config.FRAME_WIDTH}`);
    logger.info(`${EMOJIS.INIT} Frame Height: ${config.FRAME_HEIGHT}`);
    logger.info(`${EMOJIS.INIT} Max Retries: ${config.MAX_RETRIES}`);
    logger.info(`${EMOJIS.INIT} Retry Delay: ${config.RETRY_DELAY} ms`);
    logger.info(`${EMOJIS.INIT} Debug Mode: ${config.DEBUG_MODE ? 'Enabled' : 'Disabled'}`);
    logger.info(`${EMOJIS.INIT} Min Vehicle Area: ${config.MIN_VEHICLE_AREA}`);
    logger.info(`${EMOJIS.INIT} Max Vehicle Area: ${config.MAX_VEHICLE_AREA}`);
    logger.info(`${EMOJIS.INIT} Min Person Area: ${config.MIN_PERSON_AREA}`);
    logger.info(`${EMOJIS.INIT} Max Person Area: ${config.MAX_PERSON_AREA}`);
    logger.info(`${EMOJIS.INIT} ----------------------------------------`);
}

(async () => {
    await initialise();

    // Log the configuration at startup
    logConfiguration(CONFIG);

    startStreamProcessing();
    
    // Keep process alive
    process.stdin.resume();
})();