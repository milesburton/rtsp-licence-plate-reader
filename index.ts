// index.ts

import { env } from 'bun';
import ffmpeg from 'fluent-ffmpeg';
import * as fs from 'fs';
import * as path from 'path';
import winston from 'winston';
import chalk from 'chalk';
import sharp from 'sharp';
import { createWorker, Worker } from 'tesseract.js';
import * as tf from '@tensorflow/tfjs-node';
import * as cocoSsd from '@tensorflow-models/coco-ssd';

/**
 * Enumeration for different vehicle types.
 */
enum VehicleType {
    Car = "Car",
    Motorcycle = "Motorcycle",
    Bus = "Bus",
    Van = "Van",
    Truck = "Truck",
    Bicycle = "Bicycle",
    Scooter = "Scooter",
    Unknown = "Unknown",
}

/**
 * Represents a detected region in the image.
 */
interface Region {
    x: number;
    y: number;
    width: number;
    height: number;
    confidence: number;
}

/**
 * Represents a detected vehicle region with its type.
 */
interface VehicleRegion extends Region {
    type: VehicleType;
}

/**
 * Configuration for detection algorithms.
 */
interface DetectionConfig {
    minArea: number;
    maxArea: number;
    minAspectRatio: number;
    maxAspectRatio: number;
}

/**
 * Represents a recognized OCR word.
 */
interface OCRWord {
    text: string;
    confidence: number;
}

/**
 * Defines the structure of the OCR result data.
 */
interface OCRResult {
    data: {
        text: string;
        words?: Array<{
            text: string;
            confidence: number;
            bbox?: [number, number, number, number];
        }>;
    };
}

async function loadImageAsTensor(imagePath: string): Promise<tf.Tensor3D> {
    // Load image using sharp
    const { data, info } = await sharp(imagePath)
        .resize(640, 480) // Resize for consistent processing
        .toFormat('png')   // Ensure a supported format
        .removeAlpha()      // ✅ Ensure RGB by removing alpha channel
        .raw()
        .toBuffer({ resolveWithObject: true });

    // Convert image data to TensorFlow tensor
    return tf.tensor(new Uint8Array(data), [info.height, info.width, 3]); // ✅ Correct shape
}


/**
 * Classify vehicle type based on COCO-SSD class name.
 * @param className COCO-SSD detected class name.
 * @returns Corresponding VehicleType.
 */
function mapCOCOSSDClassToVehicleType(className: string): VehicleType {
    switch (className.toLowerCase()) {
        case 'car':
            return VehicleType.Car;
        case 'motorcycle':
        case 'motorbike':
            return VehicleType.Motorcycle;
        case 'bus':
            return VehicleType.Bus;
        case 'truck':
            return VehicleType.Truck;
        case 'bicycle':
            return VehicleType.Bicycle;
        case 'van':
            return VehicleType.Van;
        default:
            return VehicleType.Unknown;
    }
}

/**
 * Get color based on VehicleType for visualization.
 * @param type VehicleType.
 * @returns Color string.
 */
function getColorByVehicleType(type: VehicleType): string {
    switch (type) {
        case VehicleType.Car:
            return "blue";
        case VehicleType.Motorcycle:
            return "yellow";
        case VehicleType.Bus:
            return "red";
        case VehicleType.Van:
            return "green";
        case VehicleType.Truck:
            return "purple";
        case VehicleType.Bicycle:
            return "orange";
        case VehicleType.Scooter:
            return "cyan";
        default:
            return "gray";
    }
}

/**
 * Utility function to measure processing time.
 * @param label Label for the timed section.
 * @param startTime Start time in nanoseconds.
 */
function measureTime(label: string, startTime: number) {
    const endTime = process.hrtime.bigint();
    const duration = Number(endTime - BigInt(startTime)) / 1e6;
    logger.debug(`${EMOJIS.DEBUG} ${label} took ${duration.toFixed(2)}ms`);
}

/**
 * Finds regions in the image data that match the detection configuration.
 * @param data Image data buffer.
 * @param width Image width.
 * @param height Image height.
 * @param config Detection configuration.
 * @returns Array of detected regions.
 */
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
                    bounds.confidence = (area / config.maxArea) * 100;
                    regions.push(bounds);
                }
            }
        }
    }

    return filterOverlappingRegions(regions);
}

/**
 * Performs flood fill to identify connected regions.
 * @param data Image data buffer.
 * @param width Image width.
 * @param height Image height.
 * @param x Starting x-coordinate.
 * @param y Starting y-coordinate.
 * @param visited Set of visited positions.
 * @returns Set of positions belonging to the region.
 */
function floodFill(data: Buffer, width: number, height: number, x: number, y: number, visited: Set<number>): Set<number> {
    const region = new Set<number>();
    const stack = [{ x, y }];

    while (stack.length > 0) {
        const { x: cx, y: cy } = stack.pop()!;
        const pos = cy * width + cx;

        if (cx < 0 || cx >= width || cy < 0 || cy >= height ||
            visited.has(pos) || data[pos] !== 0) {
            continue;
        }

        visited.add(pos);
        region.add(pos);

        stack.push(
            { x: cx + 1, y: cy },
            { x: cx - 1, y: cy },
            { x: cx, y: cy + 1 },
            { x: cx, y: cy - 1 }
        );
    }

    return region;
}

/**
 * Filters out overlapping regions based on overlap criteria.
 * @param regions Array of detected regions.
 * @returns Filtered array of regions.
 */
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

/**
 * Calculates the overlap ratio between two regions.
 * @param r1 First region.
 * @param r2 Second region.
 * @returns Overlap ratio.
 */
function calculateOverlap(r1: Region, r2: Region): number {
    const xOverlap = Math.max(0, Math.min(r1.x + r1.width, r2.x + r2.width) - Math.max(r1.x, r2.x));
    const yOverlap = Math.max(0, Math.min(r1.y + r1.height, r2.y + r2.height) - Math.max(r1.y, r2.y));
    const overlapArea = xOverlap * yOverlap;
    const r1Area = r1.width * r1.height;
    const r2Area = r2.width * r2.height;
    return overlapArea / Math.min(r1Area, r2Area);
}

/**
 * Gets the bounding box for a region.
 * @param region Set of positions in the region.
 * @param width Image width.
 * @returns Bounding box as a Region.
 */
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

    return {
        x: minX,
        y: minY,
        width: maxX - minX,
        height: maxY - minY,
        confidence: 0
    };
}

// Custom logging format
const customFormat = winston.format.printf(({ level, message, timestamp }) => {
    let emoji = '';
    let color = chalk.white;

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

    return `${color(`${emoji} [${timestamp}] ${level.toUpperCase()}:`)} ${message}`;
});

// Define log directory
const logDir = path.resolve(__dirname, 'logs');

// Ensure log directory exists
if (!fs.existsSync(logDir)) {
    try {
        fs.mkdirSync(logDir, { recursive: true });
        console.log(`Created log directory at ${logDir}`);
    } catch (error) {
        console.error(`Failed to create log directory at ${logDir}:`, error);
        process.exit(1);
    }
}

// Initialize Winston logger
const logger = winston.createLogger({
    level: 'info',
    format: winston.format.combine(
        winston.format.timestamp({ format: 'YYYY-MM-DD HH:mm:ss' }),
        winston.format.errors({ stack: true }),
        customFormat
    ),
    transports: [
        new winston.transports.Console(),
        new winston.transports.File({ filename: path.join(logDir, 'error.log'), level: 'error' }),
        new winston.transports.File({ filename: path.join(logDir, 'app.log') })
    ],
});

// Emojis for logging
const EMOJIS = {
    INIT: '🚀',
    DIR_CHECK: '📂',
    RTSP_CONNECT: '📡',
    FFMPEG: '🎥',
    FRAME_WRITE: '🖼️',
    FRAME_PROCESS: '🔍',
    PEOPLE_DETECT: '👤',
    VEHICLE_DETECT: '🚗',
    PLATE_DETECT: '🚘',
    NO_PLATE: '🚫',
    DEBUG: '🐞',
};

/**
 * Validates environment variables and sets defaults with detailed logging.
 */
function validateConfig() {
    const requiredVars = ['RTSP_URL'];
    const missingVars = requiredVars.filter(varName => !env[varName]);

    if (missingVars.length > 0) {
        logger.error(`${EMOJIS.NO_PLATE} Missing required environment variables: ${missingVars.join(', ')}`);
        process.exit(1);
    }

    const numericVars = ['FPS', 'FRAME_WIDTH', 'FRAME_HEIGHT', 'MAX_RETRIES', 'RETRY_DELAY',
        'MIN_VEHICLE_AREA', 'MAX_VEHICLE_AREA', 'MIN_PERSON_AREA', 'MAX_PERSON_AREA'];

    numericVars.forEach(varName => {
        const value = env[varName];
        if (value && isNaN(Number(value))) {
            logger.error(`${EMOJIS.NO_PLATE} Invalid numeric value for ${varName}: ${value}`);
            process.exit(1);
        }
    });
}

// Validate configuration
validateConfig();

// Configuration object
const CONFIG = {
    RTSP_URL: env.RTSP_URL,
    FPS: parseInt(env.FPS || '15'),
    FRAME_WIDTH: parseInt(env.FRAME_WIDTH || '1920'),
    FRAME_HEIGHT: parseInt(env.FRAME_HEIGHT || '1080'),
    MAX_RETRIES: parseInt(env.MAX_RETRIES || '3'),
    RETRY_DELAY: parseInt(env.RETRY_DELAY || '5000'),
    DEBUG_MODE: env.DEBUG_MODE === 'true',
    MIN_VEHICLE_AREA: parseInt(env.MIN_VEHICLE_AREA || '5000'),
    MAX_VEHICLE_AREA: parseInt(env.MAX_VEHICLE_AREA || '120000'),
    MIN_PERSON_AREA: parseInt(env.MIN_PERSON_AREA || '5000'),
    MAX_PERSON_AREA: parseInt(env.MAX_PERSON_AREA || '50000'),
    PLATE_PATTERNS: {
        UK: /^[A-Z]{2}[0-9]{2}[A-Z]{3}$/,
        US: /^[A-Z0-9]{5,8}$/,
        EU: /^[A-Z]{1,2}[0-9]{1,4}[A-Z]{1,2}$/
    },
    FRAME_QUEUE_SIZE: parseInt(env.FRAME_QUEUE_SIZE || '30'),
};

/**
 * Manages a queue of frames to be processed.
 */
class FrameQueue {
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

const frameQueue: FrameQueue = new FrameQueue(CONFIG.FRAME_QUEUE_SIZE);
const TEMP_DIR = path.resolve(import.meta.dir, 'debug_output');

let worker: Worker | null = null;
let objectDetectionModel: cocoSsd.ObjectDetection;

/**
 * Initializes necessary components like TensorFlow, OCR worker, and object detection model.
 */
async function initialise() {
    // Ensure TEMP_DIR exists
    if (!fs.existsSync(TEMP_DIR)) {
        try {
            fs.mkdirSync(TEMP_DIR, { recursive: true });
            logger.info(`${EMOJIS.INIT} Created temporary directory at ${TEMP_DIR}`);
        } catch (error) {
            logger.error(`${EMOJIS.NO_PLATE} Failed to create temporary directory:`, error);
            process.exit(1);
        }
    }

    // Initialize TensorFlow
    try {
        await tf.ready();
        logger.info(`${EMOJIS.INIT} TensorFlow initialized`);
    } catch (error) {
        logger.error(`${EMOJIS.NO_PLATE} Failed to initialize TensorFlow:`, error);
        process.exit(1);
    }

    // Initialize OCR Worker
    try {
        worker = await createWorker('eng'); // British English is covered under 'eng'
        
        // Progress logging needs to be done inside recognize(), not via worker.on()
        logger.info(`${EMOJIS.INIT} OCR Worker initialized`);
    } catch (error) {
        logger.error(`${EMOJIS.NO_PLATE} Failed to initialize OCR Worker:`, error);
        process.exit(1);
    }

    // Load the COCO-SSD model
    try {
        objectDetectionModel = await cocoSsd.load();
        logger.info(`${EMOJIS.INIT} COCO-SSD Object Detection model loaded`);
    } catch (error) {
        logger.error(`${EMOJIS.NO_PLATE} Failed to load COCO-SSD model:`, error);
        process.exit(1);
    }
}

/**
 * Detects people in the given image.
 * @param imagePath Path to the image file.
 * @returns Array of detected regions representing people.
 */
async function detectPeople(imagePath: string): Promise<Region[]> {
    const startTime = process.hrtime.bigint();
    try {
        const { data, info } = await sharp(imagePath)
            .grayscale()
            .blur(1.5)
            .threshold(140)
            .raw()
            .toBuffer({ resolveWithObject: true });

        logger.debug(`${EMOJIS.DEBUG} Image preprocessing completed for person detection`);
        const regions = findRegions(data, info.width, info.height, {
            minArea: CONFIG.MIN_PERSON_AREA,
            maxArea: CONFIG.MAX_PERSON_AREA,
            minAspectRatio: 0.25,
            maxAspectRatio: 0.7
        });

        if (CONFIG.DEBUG_MODE && regions.length > 0) {
            const svgBuffer = Buffer.from(`
                <svg width="${info.width}" height="${info.height}">
                    ${regions.map(r => `
                        <rect x="${r.x}" y="${r.y}" 
                              width="${r.width}" height="${r.height}"
                              fill="none" stroke="green" stroke-width="3"/>
                    `).join('')}
                </svg>
            `);

            await sharp(imagePath)
                .composite([{
                    input: svgBuffer,
                    top: 0,
                    left: 0
                }])
                .toFile(path.join(TEMP_DIR, `detected_people_${Date.now()}.jpg`));

            logger.debug(`${EMOJIS.DEBUG} Debug visualization saved for person detection`);
        }

        measureTime('Person detection', Number(startTime));
        return regions;
    } catch (error) {
        logger.error(`${EMOJIS.NO_PLATE} Error in person detection:`, error);
        if (error instanceof Error) {
            logger.error(`${EMOJIS.NO_PLATE} Stack trace:`, error.stack);
        }
        logger.debug(`${EMOJIS.DEBUG} Image path: ${imagePath}`);
        return [];
    }
}

/**
 * Detects vehicles in the given image using COCO-SSD model.
 * @param imagePath Path to the image file.
 * @returns Array of detected VehicleRegion.
 */
async function detectVehicles(imagePath: string): Promise<VehicleRegion[]> {
    const startTime = process.hrtime.bigint();
    try {
        const decodedImage = await loadImageAsTensor(imagePath);
        const predictions = await objectDetectionModel.detect(decodedImage);

        const vehiclePredictions = predictions.filter(pred =>
            ['car', 'motorcycle', 'bus', 'truck', 'bicycle', 'motorbike', 'van'].includes(pred.class.toLowerCase())
        );

        const vehicleRegions: VehicleRegion[] = vehiclePredictions.map(pred => ({
            x: Math.round(pred.bbox[0]),
            y: Math.round(pred.bbox[1]),
            width: Math.round(pred.bbox[2]),
            height: Math.round(pred.bbox[3]),
            confidence: parseFloat((pred.score * 100).toFixed(1)),
            type: mapCOCOSSDClassToVehicleType(pred.class.toLowerCase())
        }));

        decodedImage.dispose(); // Free up memory
        return vehicleRegions;
    } catch (error) {
        logger.error(`${EMOJIS.NO_PLATE} Error in vehicle detection:`, error);
        return [];
    }
}

/**
 * Performs OCR on the given image to detect license plates.
 * @param imagePath Path to the image file.
 * @returns Array of recognized OCR words representing license plates.
 */
async function performOCR(imagePath: string): Promise<OCRWord[]> {
    const startTime = process.hrtime.bigint();
    try {
        if (!worker) {
            logger.error(`${EMOJIS.NO_PLATE} OCR Worker is not initialized`);
            return [];
        }

        // Perform OCR
        const result = await worker.recognize(imagePath);
        logger.debug(`${EMOJIS.DEBUG} OCR processing completed`);

        if (!result.data.text || result.data.text.trim().length === 0) {
            logger.warn(`${EMOJIS.NO_PLATE} No text detected in OCR result.`);
            return [];
        }

        // Split text into potential words (basic whitespace split)
        const words = result.data.text
            .split(/\s+/)
            .map(word => word.toUpperCase().replace(/[^A-Z0-9]/g, '')) // Remove non-alphanumeric characters
            .filter(word => word.length >= 5 && word.length <= 8) // Plate number length range
            .map(word => ({
                text: word,
                confidence: result.data.confidence / 100 // Approximate confidence from the entire page
            }));

        // Filter based on regex patterns
        const filteredWords = words.filter(word =>
            Object.values(CONFIG.PLATE_PATTERNS).some(pattern => pattern.test(word.text))
        );

        measureTime('OCR processing', Number(startTime));

        if (filteredWords.length > 0) {
            logger.info(`${EMOJIS.PLATE_DETECT} Found ${filteredWords.length} potential license plates`);
            filteredWords.forEach(word => {
                logger.debug(`${EMOJIS.DEBUG} Plate candidate: ${word.text} (Approx. ${(word.confidence * 100).toFixed(1)}% confidence)`);
            });
        }

        return filteredWords;
    } catch (error) {
        logger.error(`${EMOJIS.NO_PLATE} Error performing OCR:`, error);
        if (error instanceof Error) {
            logger.error(`${EMOJIS.NO_PLATE} Stack trace:`, error.stack);
        }
        logger.debug(`${EMOJIS.DEBUG} Image path: ${imagePath}`);

        // Attempt to reinitialize the worker in case of failure
        if (worker) {
            await worker.terminate();
            try {
                worker = await createWorker('eng'); // British English
                logger.info(`${EMOJIS.INIT} OCR Worker reinitialized`);
            } catch (initError) {
                logger.error(`${EMOJIS.NO_PLATE} Failed to reinitialize OCR Worker:`, initError);
                process.exit(1);
            }
        }

        return [];
    }
}


/**
 * Processes a single frame buffer for detection tasks.
 * @param buffer Frame buffer.
 */
async function processFrame(buffer: Buffer) {
    try {
        logger.info(`${EMOJIS.FRAME_PROCESS} Processing frame (Queue size: ${frameQueue['queue'].length})`);

        const tensor = tf.node.decodeImage(buffer) as tf.Tensor3D;
        logger.debug(`${EMOJIS.DEBUG} Created tensor with shape: ${tensor.shape} and dtype: ${tensor.dtype}`);

        // Write buffer to temp file for Sharp processing
        const tempFramePath = path.join(TEMP_DIR, `frame_${Date.now()}.jpg`);
        await fs.promises.writeFile(tempFramePath, buffer);
        logger.debug(`${EMOJIS.DEBUG} Frame written to ${tempFramePath}`);

        // Detect people
        const people = await detectPeople(tempFramePath);
        if (people.length > 0) {
            const peopleConfidences = people
                .map(p => `👤 Person at (${p.x}, ${p.y}): ${p.confidence.toFixed(1)}% confidence`)
                .join('\n');
            logger.info(`${EMOJIS.PEOPLE_DETECT} Detected ${people.length} people in frame:\n${peopleConfidences}`);
        }

        // Detect vehicles and process potential plates
        const vehicles: VehicleRegion[] = await detectVehicles(tempFramePath);
        if (vehicles.length > 0) {
            const vehicleDetails = vehicles
                .map(v => `🚗 ${v.type} at (${v.x}, ${v.y}): ${v.confidence.toFixed(1)}% confidence`)
                .join('\n');
            logger.info(`${EMOJIS.VEHICLE_DETECT} Detected ${vehicles.length} vehicles in frame:\n${vehicleDetails}`);

            // Process each vehicle for license plates
            for (const vehicle of vehicles) {
                try {
                    // Extract vehicle region tensor
                    const vehicleTensor = tf.slice(tensor, [vehicle.y, vehicle.x, 0], [vehicle.height, vehicle.width, 3]) as tf.Tensor3D;
                    const vehicleBuffer = await tf.node.encodeJpeg(vehicleTensor);
                    const vehiclePath = path.join(TEMP_DIR, `vehicle_${Date.now()}.jpg`);
                    await fs.promises.writeFile(vehiclePath, vehicleBuffer);
                    logger.debug(`${EMOJIS.DEBUG} Vehicle region written to ${vehiclePath}`);

                    // Perform OCR on the vehicle image
                    const plates = await performOCR(vehiclePath);
                    if (plates.length > 0) {
                        const plateConfidences = plates
                            .map(p => `🚘 Detected plate: ${p.text} (${(p.confidence * 100).toFixed(1)}% confidence)`)
                            .join('\n');
                        logger.info(`${EMOJIS.PLATE_DETECT} Detected license plates:\n${plateConfidences}`);
                    }

                    // Clean up
                    await fs.promises.unlink(vehiclePath);
                    logger.debug(`${EMOJIS.DEBUG} Vehicle region file deleted: ${vehiclePath}`);
                    vehicleTensor.dispose();
                } catch (error) {
                    logger.error(`${EMOJIS.NO_PLATE} Error processing vehicle region:`, error);
                    if (error instanceof Error) {
                        logger.error(`${EMOJIS.NO_PLATE} Stack trace:`, error.stack);
                    }
                }
            }
        }

        // Clean up
        await fs.promises.unlink(tempFramePath);
        logger.debug(`${EMOJIS.DEBUG} Frame file deleted: ${tempFramePath}`);
        tensor.dispose();

    } catch (error) {
        logger.error(`${EMOJIS.NO_PLATE} Error processing frame:`, error);
        if (error instanceof Error) {
            logger.error(`${EMOJIS.NO_PLATE} Stack trace:`, error.stack);
        }
    }
}

/**
 * Starts processing the RTSP stream.
 * @param retries Number of retry attempts.
 * @param delay Delay between retries in milliseconds.
 */
async function startStreamProcessing(retries = CONFIG.MAX_RETRIES, delay = CONFIG.RETRY_DELAY) {
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

let isShuttingDown = false;

/**
 * Cleans up resources before shutting down.
 */
async function cleanup() {
    if (isShuttingDown) return;
    isShuttingDown = true;

    logger.info(`${EMOJIS.NO_PLATE} Cleaning up...`);

    if (worker) {
        await worker.terminate();
        logger.info(`${EMOJIS.INIT} OCR Worker terminated`);
    }

    if (fs.existsSync(TEMP_DIR)) {
        await fs.promises.rm(TEMP_DIR, { recursive: true, force: true });
        logger.info(`${EMOJIS.INIT} Temporary directory deleted: ${TEMP_DIR}`);
    }

    tf.disposeVariables();

    process.exit(0);
}

// Handle unhandled promise rejections
process.on('unhandledRejection', (reason, promise) => {
    logger.error(`${EMOJIS.NO_PLATE} Unhandled Rejection at:`, promise);
    if (reason instanceof Error) {
        logger.error(`${EMOJIS.NO_PLATE} Stack trace:`, reason.stack);
    } else {
        logger.error(`${EMOJIS.NO_PLATE} Reason:`, reason);
    }
    process.exit(1);
});

// Handle process exit
process.on('exit', (code) => {
    logger.info(`${EMOJIS.NO_PLATE} Process is exiting with code: ${code}`);
});

// Handle uncaught exceptions
process.on('uncaughtException', (error) => {
    logger.error(`${EMOJIS.NO_PLATE} Uncaught exception:`, error);
    logger.error(`${EMOJIS.NO_PLATE} Stack trace:`, error.stack);
    cleanup();
    process.exit(1);
});

const HEARTBEAT_INTERVAL = 10000; // 10 seconds

// Heartbeat to indicate the application is running
setInterval(() => {
    logger.info(`${EMOJIS.INIT} Heartbeat: Application is running...`);
}, HEARTBEAT_INTERVAL);

// Memory statistics logging
setInterval(() => {
    const memInfo = tf.memory();
    logger.info(
        `${EMOJIS.DEBUG} TensorFlow Memory Stats:
        Active Tensors: ${memInfo.numTensors}
        Data Buffers: ${memInfo.numDataBuffers}
        Memory Used: ${(memInfo.numBytes / 1024 / 1024).toFixed(2)} MB
        Memory State: ${memInfo.unreliable ? 'Unreliable' : 'Reliable'}`
    );
}, HEARTBEAT_INTERVAL);

/**
 * Logs the application configuration.
 * @param config Configuration object.
 */
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
    logger.info(`${EMOJIS.INIT} Frame Queue Size: ${config.FRAME_QUEUE_SIZE}`);
    logger.info(`${EMOJIS.INIT} ----------------------------------------`);
}

/**
 * Sanitizes the FFmpeg command to hide sensitive information.
 * @param cmd FFmpeg command line.
 * @returns Sanitized command line.
 */
function sanitizeFFmpegCommand(cmd: string): string {
    const rtspUrlRegex = /(-i\s+)(rtsp:\/\/[^@]+@[^\s]+)/;
    const sanitizedCmd = cmd.replace(rtspUrlRegex, (match, p1, p2) => {
        try {
            const url = new URL(p2);
            return `${p1}rtsp://*****:*****@${url.hostname}${url.pathname}`;
        } catch {
            return match; // If URL parsing fails, return original match
        }
    });
    return sanitizedCmd;
}

/**
 * Starts the application by initializing components and starting stream processing.
 */
(async () => {
    await initialise();
    logConfiguration(CONFIG);
    startStreamProcessing();
})();
