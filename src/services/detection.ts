import * as tf from '@tensorflow/tfjs-node';
import * as cocoSsd from '@tensorflow-models/coco-ssd';
import sharp from 'sharp';
import { createWorker, type Worker } from 'tesseract.js';
import * as fs from 'fs';
import * as path from 'path';
import { VehicleType, Region, VehicleRegion, OCRWord } from '../types';
import { logger } from '../utils/logger';
import { CONFIG, TEMP_DIR } from '../config';
import { loadImageAsTensor, findRegions } from '../utils/image-processing';
import { EMOJIS } from '../utils/logger';

let worker: Worker | null = null;
let objectDetectionModel: cocoSsd.ObjectDetection;

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

export async function initialise() {
    if (!fs.existsSync(TEMP_DIR)) {
        try {
            fs.mkdirSync(TEMP_DIR, { recursive: true });
            logger.info(`${EMOJIS.INIT} Created temporary directory at ${TEMP_DIR}`);
        } catch (error) {
            logger.error(`${EMOJIS.NO_PLATE} Failed to create temporary directory:`, error);
            process.exit(1);
        }
    }

    try {
        await tf.ready();
        logger.info(`${EMOJIS.INIT} TensorFlow initialized`);
    } catch (error) {
        logger.error(`${EMOJIS.NO_PLATE} Failed to initialize TensorFlow:`, error);
        process.exit(1);
    }

    try {
        worker = await createWorker('eng');
        logger.info(`${EMOJIS.INIT} OCR Worker initialized`);
    } catch (error) {
        logger.error(`${EMOJIS.NO_PLATE} Failed to initialize OCR Worker:`, error);
        process.exit(1);
    }

    try {
        objectDetectionModel = await cocoSsd.load();
        logger.info(`${EMOJIS.INIT} COCO-SSD Object Detection model loaded`);
    } catch (error) {
        logger.error(`${EMOJIS.NO_PLATE} Failed to load COCO-SSD model:`, error);
        process.exit(1);
    }
}

export async function detectPeople(imagePath: string): Promise<Region[]> {
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

export async function detectVehicles(imagePath: string): Promise<VehicleRegion[]> {
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

        decodedImage.dispose();
        return vehicleRegions;
    } catch (error) {
        logger.error(`${EMOJIS.NO_PLATE} Error in vehicle detection:`, error);
        return [];
    }
}

export async function performOCR(imagePath: string): Promise<OCRWord[]> {
    const startTime = process.hrtime.bigint();
    try {
        if (!worker) {
            logger.error(`${EMOJIS.NO_PLATE} OCR Worker is not initialized`);
            return [];
        }

        const result = await worker.recognize(imagePath);
        logger.debug(`${EMOJIS.DEBUG} OCR processing completed`);

        if (!result.data.text || result.data.text.trim().length === 0) {
            logger.warn(`${EMOJIS.NO_PLATE} No text detected in OCR result.`);
            return [];
        }

        const words = result.data.text
            .split(/\s+/)
            .map(word => word.toUpperCase().replace(/[^A-Z0-9]/g, ''))
            .filter(word => word.length >= 5 && word.length <= 8)
            .map(word => ({
                text: word,
                confidence: result.data.confidence / 100
            }));

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

        if (worker) {
            await worker.terminate();
            try {
                worker = await createWorker('eng');
                logger.info(`${EMOJIS.INIT} OCR Worker reinitialized`);
            } catch (initError) {
                logger.error(`${EMOJIS.NO_PLATE} Failed to reinitialize OCR Worker:`, initError);
                process.exit(1);
            }
        }

        return [];
    }
}

function measureTime(label: string, startTime: number) {
    const endTime = process.hrtime.bigint();
    const duration = Number(endTime - BigInt(startTime)) / 1e6;
    logger.debug(`${EMOJIS.DEBUG} ${label} took ${duration.toFixed(2)}ms`);
}