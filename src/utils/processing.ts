import sharp from 'sharp';
import type { Region, DetectionConfig, PlateDetection } from './types';
import logger from './logger';
import { findRegions, filterOverlappingRegions } from './regionUtils';
import CONFIG from '../config';
import { createWorker, Worker } from 'tesseract.js';

/**
 * The Tesseract OCR Worker instance.
 */
let worker: Worker | null = null;

/**
 * Initializes the Tesseract OCR worker.
 */
export async function initialiseWorker(): Promise<void> {
    try {
        worker = await createWorker({
            logger: m => logger.info(`[Tesseract]: ${m.status} (${Math.round(m.progress * 100)}%)`)
        });
        await worker.loadLanguage('eng');
        await worker.initialize('eng');
        logger.info('✅ Tesseract OCR Worker initialized.');
    } catch (error) {
        logger.error(`❌ Failed to initialize Tesseract OCR Worker: ${error}`);
        throw error;
    }
}

/**
 * Terminates the Tesseract OCR worker.
 */
export async function terminateWorker(): Promise<void> {
    try {
        if (worker) {
            await worker.terminate();
            logger.info('✅ Tesseract OCR Worker terminated.');
            worker = null;
        }
    } catch (error) {
        logger.error(`❌ Failed to terminate Tesseract OCR Worker: ${error}`);
    }
}

/**
 * Detects people in the given image buffer using image processing techniques.
 *
 * @param imageBuffer - The buffer of the image data.
 * @returns An array of detected regions representing people.
 */
export async function detectPeople(imageBuffer: Buffer): Promise<Region[]> {
    try {
        logger.info(`🔍 Starting person detection.`);
        const { data, info } = await sharp(imageBuffer)
            .grayscale()
            .blur(1.5)
            .threshold(140)
            .raw()
            .toBuffer({ resolveWithObject: true });

        const config: DetectionConfig = {
            minArea: CONFIG.MIN_PERSON_AREA,
            maxArea: CONFIG.MAX_PERSON_AREA,
            minAspectRatio: 0.25,
            maxAspectRatio: 0.7
        };

        const regions = findRegions(data, info.width, info.height, config);
        const filteredRegions = filterOverlappingRegions(regions);

        if (CONFIG.DEBUG_MODE && filteredRegions.length > 0) {
            await saveDebugImage(imageBuffer, filteredRegions, info.width, info.height, 'green', 'detected_people');
            logger.debug(`🖼️ Saved debug image for detected people.`);
        }

        logger.info(`👥 Detected ${filteredRegions.length} people.`);
        return filteredRegions;
    } catch (error) {
        logger.error(`❌ Error in person detection: ${error}`);
        return [];
    }
}

/**
 * Detects vehicles in the given image buffer using image processing techniques.
 *
 * @param imageBuffer - The buffer of the image data.
 * @returns An array of detected regions representing vehicles.
 */
export async function detectVehicles(imageBuffer: Buffer): Promise<Region[]> {
    try {
        logger.info(`🚗 Starting vehicle detection.`);
        const { data, info } = await sharp(imageBuffer)
            .grayscale()
            .blur(2)
            .threshold(128)
            .raw()
            .toBuffer({ resolveWithObject: true });

        const config: DetectionConfig = {
            minArea: CONFIG.MIN_VEHICLE_AREA,
            maxArea: CONFIG.MAX_VEHICLE_AREA,
            minAspectRatio: 0.5,
            maxAspectRatio: 2.5
        };

        const regions = findRegions(data, info.width, info.height, config);
        const filteredRegions = filterOverlappingRegions(regions);

        if (CONFIG.DEBUG_MODE && filteredRegions.length > 0) {
            await saveDebugImage(imageBuffer, filteredRegions, info.width, info.height, 'red', 'detected_vehicles');
            logger.debug(`🖼️ Saved debug image for detected vehicles.`);
        }

        logger.info(`🚗 Detected ${filteredRegions.length} vehicles.`);
        return filteredRegions;
    } catch (error) {
        logger.error(`❌ Error in vehicle detection: ${error}`);
        return [];
    }
}

/**
 * Saves a debug image with annotated regions.
 *
 * @param imageBuffer - The buffer of the original image.
 * @param regions - The regions to annotate.
 * @param width - The width of the image.
 * @param height - The height of the image.
 * @param color - The color of the annotation rectangles.
 * @param prefix - The prefix for the debug image filename.
 */
async function saveDebugImage(
    imageBuffer: Buffer,
    regions: Region[],
    width: number,
    height: number,
    color: string,
    prefix: string
): Promise<void> {
    try {
        const svgRects = regions.map(r => `
            <rect x="${r.x}" y="${r.y}" 
                  width="${r.width}" height="${r.height}"
                  fill="none" stroke="${color}" stroke-width="3"/>
        `).join('');

        const svgBuffer = Buffer.from(`
            <svg width="${width}" height="${height}">
                ${svgRects}
            </svg>
        `);

        const annotatedImage = sharp(imageBuffer)
            .composite([{
                input: svgBuffer,
                top: 0,
                left: 0
            }])
            .toFormat('jpg');

        const debugDir = 'debug_output';
        await sharp('').toFile(path.join(debugDir, '.gitkeep')); // Ensure directory exists
        const debugPath = path.join(debugDir, `${prefix}_${Date.now()}.jpg`);
        await annotatedImage.toFile(debugPath);

        logger.info(`🖼️ Debug image saved: ${debugPath}`);
    } catch (error) {
        logger.error(`❌ Error saving debug image: ${error}`);
    }
}

/**
 * Performs OCR on the given image buffer to detect license plates.
 *
 * @param imageBuffer - The buffer of the vehicle image.
 * @returns An array of detected license plates with their confidence scores.
 */
export async function performOCR(imageBuffer: Buffer): Promise<PlateDetection[]> {
    try {
        if (!worker) {
            throw new Error('OCR Worker not initialized.');
        }
        logger.info(`🔍 Starting OCR on image buffer.`);
        const { data } = await worker.recognize(imageBuffer);
        const words = data.words.map(word => ({
            text: word.text,
            confidence: word.confidence
        })).filter(word => {
            return Object.values(CONFIG.PLATE_PATTERNS).some(pattern => 
                pattern.test(word.text.toUpperCase())
            );
        });

        logger.info(`🔍 Detected ${words.length} license plates.`);
        return words;
    } catch (error) {
        logger.error(`❌ Error performing OCR: ${error}`);
        return [];
    }
}
