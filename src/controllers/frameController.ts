import sharp from 'sharp';
import logger from '../utils/logger';
import { detectPeople, detectVehicles } from '../utils/processing';
import type { Region, PlateDetection } from '../utils/types';
import type { Worker } from 'tesseract.js';

/**
 * Processes an image buffer by detecting people and vehicles, then performing OCR on detected vehicles.
 *
 * @param imageBuffer - The buffer of the image data.
 * @param metadata - Metadata associated with the frame.
 * @param ocrWorker - The Tesseract.js worker instance.
 * @returns An array of detected license plates.
 */
export async function processFrame(
    imageBuffer: Buffer,
    metadata: { timestamp: number; frameNumber: number },
    ocrWorker: Worker
): Promise<PlateDetection[]> {
    try {
        logger.info(`📄 Processing frame #${metadata.frameNumber} at ${new Date(metadata.timestamp).toISOString()}`);

        // Detect people
        const people = await detectPeople(imageBuffer);
        if (people.length > 0) {
            logger.info(`👥 Detected ${people.length} people in frame #${metadata.frameNumber}.`);
        }

        // Detect vehicles
        const vehicles = await detectVehicles(imageBuffer);
        logger.info(`🚗 Detected ${vehicles.length} vehicles in frame #${metadata.frameNumber}.`);

        if (vehicles.length === 0 && people.length === 0) {
            logger.info(`🛑 No objects detected in frame #${metadata.frameNumber}.`);
            return [];
        }

        // Perform OCR on detected vehicles
        const plateDetections: PlateDetection[] = [];
        for (const vehicle of vehicles) {
            try {
                const vehicleBuffer = await extractRegionBuffer(imageBuffer, vehicle);
                const plates = await performOCR(vehicleBuffer, ocrWorker);
                if (plates.length > 0) {
                    plates.forEach(plate => {
                        plateDetections.push({
                            text: plate.text,
                            confidence: plate.confidence,
                            frameNumber: metadata.frameNumber,
                            timestamp: metadata.timestamp,
                        });
                    });
                    logger.info(`🔍 Detected license plates in frame #${metadata.frameNumber}: ${plates.map(p => `${p.text} (${p.confidence}%)`).join(', ')}`);
                }
            } catch (error) {
                logger.error(`❌ Error processing vehicle region in frame #${metadata.frameNumber}: ${error}`);
            }
        }

        if (plateDetections.length === 0) {
            logger.info(`🔎 No license plates detected in frame #${metadata.frameNumber}.`);
        }

        return plateDetections;

    } catch (error) {
        logger.error(`❌ Error processing frame #${metadata.frameNumber}: ${error}`);
        return [];
    }
}

/**
 * Extracts a specific region from the image buffer.
 *
 * @param imageBuffer - The buffer of the original image.
 * @param region - The region to extract.
 * @returns A buffer of the extracted region.
 */
async function extractRegionBuffer(imageBuffer: Buffer, region: Region): Promise<Buffer> {
    try {
        const extractedBuffer = await sharp(imageBuffer)
            .extract({ left: region.x, top: region.y, width: region.width, height: region.height })
            .toBuffer();
        logger.debug(`✂️ Extracted region: ${JSON.stringify(region)}`);
        return extractedBuffer;
    } catch (error) {
        logger.error(`❌ Error extracting region: ${error}`);
        throw error;
    }
}

/**
 * Performs OCR on the given image buffer to detect license plates.
 *
 * @param imageBuffer - The buffer of the vehicle image.
 * @param ocrWorker - The Tesseract.js worker instance.
 * @returns An array of detected license plates with their confidence scores.
 */
export async function performOCR(
    imageBuffer: Buffer,
    ocrWorker: Worker
): Promise<{ text: string; confidence: number }[]> {
    try {
        logger.info(`🔍 Starting OCR on image buffer.`);
        const { data } = await ocrWorker.recognize(imageBuffer);
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