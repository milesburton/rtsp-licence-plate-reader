import * as tf from '@tensorflow/tfjs-node';
import sharp from 'sharp';
import { Region, DetectionConfig } from '../types';

export async function loadImageAsTensor(imagePath: string): Promise<tf.Tensor3D> {
    const { data, info } = await sharp(imagePath)
        .resize(640, 480)
        .toFormat('png')
        .removeAlpha()
        .raw()
        .toBuffer({ resolveWithObject: true });

    return tf.tensor(new Uint8Array(data), [info.height, info.width, 3]);
}

export function findRegions(data: Buffer, width: number, height: number, config: DetectionConfig): Region[] {
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

export function floodFill(data: Buffer, width: number, height: number, x: number, y: number, visited: Set<number>): Set<number> {
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

export function filterOverlappingRegions(regions: Region[]): Region[] {
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

export function calculateOverlap(r1: Region, r2: Region): number {
    const xOverlap = Math.max(0, Math.min(r1.x + r1.width, r2.x + r2.width) - Math.max(r1.x, r2.x));
    const yOverlap = Math.max(0, Math.min(r1.y + r1.height, r2.y + r2.height) - Math.max(r1.y, r2.y));
    const overlapArea = xOverlap * yOverlap;
    const r1Area = r1.width * r1.height;
    const r2Area = r2.width * r2.height;
    return overlapArea / Math.min(r1Area, r2Area);
}

export function getBoundingBox(region: Set<number>, width: number): Region {
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
