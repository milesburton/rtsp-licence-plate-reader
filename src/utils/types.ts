export interface Region {
    x: number;
    y: number;
    width: number;
    height: number;
}

export interface DetectionConfig {
    minArea: number;
    maxArea: number;
    minAspectRatio: number;
    maxAspectRatio: number;
}

export interface PlateDetection {
    text: string;
    confidence: number;
    frameNumber: number;
    timestamp: number;
}