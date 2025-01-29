export enum VehicleType {
    Car = "Car",
    Motorcycle = "Motorcycle",
    Bus = "Bus",
    Van = "Van",
    Truck = "Truck",
    Bicycle = "Bicycle",
    Scooter = "Scooter",
    Unknown = "Unknown",
}

export interface Region {
    x: number;
    y: number;
    width: number;
    height: number;
    confidence: number;
}

export interface VehicleRegion extends Region {
    type: VehicleType;
}

export interface DetectionConfig {
    minArea: number;
    maxArea: number;
    minAspectRatio: number;
    maxAspectRatio: number;
}

export interface OCRWord {
    text: string;
    confidence: number;
}

export interface OCRResult {
    data: {
        text: string;
        words?: Array<{
            text: string;
            confidence: number;
            bbox?: [number, number, number, number];
        }>;
    };
}
