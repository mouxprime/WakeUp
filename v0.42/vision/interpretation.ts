/*
 * Interprétation haut-niveau des sorties des modèles BlazePalm / HandLandmarks.
 */

export interface Anchor {
  xCenter: number;
  yCenter: number;
  width: number;
  height: number;
}

export interface MLRawDetectorOutput {
  rawBoxes: Float32Array; // shape [numAnchors, 18]
  rawScores: Float32Array; // shape [numAnchors, 1]
}

export interface MLRawLandmarksOutput {
  landmarks: Float32Array; // 21 * 3 valeurs
  handedness: 'Left' | 'Right';
  handednessScore: number;
  presenceScore?: number;
  visibility?: Float32Array;
  landmarkScore?: number;
}

export interface FrameMeta {
  frameWidth: number;
  frameHeight: number;
  modelInputWidth: number;
  modelInputHeight: number;
  frameId?: number;
}

export interface Point2D {
  x: number;
  y: number;
}

export interface Landmark3D extends Point2D {
  z: number;
}

export interface DetectionBox {
  id: number;
  score: number;
  cx: number;
  cy: number;
  w: number;
  h: number;
  rotation: number;
  palmKeypoints: Point2D[];
}

export interface DetectionBoxOnFrame extends DetectionBox {
  cxPx: number;
  cyPx: number;
  wPx: number;
  hPx: number;
  xMin: number;
  yMin: number;
  xMax: number;
  yMax: number;
  palmKeypointsPx: Point2D[];
}

export interface HandOrientation {
  palmNormal: [number, number, number];
  fingerDirection: [number, number, number];
  yaw: number;
  pitch: number;
  roll: number;
}

export type FingerName = 'Thumb' | 'Index' | 'Middle' | 'Ring' | 'Pinky';
export type FingerState = 'Open' | 'Closed' | 'Pointing' | 'Unknown';

export interface HandPose {
  fingers: Record<FingerName, FingerState>;
  isPinching: boolean;
  pinchStrength: number;
  spread: number;
}

export interface HandSpatialInfo {
  center: { x: number; y: number };
  normalizedCenter: { x: number; y: number };
  size: { w: number; h: number };
  depthApprox: number;
  isNearCenter: boolean;
}

export interface HandState {
  id: string;
  handedness: 'Left' | 'Right';
  handednessScore: number;
  qualityScore: number;
  detectionBox: DetectionBoxOnFrame;
  landmarks: Landmark3D[];
  orientation: HandOrientation;
  pose: HandPose;
  spatial: HandSpatialInfo;
}

export type InterHandRelationType =
  | 'HandshakeCandidate'
  | 'HandsClose'
  | 'Crossing'
  | 'Unknown';

export interface InterHandRelation {
  handIdA: string;
  handIdB: string;
  type: InterHandRelationType;
  distancePx: number;
  confidence: number;
}

export interface SceneInterpretation {
  frameId: number;
  hands: HandState[];
  interHandRelations: InterHandRelation[];
  rawMeta: {
    numDetectionsBeforeNMS: number;
    processingTimeMs?: number;
    frameWidth: number;
    frameHeight: number;
  };
}

interface DetectionParams {
  scoreThreshold: number;
  maxHands: number;
  iouThreshold: number;
  roiExpansion: number;
}

export interface PreparedDetections {
  boxes: DetectionBox[];
  boxesOnFrame: DetectionBoxOnFrame[];
  roiBoxesOnFrame: DetectionBoxOnFrame[];
  numDetectionsBeforeNMS: number;
}

export interface InterpretFrameOptions {
  precomputed?: PreparedDetections;
  detectionParams?: Partial<DetectionParams>;
  includePerformanceMetrics?: boolean;
}

const PALM_KEYPOINT_COUNT = 7;
const DETECTION_VALUES_PER_ANCHOR = 18;
const DEFAULT_DETECTION_PARAMS: DetectionParams = {
  scoreThreshold: 0.55,
  maxHands: 2,
  iouThreshold: 0.3,
  roiExpansion: 1.25,
};

const DETECTOR_SCALE = 192.0; // x/y/w/h scale
const DEFAULT_FRAME_ID = 0;
const EPSILON = 1e-6;

interface AnchorGeneratorOptions {
  inputWidth: number;
  inputHeight: number;
  minScale: number;
  maxScale: number;
  strides: number[];
  anchorOffsetX: number;
  anchorOffsetY: number;
  aspectRatios: number[];
  interpolatedScaleAspectRatio: number;
  fixedAnchorSize: boolean;
}

export function createPalmDetectionAnchors(): Anchor[] {
  const options: AnchorGeneratorOptions = {
    inputWidth: 192,
    inputHeight: 192,
    minScale: 0.1484375,
    maxScale: 0.75,
    strides: [8, 16, 16, 16],
    anchorOffsetX: 0.5,
    anchorOffsetY: 0.5,
    aspectRatios: [1.0],
    interpolatedScaleAspectRatio: 1.0,
    fixedAnchorSize: true,
  };

  const anchors: Anchor[] = [];
  const numLayers = options.strides.length;

  for (let layer = 0; layer < numLayers; layer++) {
    const stride = options.strides[layer];
    const fmHeight = Math.ceil(options.inputHeight / stride);
    const fmWidth = Math.ceil(options.inputWidth / stride);
    const scale =
      options.minScale +
      ((options.maxScale - options.minScale) * layer) / Math.max(1, numLayers - 1);
    const scaleNext =
      layer === numLayers - 1
        ? 1.0
        : options.minScale +
          ((options.maxScale - options.minScale) * (layer + 1)) /
            Math.max(1, numLayers - 1);

    for (let y = 0; y < fmHeight; y++) {
      for (let x = 0; x < fmWidth; x++) {
        const xCenter = (x + options.anchorOffsetX) / fmWidth;
        const yCenter = (y + options.anchorOffsetY) / fmHeight;

        for (const ratio of options.aspectRatios) {
          const ratioSqrt = Math.sqrt(ratio);
          const anchorHeight = options.fixedAnchorSize ? 1 : scale / ratioSqrt;
          const anchorWidth = options.fixedAnchorSize ? 1 : scale * ratioSqrt;
          anchors.push({ xCenter, yCenter, width: anchorWidth, height: anchorHeight });
        }

        if (options.interpolatedScaleAspectRatio > 0) {
          const ratio = options.interpolatedScaleAspectRatio;
          const ratioSqrt = Math.sqrt(ratio);
          const interpolatedScale = Math.sqrt(scale * scaleNext);
          const anchorHeight = options.fixedAnchorSize ? 1 : interpolatedScale / ratioSqrt;
          const anchorWidth = options.fixedAnchorSize ? 1 : interpolatedScale * ratioSqrt;
          anchors.push({ xCenter, yCenter, width: anchorWidth, height: anchorHeight });
        }
      }
    }
  }
  return anchors;
}

export const palmDetectionAnchors: Anchor[] = createPalmDetectionAnchors();

function mergeDetectionParams(overrides?: Partial<DetectionParams>): DetectionParams {
  'worklet';
  return {
    scoreThreshold: overrides?.scoreThreshold ?? DEFAULT_DETECTION_PARAMS.scoreThreshold,
    maxHands: overrides?.maxHands ?? DEFAULT_DETECTION_PARAMS.maxHands,
    iouThreshold: overrides?.iouThreshold ?? DEFAULT_DETECTION_PARAMS.iouThreshold,
    roiExpansion: overrides?.roiExpansion ?? DEFAULT_DETECTION_PARAMS.roiExpansion,
  };
}

// ---- Helpers worklet globaux utilisés plus loin ----

function getNow(): number {
  'worklet';
  return typeof performance !== 'undefined' && typeof performance.now === 'function'
    ? performance.now()
    : Date.now();
}

function mapBoxToFrame(box: DetectionBox, meta: FrameMeta): DetectionBoxOnFrame {
  'worklet';
  const cxPx = box.cx * meta.frameWidth;
  const cyPx = box.cy * meta.frameHeight;
  const wPx = box.w * meta.frameWidth;
  const hPx = box.h * meta.frameHeight;
  const xMin = cxPx - wPx / 2;
  const yMin = cyPx - hPx / 2;
  const xMax = cxPx + wPx / 2;
  const yMax = cyPx + hPx / 2;

  return {
    ...box,
    cxPx,
    cyPx,
    wPx,
    hPx,
    xMin,
    yMin,
    xMax,
    yMax,
    palmKeypointsPx: box.palmKeypoints.map((kp) => ({
      x: kp.x * meta.frameWidth,
      y: kp.y * meta.frameHeight,
    })),
  };
}

function expandRoiForLandmarks(
  box: DetectionBoxOnFrame,
  meta: FrameMeta,
  expansion: number,
): DetectionBoxOnFrame {
  'worklet';
  const size = Math.max(box.wPx, box.hPx) * expansion;
  const half = size / 2;
  const cx = box.cxPx;
  const cy = box.cyPx;
  const xMin = Math.min(Math.max(cx - half, 0), meta.frameWidth - 1);
  const yMin = Math.min(Math.max(cy - half, 0), meta.frameHeight - 1);
  const xMax = Math.min(Math.max(cx + half, 0), meta.frameWidth - 1);
  const yMax = Math.min(Math.max(cy + half, 0), meta.frameHeight - 1);

  return {
    ...box,
    wPx: xMax - xMin,
    hPx: yMax - yMin,
    xMin,
    yMin,
    xMax,
    yMax,
  };
}

function mapLandmarksToFrame(
  raw: MLRawLandmarksOutput,
  roi: DetectionBoxOnFrame,
): Landmark3D[] {
  'worklet';
  const results: Landmark3D[] = [];
  const { landmarks } = raw;
  for (let i = 0; i < 21; i++) {
    const base = i * 3;
    const xNorm = landmarks[base + 0];
    const yNorm = landmarks[base + 1];
    const zRel = landmarks[base + 2];

    const limitedX = Math.min(Math.max(xNorm, -0.5), 1.5);
    const limitedY = Math.min(Math.max(yNorm, -0.5), 1.5);
    results.push({
      x: roi.xMin + limitedX * roi.wPx,
      y: roi.yMin + limitedY * roi.hPx,
      z: zRel,
    });
  }
  return results;
}

function visibilityReduceMean(values: Float32Array): number {
  'worklet';
  let sum = 0;
  for (let i = 0; i < values.length; i++) {
    sum += values[i];
  }
  const mean = sum / Math.max(1, values.length);
  return Math.min(Math.max(mean, 0), 1);
}

function computeHandQuality(
  detectionScore: number,
  landmarkScore: number,
  visibility?: Float32Array,
): number {
  'worklet';
  const visMean = visibility && visibility.length > 0
    ? visibilityReduceMean(visibility)
    : 1;
  const score = 0.5 * detectionScore + 0.3 * landmarkScore + 0.2 * visMean;
  return Math.min(Math.max(score, 0), 1);
}

const FINGER_INDICES: Record<FingerName, [number, number, number, number]> = {
  Thumb: [1, 2, 3, 4],
  Index: [5, 6, 7, 8],
  Middle: [9, 10, 11, 12],
  Ring: [13, 14, 15, 16],
  Pinky: [17, 18, 19, 20],
};

function subtractVec(a: Landmark3D, b: Landmark3D): [number, number, number] {
  'worklet';
  return [a.x - b.x, a.y - b.y, (a.z ?? 0) - (b.z ?? 0)];
}

function crossVec(a: [number, number, number], b: [number, number, number]): [number, number, number] {
  'worklet';
  return [
    a[1] * b[2] - a[2] * b[1],
    a[2] * b[0] - a[0] * b[2],
    a[0] * b[1] - a[1] * b[0],
  ];
}

function normalizeVec(vec: [number, number, number]): [number, number, number] {
  'worklet';
  const length = Math.hypot(vec[0], vec[1], vec[2]);
  if (length < EPSILON) return [0, 0, 1];
  return [vec[0] / length, vec[1] / length, vec[2] / length];
}

function avgVec(vectors: [number, number, number][]): [number, number, number] {
  'worklet';
  if (!vectors.length) return [0, 0, 1];
  const sum = vectors.reduce(
    (acc, vec) => [acc[0] + vec[0], acc[1] + vec[1], acc[2] + vec[2]],
    [0, 0, 0],
  );
  return [sum[0] / vectors.length, sum[1] / vectors.length, sum[2] / vectors.length];
}

function angleBetween(a: [number, number, number], b: [number, number, number]): number {
  'worklet';
  const normA = Math.hypot(a[0], a[1], a[2]);
  const normB = Math.hypot(b[0], b[1], b[2]);
  if (normA < EPSILON || normB < EPSILON) return 0;
  const dot = a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
  return Math.acos(Math.min(Math.max(dot / (normA * normB), -1), 1));
}

function distance2D(a: Point2D, b: Point2D): number {
  'worklet';
  return Math.hypot(a.x - b.x, a.y - b.y);
}

function computeHandPose(landmarks: Landmark3D[]): HandPose {
  'worklet';
  const fingers: Record<FingerName, FingerState> = {
    Thumb: 'Unknown',
    Index: 'Unknown',
    Middle: 'Unknown',
    Ring: 'Unknown',
    Pinky: 'Unknown',
  };

  const palmSize = distance2D(landmarks[0], landmarks[9]) + distance2D(landmarks[0], landmarks[13]);
  const palmSpan = Math.max(palmSize / 2, 1);

  (Object.keys(FINGER_INDICES) as FingerName[]).forEach((finger) => {
    const [mcpIdx, pipIdx, dipIdx, tipIdx] = FINGER_INDICES[finger];
    const mcp = landmarks[mcpIdx];
    const pip = landmarks[pipIdx];
    const dip = landmarks[dipIdx];
    const tip = landmarks[tipIdx];

    const vec1 = subtractVec(mcp, pip);
    const vec2 = subtractVec(pip, tip);
    const angle = angleBetween(vec1, vec2);
    if (!isFinite(angle)) {
      fingers[finger] = 'Unknown';
      return;
    }
    const angleDeg = (angle * 180) / Math.PI;
    if (angleDeg < 55) {
      fingers[finger] = 'Open';
    } else if (angleDeg < 75) {
      fingers[finger] = 'Pointing';
    } else {
      fingers[finger] = 'Closed';
    }
  });

  const thumbTip = landmarks[4];
  const indexTip = landmarks[8];
  const pinchDistance = distance2D(thumbTip, indexTip);
  const pinchStrength = Math.min(
    Math.max(1 - pinchDistance / (palmSpan * 0.8), 0),
    1,
  );
  const isPinching = pinchStrength > 0.6;

  const spread = Math.min(
    Math.max(distance2D(landmarks[5], landmarks[17]) / palmSize, 0),
    1,
  );

  return {
    fingers,
    isPinching,
    pinchStrength,
    spread,
  };
}

function computeHandSpatialInfo(
  detectionBox: DetectionBoxOnFrame,
  landmarks: Landmark3D[],
  meta: FrameMeta,
): HandSpatialInfo {
  'worklet';
  const normalizedCenter = {
    x: Math.min(Math.max(detectionBox.cxPx / meta.frameWidth, 0), 1),
    y: Math.min(Math.max(detectionBox.cyPx / meta.frameHeight, 0), 1),
  };
  const depthApprox = landmarks.length
    ? landmarks.reduce((sum, point) => sum + point.z, 0) / landmarks.length
    : 0;
  const center = { x: detectionBox.cxPx, y: detectionBox.cyPx };
  const size = { w: detectionBox.wPx, h: detectionBox.hPx };
  const isNearCenter = Math.hypot(normalizedCenter.x - 0.5, normalizedCenter.y - 0.5) < 0.25;

  return {
    center,
    normalizedCenter,
    size,
    depthApprox,
    isNearCenter,
  };
}

function inferInterHandRelations(hands: HandState[]): InterHandRelation[] {
  'worklet';
  const relations: InterHandRelation[] = [];
  if (hands.length < 2) return relations;

  for (let i = 0; i < hands.length; i++) {
    for (let j = i + 1; j < hands.length; j++) {
      const handA = hands[i];
      const handB = hands[j];
      const dx = handA.spatial.center.x - handB.spatial.center.x;
      const dy = handA.spatial.center.y - handB.spatial.center.y;
      const distancePx = Math.hypot(dx, dy);
      const avgSize = (handA.spatial.size.w + handB.spatial.size.w) / 2;
      let type: InterHandRelationType = 'Unknown';
      let confidence = 0;

      if (
        handA.handedness !== handB.handedness &&
        Math.abs(handA.spatial.center.y - handB.spatial.center.y) < avgSize * 0.4 &&
        distancePx < avgSize * 1.6
      ) {
        type = 'HandshakeCandidate';
        confidence = 0.8;
      } else if (distancePx < avgSize * 1.2) {
        type = 'HandsClose';
        confidence = 0.6;
      } else if (Math.abs(dx) < avgSize * 0.5) {
        type = 'Crossing';
        confidence = 0.4;
      }

      relations.push({
        handIdA: handA.id,
        handIdB: handB.id,
        type,
        distancePx,
        confidence,
      });
    }
  }

  return relations;
}

// --------- Préparation des détections (decode + NMS) ---------

export function prepareDetections(
  detectorOutput: MLRawDetectorOutput,
  anchors: Anchor[],
  meta: FrameMeta,
  overrides?: Partial<DetectionParams>,
): PreparedDetections {
  'worklet';
  const params = mergeDetectionParams(overrides);

  function decodeDetectionsInternal(
    detectorOutputInner: MLRawDetectorOutput,
    anchorsInner: Anchor[],
  ): DetectionBox[] {
    'worklet';
    const boxes: DetectionBox[] = [];
    const { rawBoxes, rawScores } = detectorOutputInner;
    const anchorsCount = Math.min(
      Math.floor(rawBoxes.length / DETECTION_VALUES_PER_ANCHOR),
      anchorsInner.length,
    );

    for (let i = 0; i < anchorsCount; i++) {
      const rawScore = rawScores[i] ?? rawScores[i * 1] ?? 0;
      const score =
        rawScore >= 0
          ? 1 / (1 + Math.exp(-rawScore))
          : Math.exp(rawScore) / (1 + Math.exp(rawScore));
      const boxOffset = i * DETECTION_VALUES_PER_ANCHOR;
      const anchor = anchorsInner[i];

      const yCenter = rawBoxes[boxOffset + 0];
      const xCenter = rawBoxes[boxOffset + 1];
      const h = rawBoxes[boxOffset + 2];
      const w = rawBoxes[boxOffset + 3];

      const cx = anchor.xCenter + (xCenter / DETECTOR_SCALE) * anchor.width;
      const cy = anchor.yCenter + (yCenter / DETECTOR_SCALE) * anchor.height;
      const decodedW = anchor.width * Math.exp(w / DETECTOR_SCALE);
      const decodedH = anchor.height * Math.exp(h / DETECTOR_SCALE);

      const palmKeypoints: Point2D[] = [];
      for (let k = 0; k < PALM_KEYPOINT_COUNT; k++) {
        const kpX = rawBoxes[boxOffset + 4 + k * 2 + 1];
        const kpY = rawBoxes[boxOffset + 4 + k * 2];
        palmKeypoints.push({
          x: anchor.xCenter + (kpX / DETECTOR_SCALE) * anchor.width,
          y: anchor.yCenter + (kpY / DETECTOR_SCALE) * anchor.height,
        });
      }

      const rotation =
        palmKeypoints.length >= 2
          ? Math.atan2(
              palmKeypoints[palmKeypoints.length - 1].y - palmKeypoints[0].y,
              palmKeypoints[palmKeypoints.length - 1].x - palmKeypoints[0].x,
            )
          : 0;

      boxes.push({
        id: i,
        score,
        cx: Math.min(Math.max(cx, 0), 1),
        cy: Math.min(Math.max(cy, 0), 1),
        w: Math.min(Math.max(decodedW, 0), 1),
        h: Math.min(Math.max(decodedH, 0), 1),
        rotation,
        palmKeypoints,
      });
    }

    return boxes;
  }

  const decoded = decodeDetectionsInternal(detectorOutput, anchors);
  const filtered = decoded.filter((box) => box.score >= params.scoreThreshold);
  const sorted = [...filtered].sort((a, b) => b.score - a.score);
  const selected: DetectionBox[] = [];

  const intersectionOverUnionLocal = (a: DetectionBox, b: DetectionBox): number => {
    'worklet';
    const axMin = a.cx - a.w / 2;
    const ayMin = a.cy - a.h / 2;
    const axMax = a.cx + a.w / 2;
    const ayMax = a.cy + a.h / 2;

    const bxMin = b.cx - b.w / 2;
    const byMin = b.cy - b.h / 2;
    const bxMax = b.cx + b.w / 2;
    const byMax = b.cy + b.h / 2;

    const interXMin = Math.max(axMin, bxMin);
    const interYMin = Math.max(ayMin, byMin);
    const interXMax = Math.min(axMax, bxMax);
    const interYMax = Math.min(ayMax, byMax);

    const interArea =
      Math.max(0, interXMax - interXMin) * Math.max(0, interYMax - interYMin);
    const areaA = Math.max(0, axMax - axMin) * Math.max(0, ayMax - ayMin);
    const areaB = Math.max(0, bxMax - bxMin) * Math.max(0, byMax - byMin);

    const union = areaA + areaB - interArea;
    if (union <= 0) return 0;
    return interArea / union;
  };

  for (const candidate of sorted) {
    let overlaps = false;
    for (const chosen of selected) {
      if (intersectionOverUnionLocal(candidate, chosen) > params.iouThreshold) {
        overlaps = true;
        break;
      }
    }
    if (!overlaps) {
      selected.push(candidate);
    }
    if (selected.length >= params.maxHands) break;
  }

  const boxesOnFrame = selected.map((box) => mapBoxToFrame(box, meta));
  const roiBoxes = boxesOnFrame.map((box) =>
    expandRoiForLandmarks(box, meta, params.roiExpansion),
  );

  return {
    boxes: selected,
    boxesOnFrame,
    roiBoxesOnFrame: roiBoxes,
    numDetectionsBeforeNMS: decoded.length,
  };
}

// --------- Fonction principale d’interprétation d’un frame ---------
function computeHandOrientation(
  landmarks: Landmark3D[],
  detectionBox: DetectionBoxOnFrame,
): HandOrientation {
  'worklet';
  const wrist = landmarks[0];
  const indexMcp = landmarks[5];
  const middleMcp = landmarks[9];
  const pinkyMcp = landmarks[17];

  const palmVecA = subtractVec(wrist, indexMcp);
  const palmVecB = subtractVec(wrist, pinkyMcp);
  const palmNormal = normalizeVec(crossVec(palmVecA, palmVecB));

  const fingerVecs = [
    subtractVec(landmarks[8], landmarks[5]),
    subtractVec(landmarks[12], landmarks[9]),
    subtractVec(landmarks[16], landmarks[13]),
    subtractVec(landmarks[20], landmarks[17]),
  ];
  const fingerDirection = normalizeVec(avgVec(fingerVecs));

  const roll =
    detectionBox.rotation ||
    Math.atan2(pinkyMcp.y - indexMcp.y, pinkyMcp.x - indexMcp.x);
  const yaw = Math.atan2(fingerDirection[0], fingerDirection[2] || EPSILON);
  const pitch = Math.atan2(
    -fingerDirection[1],
    Math.sqrt(fingerDirection[0] ** 2 + fingerDirection[2] ** 2) || EPSILON,
  );

  return {
    palmNormal,
    fingerDirection,
    yaw,
    pitch,
    roll,
  };
}

export function interpretFrame(
  detectorOutput: MLRawDetectorOutput,
  landmarkOutputsPerHand: MLRawLandmarksOutput[] | null,
  anchors: Anchor[],
  meta: FrameMeta,
  options?: InterpretFrameOptions,
): SceneInterpretation {
  'worklet';
  const startTime = getNow();
  const params = mergeDetectionParams(options?.detectionParams);
  const prepared = options?.precomputed ?? prepareDetections(detectorOutput, anchors, meta, params);

  const hands: HandState[] = [];
  const landmarkOutputs = landmarkOutputsPerHand ?? [];
  const usableCount = Math.min(prepared.roiBoxesOnFrame.length, landmarkOutputs.length);

  for (let i = 0; i < usableCount; i++) {
    const roi = prepared.roiBoxesOnFrame[i];
    const detectionBox = prepared.boxesOnFrame[i];
    const rawLandmarks = landmarkOutputs[i];
    const mappedLandmarks = mapLandmarksToFrame(rawLandmarks, roi);

    const qualityScore = computeHandQuality(
      detectionBox.score,
      rawLandmarks.landmarkScore ?? rawLandmarks.presenceScore ?? rawLandmarks.handednessScore,
      rawLandmarks.visibility,
    );

    const orientation = computeHandOrientation(mappedLandmarks, detectionBox);
    const pose = computeHandPose(mappedLandmarks);
    const spatial = computeHandSpatialInfo(detectionBox, mappedLandmarks, meta);

    const hand: HandState = {
      id: `${meta.frameId ?? DEFAULT_FRAME_ID}-${i}`,
      handedness: rawLandmarks.handedness,
      handednessScore: rawLandmarks.handednessScore,
      qualityScore,
      detectionBox,
      landmarks: mappedLandmarks,
      orientation,
      pose,
      spatial,
    };

    hands.push(hand);
  }

  const interHandRelations = inferInterHandRelations(hands);
  const processingTimeMs = options?.includePerformanceMetrics
    ? getNow() - startTime
    : undefined;

  return {
    frameId: meta.frameId ?? DEFAULT_FRAME_ID,
    hands,
    interHandRelations,
    rawMeta: {
      numDetectionsBeforeNMS: prepared.numDetectionsBeforeNMS,
      processingTimeMs,
      frameWidth: meta.frameWidth,
      frameHeight: meta.frameHeight,
    },
  };
}